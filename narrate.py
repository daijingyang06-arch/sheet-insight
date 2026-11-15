"""
M3 · AI 表达层 —— 把统计结果翻译成能直接放进报告的中文
=========================================================
输入：results/discoveries.json（M2 的产物，含精确数字）
输出：results/report.md（成品报告）+ results/charts/*.png + results/narration.json（质检记录）

这一层有两条路，靠 `--backend` 切换：

  template  纯模板生成，**不联网、不需要 API Key**，永远可用。
            它写得比大模型死板，但每个字都来自统计值，绝不会编。
  llm       调大模型（任何 OpenAI 兼容接口：DeepSeek / 智谱 / 通义 / 本地 Ollama 均可）。
            文笔自然，但**每一条输出都必须过数字守卫**，过不了就重试，
            重试还过不了就退回模板。宁可写得平淡，不可写得不准。

这个"降级链"是产品的安全底线：
    LLM 正常 → 用 LLM 文案
    LLM 编数字 → 带着违规清单重试一次
    还是编 / 调用失败 → 用模板文案
    任何情况下，报告都能生成，且数字都对。

用法：
    python narrate.py                        # 自动选后端（有 Key 用 LLM，否则模板）
    python narrate.py --backend template     # 强制离线模板
    python narrate.py --backend llm          # 强制 LLM（无 Key 会报错）
    python narrate.py 你的表格.xlsx           # 指定数据源
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd

import analyze as A
import charts
from number_guard import (allowed_numbers_hint, build_whitelist, check_text)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results")

# 任何 OpenAI 兼容服务都能用；默认给 DeepSeek（便宜、中文好、注册简单）
DEFAULT_BASE_URL = os.environ.get("SHEET_INSIGHT_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_MODEL = os.environ.get("SHEET_INSIGHT_MODEL", "deepseek-chat")
API_KEY_ENV = "SHEET_INSIGHT_API_KEY"

KIND_CN = {
    "correlation": "关联关系",
    "group_diff": "组间差异",
    "trend": "时间趋势",
    "outlier": "异常值",
}


# ============================================================
# 1. 提示词
# ============================================================
SYSTEM_PROMPT = """你是一名中文数据分析写作助手。你的读者是不懂统计学的学生，
他们要把你写的这几句话直接粘贴进课程报告的「结果分析」部分。

必须遵守的五条规则：
1. 只能使用【允许使用的数字】里给出的数字。一个都不许自己计算、换算、相加或推测。
   列表里没有的数字，绝对不许出现——这是硬性要求，违反会被系统打回。
2. 不要提及 p 值、置信区间、显著性水平这类术语，读者看不懂。
3. 不许下因果结论。用「存在相关」「倾向于」「可能与…有关」，禁止用「导致」「造成」。
4. 不要写「建议进一步研究」「数据仅供参考」这类套话。
5. 直接输出结论正文，2~3 句话，不要标题、不要编号、不要「结论：」这样的前缀。

风格要求：像人写报告一样自然，先说发现了什么，再说这个发现意味着什么。"""


def build_messages(discovery, retry_bad=None):
    """
    组装对话消息。retry_bad 不为空时说明上一次输出违规了，
    把违规数字回传给模型——这比单纯"再试一次"有效得多。
    """
    kind = KIND_CN.get(discovery["kind"], discovery["kind"])
    cols = "、".join(f"「{c}」" for c in discovery["columns"])
    nums = "、".join(allowed_numbers_hint(discovery))

    user = (
        f"发现类型：{kind}\n"
        f"涉及字段：{cols}\n"
        f"统计结果原文：{discovery['headline']}\n"
        f"允许使用的数字：{nums}\n"
    )
    if retry_bad:
        user += (
            f"\n【上一次的输出被打回了】它使用了不被允许的数字：{'、'.join(retry_bad)}。\n"
            f"请重写，严格只使用上面列出的数字。如果某个数字没有出现在列表里，就换一种说法绕开它。\n"
        )
    user += "\n请写出 2~3 句结论。"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ============================================================
# 2. 模板生成器（离线兜底）
# ============================================================
def _fmt(v, nd=2):
    """格式化数字。必须和 number_guard._variants 用同样的 round，否则会被自己的守卫拦下。"""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return f"{round(float(v), nd):g}"
    return str(v)


def _strength(r):
    # 与 analyze.corr_strength 共用同一套分级，避免标题说"高度"、正文说"较强"互相打脸
    return {"高度": "很强", "中度": "中等", "低度": "较弱"}[A.corr_strength(r)]


class TemplateNarrator:
    """
    不联网的兜底方案。

    写起来像流水线，但它是这个产品的"最后一道防线"：
    只要统计层没问题，它就永远不会产出错误内容。
    """

    name = "template"
    label = "模板（离线）"

    def narrate(self, messages, discovery=None):      # noqa: ARG002  —— 接口对齐用
        return self._write(discovery)

    def _write(self, d):
        n = d["numbers"]
        kind = d["kind"]

        if kind == "correlation":
            a, b = d["columns"]
            r = n["r"]
            direction = "正" if r > 0 else "负"
            # 注意：正文里只出现 r / n / 均值这几个已经在 numbers 里的数字
            return (
                f"「{a}」与「{b}」之间存在{direction}相关（r = {_fmt(r, 3)}，样本 {_fmt(n['n'], 0)} 条）。"
                f"相关系数 {_fmt(abs(r), 2)} 属于{_strength(r)}程度，"
                f"也就是说{a}较高的样本，{b}倾向于也{'较' + ('高' if r > 0 else '低')}。"
                f"需要说明的是，这只是相关关系，不能说明其中一个直接决定了另一个。"
            )

        if kind == "group_diff":
            cat, num = d["columns"]
            means = n["组均值"]
            hi, lo = n["最高组"], n["最低组"]
            eta2 = n["eta2"]
            pct = _fmt(eta2 * 100, 1)
            return (
                f"不同「{cat}」的「{num}」存在明显差异：「{hi}」组最高"
                f"（平均 {_fmt(means[hi])}），「{lo}」组最低（{_fmt(means[lo])}）。"
                f"效应量 η² = {_fmt(eta2, 4)}，意味着「{cat}」可以解释约 {pct}% 的{num}差异，"
                f"这个差异幅度在实际中是可以感受到的。"
            )

        if kind == "trend":
            date_col, num = d["columns"]
            weekly, total = n["每周变化"], n["全期变化"]
            up = weekly > 0
            return (
                f"「{num}」随时间呈{'上升' if up else '下降'}趋势："
                f"平均每周{'增加' if up else '减少'} {_fmt(abs(weekly))}，"
                f"{n['跨度天数']} 天里累计{'增加' if up else '减少'} {_fmt(abs(total))}。"
                f"这个变化幅度超过了数据本身的日常波动，"
                f"因此更可能是真实趋势，而不是随机起伏。"
            )

        if kind == "outlier":
            num = d["columns"][0]
            return (
                f"「{num}」中有 {n['离群个数']} 个明显偏离整体的值，最极端的是 {_fmt(n['最极端值'])}，"
                f"而中位数只有 {_fmt(n['中位数'])}，偏离了 {_fmt(n['偏离倍数'])} 倍稳健标准差。"
                f"这类值通常是录入错误或极端个案，建议回到原始问卷核对这一条记录，"
                f"确认后再决定是否保留。"
            )

        return d["headline"]


# ============================================================
# 3. LLM 生成器
# ============================================================
class LLMError(Exception):
    pass


class LLMNarrator:
    """
    调用任何 OpenAI 兼容的 /chat/completions 接口。

    用 urllib 而不是 requests/openai SDK——少一个依赖就少一个装不上的理由，
    而且这个调用只有十几行，不值得引库。
    """

    name = "llm"

    def __init__(self, api_key, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
                 timeout=45, temperature=0.3):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    @property
    def label(self):
        return f"大模型（{self.model}）"

    def narrate(self, messages, discovery=None):      # noqa: ARG002
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 400,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:200]
            raise LLMError(f"HTTP {e.code}：{body}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"网络不可达：{e.reason}") from e
        except TimeoutError as e:
            raise LLMError("请求超时") from e

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"返回格式不符合预期：{str(data)[:200]}") from e

        return str(text).strip()


# ============================================================
# 4. 单条发现：生成 + 守卫 + 降级
# ============================================================
def narrate_one(discovery, narrator, max_attempts=2, template=None):
    """
    生成一条发现的结论文字，全程受数字守卫约束。

    返回的记录里保存了 backend / attempts / violations / final_check，
    这些字段是后面算「结论准确率」的证据，也是排查问题时的线索。

    注意：**模板文案同样要过守卫**。
    模板是最后一道防线，如果连它都输出了不认识的数字，说明统计层或模板本身有 bug，
    必须暴露出来而不是默默放过。

    参数 template 可以注入自定义的兜底生成器（验收脚本用它）。
    """
    template = template or TemplateNarrator()
    whitelist = build_whitelist(discovery)
    messages = build_messages(discovery)
    violations, attempts = [], 0

    def finish(rec):
        rec["final_check"] = check_text(rec["text"], whitelist)
        return rec

    # 模板后端没什么可"试"的，直接生成
    if getattr(narrator, "name", "") == "template":
        return finish({
            "text": narrator.narrate(messages, discovery=discovery),
            "backend": "template",
            "attempts": 0,
            "violations": [],
            "fallback_reason": None,
        })

    while attempts < max_attempts:
        attempts += 1
        try:
            text = narrator.narrate(messages, discovery=discovery).strip()
        except LLMError as e:
            return finish({
                "text": template.narrate(messages, discovery=discovery),
                "backend": "template",
                "attempts": attempts,
                "violations": violations,
                "fallback_reason": f"接口调用失败：{e}",
            })

        bad = check_text(text, whitelist)
        if not bad:
            return finish({
                "text": text,
                "backend": getattr(narrator, "name", "llm"),
                "attempts": attempts,
                "violations": violations,
                "fallback_reason": None,
            })

        violations.append({"attempt": attempts, "bad_numbers": bad})
        messages = build_messages(discovery, retry_bad=bad)

    # 两次都编数字 → 退回模板，并记下原因
    return finish({
        "text": template.narrate(messages, discovery=discovery),
        "backend": "template",
        "attempts": attempts,
        "violations": violations,
        "fallback_reason": "大模型两次输出都出现未授权数字，已退回模板文案",
    })


# ============================================================
# 5. 报告排版
# ============================================================
def _format_key_numbers(numbers):
    """把 numbers 渲染成一行可核对的文本（含嵌套的组均值）。"""
    parts, nested = [], []
    for k, v in numbers.items():
        if isinstance(v, dict):
            nested.append(f"{k}：" + " / ".join(f"{kk} {vv}" for kk, vv in v.items()))
        elif isinstance(v, float) and abs(v) < 1e-3 and v != 0:
            parts.append(f"{k} = {v:.2e}")
        else:
            parts.append(f"{k} = {v}")
    text = "、".join(parts)
    if nested:
        text += "；" + "；".join(nested)
    return text


def _profile_table(profile, roles):
    type_of = {}
    for col in roles["numeric"]:
        type_of[col] = "数值"
    for col in roles["categorical"]:
        type_of[col] = "分类"
    for col in roles["datetime"]:
        type_of[col] = "日期"
    for col in roles["text"]:
        type_of[col] = "文本"
    for col in roles["id"]:
        type_of[col] = "编号"

    lines = ["| 字段 | 类型 | 缺失率 | 唯一值 | 概要 |", "|---|---|---|---|---|"]
    for col, info in profile["columns"].items():
        t = type_of.get(col, "—")
        miss = f"{info['缺失率'] * 100:.0f}%"
        if "均值" in info:
            brief = f"均值 {info['均值']}，范围 {info['最小值']} ~ {info['最大值']}"
        elif "最多类别" in info:
            brief = f"最多为「{info['最多类别']}」（{info['最多类别占比'] * 100:.0f}%）"
        else:
            brief = "—"
        lines.append(f"| {col} | {t} | {miss} | {info['唯一值数']} | {brief} |")
    return "\n".join(lines)


def _cn_num(n):
    """1 → 一，用于章节序号（报告里最多到五，不需要通用实现）。"""
    return "一二三四五六七八九十"[n - 1]


def render_report(source, profile, roles, records, discoveries, issues, backends):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_llm = sum(1 for b in backends if b == "llm")
    if not records:
        mode = "未生成结论"
    elif n_llm:
        mode = f"大模型生成 {n_llm} 条，其余为模板兜底"
    else:
        mode = "全部由模板生成（未联网）"

    L = []
    L.append(f"# 数据分析报告：{source}\n")
    L.append(f"> 自动生成于 {ts} · 共 {len(records)} 条发现 · {mode}\n")
    L.append("> 本报告中所有数值均由 pandas / scipy 直接计算，语言模型只负责表达，"
             "且经过数字校验。\n")
    L.append("---\n")

    L.append("## 一、数据概况\n")
    L.append(f"共 **{profile['rows']}** 行 × **{profile['cols']}** 列。\n")
    L.append(_profile_table(profile, roles) + "\n")

    L.append("## 二、主要发现\n")
    if not discoveries:
        # 没有发现也要出报告。之前这里是直接 return，用户只看到终端里一句话、拿不到文件，
        # 等于白跑一次。踏踏实实说"为什么没有"，比什么都不给要有用。
        L.append(f"本次**没有发现达到统计门槛的关系**"
                 f"（要求 |r| ≥ {A.MIN_R:.2f} 且 p < {A.P_THRESHOLD}，或分组差异显著）。\n")
        L.append("常见原因有三种：\n")
        L.append("1. **字段之间确实缺少稳定关联**——这份数据本身就是描述性的，不需要硬造结论；")
        L.append("2. **样本量偏少**——几十行以下时，真实的规律也常常达不到显著性；")
        L.append(f"3. **数据变异太小**——比如某一列几乎全是同一个值，算不出关系。\n")
        L.append("如果你确信数据里有规律，可以把表格补全后重新分析，"
                 "或直接指定想看的两个字段来单独检验。\n")
    for i, (rec, d) in enumerate(zip(records, discoveries), 1):
        L.append(f"### {i}. 【{KIND_CN.get(d['kind'], d['kind'])}】{d['headline']}\n")
        if rec.get("chart"):
            L.append(f"![图 {i}]({rec['chart']})\n")
        L.append(f"**结论**：{rec['text']}\n")
        L.append(f"**关键数字**：{_format_key_numbers(d['numbers'])}\n")
        L.append(f"**依据**：基于 {', '.join(d['columns'])}；"
                 f"统计量见上，可自行核对。\n")

    # 章节序号必须跟着实际出现的章节走。
    # 之前是写死的"三、数据质量提示""四、局限说明"，一旦这张表没有数据质量问题，
    # 读者看到的就是"二、主要发现 → 四、局限说明"，中间凭空缺了一节，像是生成坏了。
    sec = 2

    if issues:
        sec += 1
        L.append(f"## {_cn_num(sec)}、数据质量提示\n")
        L.append("| 字段 | 问题 | 严重度 |")
        L.append("|---|---|---|")
        for it in issues:
            L.append(f"| {it['列']} | {it['问题']} | {it['严重度']} |")
        L.append("")

    sec += 1
    L.append(f"## {_cn_num(sec)}、局限说明\n")
    L.append("- 本报告只做**描述性分析**，说明数据中呈现的关系，"
             "**不能证明因果关系**。")
    L.append("- 结论的可靠性依赖样本质量；缺失率高或类别极不均衡的字段已在上表中标出。")
    n_rows = profile["rows"]
    if n_rows < A.SMALL_SAMPLE_N:
        # 小样本要主动说破，而不是塞进一句放之四海皆准的"个别极端值可能有影响"
        L.append(f"- **样本量仅 {n_rows} 行，属于小样本。** 相关系数与显著性都容易受"
                 f"个别极端值影响，结论应视为探索性参考，不宜据此下强判断；"
                 f"需要更可靠的结论时建议补充样本（{A.SMALL_SAMPLE_N} 行以上）。\n")
    else:
        L.append(f"- 样本量 {n_rows} 行；行数不多时，个别极端值可能对统计量产生较大影响。\n")
    L.append("---\n")
    L.append("*由「表格分析助手」生成：统计层负责计算，表达层负责措辞，"
             "两件事严格分开。*")

    return "\n".join(L)


# ============================================================
# 6. 主流程
# ============================================================
def pick_narrator(backend):
    key = os.environ.get(API_KEY_ENV, "").strip()

    if backend == "llm":
        if not key:
            print(f"未设置环境变量 {API_KEY_ENV}，无法使用大模型后端。")
            print("  设置方法（PowerShell）：$env:SHEET_INSIGHT_API_KEY=\"sk-...\"")
            sys.exit(2)
        return LLMNarrator(key)

    if backend == "auto":
        if key:
            return LLMNarrator(key)
        print(f"未检测到 {API_KEY_ENV}，本次使用离线模板后端（效果可用，文笔较平）。")
        return TemplateNarrator()

    return TemplateNarrator()


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成报告（M3 表达层）")
    parser.add_argument("path", nargs="?", default=A.DEFAULT_PATH, help="数据文件路径")
    parser.add_argument("--backend", choices=["auto", "llm", "template"], default="auto")
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "report.md"), help="报告输出路径")
    args = parser.parse_args(argv)

    if not os.path.exists(args.path):
        print(f"找不到文件：{args.path}")
        return 1

    # --- 统计层：拿 discoveries ---
    df = A.load_table(args.path)
    roles = A.classify_columns(df)
    profile = A.build_profile(df, roles)
    all_found = (A.find_correlations(df, roles) + A.find_group_differences(df, roles)
                 + A.find_trends(df, roles) + A.find_outliers(df, roles))
    issues = A.find_quality_issues(df, roles)
    discoveries = A.select_top(all_found, top_k=6)

    if not discoveries:
        print("没有找到达到统计门槛的发现——仍会生成一份报告，说清楚为什么。")

    # --- 表达层：逐条生成 ---
    narrator = pick_narrator(args.backend)
    print(f"\n表达层后端：{getattr(narrator, 'label', '模板（离线）')}")
    print("=" * 62)

    records, backends = [], []
    for i, d in enumerate(discoveries, 1):
        rec = narrate_one(d, narrator)
        chart = charts.render(d, df, i)
        rec.update({"index": i, "kind": d["kind"], "chart": chart, "headline": d["headline"]})
        records.append(rec)
        backends.append(rec["backend"])

        flag = "✓" if rec["backend"] != "template" or not rec["fallback_reason"] else "↩"
        print(f"  [{flag}] {i}. {KIND_CN.get(d['kind'], d['kind']):<5} "
              f"{rec['backend']:<8} 尝试 {rec['attempts']} 次"
              f"{'  图:' + rec['chart'] if chart else ''}")
        if rec["fallback_reason"]:
            print(f"      降级原因：{rec['fallback_reason']}")

    # --- 落盘 ---
    md = render_report(os.path.basename(args.path), profile, roles,
                       records, discoveries, issues, backends)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)

    guard_stats = {
        "total": len(records),
        "from_llm": backends.count("llm"),
        "fallback": sum(1 for r in records if r["fallback_reason"]),
        "retries": sum(1 for r in records if r["attempts"] > 1),
        # 被守卫拦下的 LLM 违规次数（发生在生成过程中）
        "llm_rejected": sum(len(v["bad_numbers"])
                            for r in records for v in r["violations"]),
        # 最终报告里残留的未授权数字——这个必须恒为 0，是产品的硬指标
        "final_unauthorized": sum(len(r["final_check"]) for r in records),
    }
    with open(os.path.join(OUT_DIR, "narration.json"), "w", encoding="utf-8") as f:
        json.dump({"source": os.path.basename(args.path),
                   "generated_at": datetime.now().isoformat(timespec="seconds"),
                   "backend": getattr(narrator, "name", "template"),
                   "stats": guard_stats,
                   "records": records}, f, ensure_ascii=False, indent=2)

    print("=" * 62)
    print(f"报告已生成：{args.out}")
    print(f"数字守卫：{guard_stats['total']} 条结论，"
          f"生成过程中拦截未授权数字 {guard_stats['llm_rejected']} 处，"
          f"降级 {guard_stats['fallback']} 条，重试 {guard_stats['retries']} 条")
    print(f"最终报告中残留的未授权数字：{guard_stats['final_unauthorized']} 处"
          f"{'  ✓' if guard_stats['final_unauthorized'] == 0 else '  ✗ 需要排查'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
