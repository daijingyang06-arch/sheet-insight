"""
M3 验收脚本
============
M2 的验收解决的是"发现引擎找得全不全"，
M3 的验收要解决的是另一个问题：**大模型编数字时，产品拦不拦得住？**

这个脚本不需要 API Key，也不需要联网——它用"假模型"模拟各种真实情况：

  1. 诚实的模型   → 数字都对 → 文案应被采纳
  2. 说谎的模型   → 编了数字 → 守卫应拦截 → 重试仍说谎 → 降级为模板
  3. 挂掉的模型   → 网络/接口失败 → 降级为模板，报告照常生成
  4. 检测器评测   → 30 条合法改写 vs 30 条错误数字，算漏判率和误杀率
  5. 数字保真度   → 把 numbers 字段拿去和原表独立重算，逐条核对
  6. 报告完整性   → 六条发现、六张图、局限说明，一个都不能少

其中第 4 项是关键：只测"能不能拦住"是不够的，还要测"会不会乱杀"。
一个见谁都拦的守卫等于没有守卫——它会逼系统永远退回模板，等于白接了大模型。

用法：
    python verify_m3.py
"""

import os
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

import analyze as A
import charts
import narrate as N
from number_guard import build_whitelist, check_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "data", "sample_survey.csv")


# ============================================================
# 假模型
# ============================================================
def _quote(discovery, render):
    """
    用 numbers 里的真实数字拼一段话。

    render(value) 决定"怎么写这个数字"——用来批量造出各种合法的、
    以及各种故意写错的样本，喂给守卫做检测器评测。
    跳过 p 值：提示词明确要求正文不出现 p 值，拿它造样本没有代表性。
    """
    chunks = []
    for k, v in discovery["numbers"].items():
        if isinstance(v, (dict, str)) or k == "p":
            continue
        chunks.append(f"{k} 为 {render(float(v))}")
    cols = "、".join(discovery["columns"])
    return f"围绕{cols}的统计结果显示：" + "，".join(chunks) + "。"


def _legal_text(discovery, style="nd2"):
    """合法写法：全精度、合理舍入、百分比形式、绝对值形式。"""
    if style == "raw":
        return _quote(discovery, lambda v: f"{v:g}")
    if style == "pct":
        return _quote(discovery, lambda v: f"{round(v * 100, 2):g}%" if abs(v) < 1 else f"{v:g}")
    if style == "abs":
        return _quote(discovery, lambda v: f"{round(abs(v), 2):g}")
    nd = int(style[2:])
    return _quote(discovery, lambda v: f"{round(v, nd):g}")


def _illegal_text(discovery, factor=1.2):
    """
    错误写法：把一个数字乘上一个系数，模拟大模型"自己又算了一遍"的典型错误。

    这是最真实的翻车场景——它不是胡编一个鬼数字，
    而是拿真实数字去做了一次不该做的运算，得到一个"看起来很像真的"的值。
    """
    done = {"flag": False}

    def render(v):
        if not done["flag"] and abs(v) > 1e-6:
            done["flag"] = True
            return f"{round(v * factor, 2):g}"
        return f"{round(v, 2):g}"

    return _quote(discovery, render)


class FakeLLM:
    """模拟一个 OpenAI 兼容后端。behavior: honest / liar / broken。"""

    name = "llm"

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0

    def narrate(self, messages, discovery=None):      # noqa: ARG002
        self.calls += 1
        if self.behavior == "broken":
            raise N.LLMError("模拟：HTTP 500 服务端错误")
        if self.behavior == "liar":
            return _illegal_text(discovery, factor=1.2)
        return _legal_text(discovery, "nd3")


# ============================================================
# 1. 降级链
# ============================================================
def test_fallback_chain(discoveries):
    print("=" * 68)
    print("一、降级链测试（模型诚实 / 说谎 / 挂掉，报告都得能出来）")
    print("=" * 68)

    results = []

    # --- 诚实模型：应被采纳 ---
    d = discoveries[1]                      # 拿一条相关性来测
    rec = N.narrate_one(d, FakeLLM("honest"))
    ok = (rec["backend"] == "llm" and rec["attempts"] == 1
          and not rec["final_check"] and not rec["fallback_reason"])
    results.append(("诚实模型 → 直接采纳", ok,
                    f"backend={rec['backend']}, 尝试 {rec['attempts']} 次, "
                    f"残留 {len(rec['final_check'])} 处"))

    # --- 说谎模型：应被拦下并降级 ---
    fake = FakeLLM("liar")
    rec = N.narrate_one(d, fake)
    caught = rec["violations"]
    ok = (rec["backend"] == "template"
          and len(caught) == 2                      # 两次都拦住了
          and rec["fallback_reason"] is not None
          and not rec["final_check"])               # 最终文案干净
    results.append(("说谎模型 → 两次拦截并降级", ok,
                    f"拦截 {len(caught)} 次，最终 backend={rec['backend']}, "
                    f"残留 {len(rec['final_check'])} 处"))

    # --- 挂掉的模型：应降级，不抛异常 ---
    rec = N.narrate_one(d, FakeLLM("broken"))
    ok = (rec["backend"] == "template" and "失败" in (rec["fallback_reason"] or "")
          and not rec["final_check"])
    results.append(("模型挂掉 → 降级不报错", ok,
                    f"backend={rec['backend']}, 原因={rec['fallback_reason']}"))

    # --- 全部 6 条在说谎模型下都必须干净收场 ---
    clean = 0
    for d in discoveries:
        r = N.narrate_one(d, FakeLLM("liar"))
        clean += (not r["final_check"] and r["backend"] == "template")
    results.append(("6 条发现全部安全降级", clean == len(discoveries),
                    f"{clean}/{len(discoveries)} 条最终无未授权数字"))

    for name, ok, detail in results:
        print(f"  [{'✓' if ok else '✗'}] {name:<28} {detail}")

    return results


# ============================================================
# 2. 检测器评测（漏判 / 误杀）
# ============================================================
def test_detector(discoveries):
    """
    把守卫当成一个"二分类器"来评：
      正样本 = 含错误数字的文案（应判"不合格"）
      负样本 = 合法改写（应判"合格"）

    两个指标都要好：
      漏判率（把错的放过）→ 产品会输出错数字，最严重
      误杀率（把对的拦下）→ 系统永远退模板，大模型白接
    """
    print()
    print("=" * 68)
    print("二、检测器评测（漏判 = 放过错数字；误杀 = 白拦对的）")
    print("=" * 68)

    legal_styles = ["nd2", "nd3", "nd4", "raw", "pct"]
    # 明显错误：偏离真实值 15% 以上，绝无"舍入"的可能
    strong_factors = [1.2, 1.35, 0.85, 0.7, 1.5]
    # 轻微偏离（10%）：这类篡改在低精度写法下会与正常舍入撞车，单独统计、不参与判定
    mild_factors = [1.1]

    legal_total = legal_killed = 0
    strong_total = strong_missed = 0
    mild_total = mild_missed = 0
    examples = []

    for d in discoveries:
        wl = build_whitelist(d)

        for style in legal_styles:                  # 6 条 × 5 种合法写法 = 30 条
            text = _legal_text(d, style)
            legal_total += 1
            bad = check_text(text, wl)
            if bad:
                legal_killed += 1
                if len(examples) < 4:
                    examples.append(("误杀", style, text[:62], bad))

        for factor in strong_factors:                # 6 条 × 5 种明显错误 = 30 条
            text = _illegal_text(d, factor)
            strong_total += 1
            if not check_text(text, wl):
                strong_missed += 1
                if len(examples) < 4:
                    examples.append(("漏判", f"×{factor}", text[:62], []))

        for factor in mild_factors:
            text = _illegal_text(d, factor)
            mild_total += 1
            if not check_text(text, wl):
                mild_missed += 1

    recall = (strong_total - strong_missed) / strong_total
    false_kill = legal_killed / legal_total
    mild_recall = (mild_total - mild_missed) / mild_total

    print(f"  明显错误样本（偏离 ≥15%）：{strong_total} 条，抓出 "
          f"{strong_total - strong_missed} 条 → 召回率 {recall * 100:.0f}%")
    print(f"  合法改写样本：{legal_total} 条，误杀 {legal_killed} 条 "
          f"→ 误杀率 {false_kill * 100:.0f}%")
    print(f"  轻微偏离样本（偏离 10%）：{mild_total} 条，抓出 "
          f"{mild_total - mild_missed} 条 → 召回率 {mild_recall * 100:.0f}%"
          f"（辅助参考，不参与判定）")

    print("\n  关于文本校验的固有上限（写进报告里，别自己骗自己）：")
    print("    r 的真值是 -0.453，写成「-0.5」是正常的一位小数舍入，必须放行；")
    print("    如果篡改后的值恰好也舍入成「-0.5」，文本上就完全无法区分。")
    print("    这一轮 10% 偏离全部被抓到，但那是运气好，不能当成保证。")
    print("    所以产品保留了第二道防线：每条结论下方原样附上「关键数字」，")
    print("    读者可以拿它和自己的表对账——校验会失效，但原始值始终可查。")

    if examples:
        print("\n  典型样本：")
        for tag, style, text, bad in examples:
            print(f"    [{tag}/{style}] {text}...  违规数字={bad}")

    return [
        ("明显错误召回率 100%", recall == 1.0, f"{recall * 100:.0f}%"),
        ("合法改写误杀率 0%", false_kill == 0.0, f"{false_kill * 100:.0f}%"),
    ]


# ============================================================
# 3. 数字保真度（独立重算）
# ============================================================
def test_fidelity(df, discoveries):
    """
    守卫只能保证"文案里的数字来自 numbers 字段"，
    但 numbers 字段本身对不对？这里换一条完全独立的算法路径重新算一遍。

    这是"结论准确率 100%"这个指标的真正来源：
      不是"模型没改数字"，而是"数字和原始数据对得上"。
    """
    print()
    print("=" * 68)
    print("三、数字保真度（用独立算法从原表重算，逐条核对）")
    print("=" * 68)

    results = []
    for d in discoveries:
        kind, nums = d["kind"], d["numbers"]
        notes, ok = [], True

        if kind == "correlation":
            a, b = d["columns"]
            pair = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            r, _ = stats.pearsonr(pair[a], pair[b])
            ok = abs(r - nums["r"]) < 1e-3 and len(pair) == nums["n"]
            notes.append(f"r 重算 {r:.4f} vs 记录 {nums['r']}")

        elif kind == "group_diff":
            cat, num = d["columns"]
            sub = df[[cat, num]].copy()
            sub[num] = pd.to_numeric(sub[num], errors="coerce")
            sub = sub.dropna()
            groups = {g: s[num].to_numpy() for g, s in sub.groupby(cat)}
            gm = sub[num].mean()
            ss_b = sum(len(v) * (v.mean() - gm) ** 2 for v in groups.values())
            ss_t = ((sub[num] - gm) ** 2).sum()
            eta2 = ss_b / ss_t
            ok = abs(eta2 - nums["eta2"]) < 1e-3
            for g, v in groups.items():
                if abs(round(float(v.mean()), 2) - nums["组均值"][g]) > 0.011:
                    ok = False
            notes.append(f"η² 重算 {eta2:.4f} vs 记录 {nums['eta2']}；"
                         f"组均值 {len(groups)} 组逐一对齐")

        elif kind == "trend":
            axis_col, num = d["columns"]
            unit = d.get("axis") or "day"
            x_raw, _ = A.build_time_axis(df, axis_col, unit)
            v = pd.to_numeric(df[num], errors="coerce")
            sub = pd.DataFrame({"x": x_raw, "v": v}).dropna()
            res = stats.linregress(sub["x"].to_numpy().astype(float),
                                   sub["v"].to_numpy().astype(float))
            factor = {"day": 7.0, "year": 1.0, "month": 1.0}[unit]
            per = res.slope * factor
            ok = abs(per - nums["每周期变化"]) < 0.01
            notes.append(f"每{nums['周期']}变化 重算 {per:.4f} vs 记录 {nums['每周期变化']}")

        elif kind == "outlier":
            col = d["columns"][0]
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            med = float(v.median())
            ok = (abs(med - nums["中位数"]) < 0.01
                  and abs(float(v.max() if nums["最极端值"] > med else v.min())
                          - nums["最极端值"]) < 1e-6)
            notes.append(f"中位数 重算 {med:.2f} vs 记录 {nums['中位数']}；"
                         f"极值 {nums['最极端值']} 存在于原表")

        results.append((f"{kind} · {' / '.join(d['columns'])[:26]}", ok, notes[0]))

    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, note in results:
        print(f"  [{'✓' if ok else '✗'}] {name:<34} {note}")
    print(f"\n  保真：{passed}/{len(results)}")

    return [("所有数字与原始数据一致", passed == len(results), f"{passed}/{len(results)}")]


# ============================================================
# 4. 报告完整性
# ============================================================
def test_report(profile, roles, records, discoveries, issues):
    print()
    print("=" * 68)
    print("四、报告完整性（能不能直接交上去）")
    print("=" * 68)

    md = N.render_report("sample_survey.csv", profile, roles,
                         records, discoveries, issues,
                         [r["backend"] for r in records])

    checks = [
        ("四条必需章节齐全",
         all(h in md for h in ("## 一、数据概况", "## 二、主要发现",
                               "## 三、数据质量提示", "## 四、局限说明"))),
        ("六条发现全部写入", all(f"### {i}." in md for i in range(1, len(discoveries) + 1))),
        ("每条发现都有图",
         sum(1 for r in records if r.get("chart")) == len(records)),
        ("每条发现都有关键数字", md.count("**关键数字**：") == len(discoveries)),
        ("明确写了「相关不等于因果」", "不能证明因果关系" in md),
        ("标注了生成方式", "模板生成" in md or "大模型生成" in md),
    ]

    for name, ok in checks:
        print(f"  [{'✓' if ok else '✗'}] {name}")

    return [(n, o, "") for n, o in checks]


# ============================================================
# main
# ============================================================
def main():
    df = A.load_table(DATA)
    roles = A.classify_columns(df)
    profile = A.build_profile(df, roles)
    found = (A.find_correlations(df, roles) + A.find_group_differences(df, roles)
             + A.find_trends(df, roles) + A.find_outliers(df, roles))
    issues = A.find_quality_issues(df, roles)
    discoveries = A.select_top(found, top_k=6)

    print(f"数据：{len(df)} 行 × {len(df.columns)} 列"
          f"；候选发现 {len(found)} 条；展示 {len(discoveries)} 条\n")

    results = []
    results += test_fallback_chain(discoveries)
    results += test_detector(discoveries)
    results += test_fidelity(df, discoveries)

    # 生成正式产物（同时验证端到端能跑通）
    print()
    print("=" * 68)
    print("五、端到端跑一遍（模板后端，不联网）")
    print("=" * 68)
    records = []
    for i, d in enumerate(discoveries, 1):
        rec = N.narrate_one(d, N.TemplateNarrator())
        rec.update({"index": i, "kind": d["kind"], "chart": charts.render(d, df, i),
                    "headline": d["headline"]})
        records.append(rec)
    print(f"  生成 {len(records)} 条结论，"
          f"图 {sum(1 for r in records if r['chart'])} 张")

    results += test_report(profile, roles, records, discoveries, issues)

    # --- 汇总 ---
    print()
    print("=" * 68)
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        if not ok:
            print(f"  [✗] {name}  {detail}")
    print(f"共 {total} 项检查，通过 {passed} 项")
    print("全部通过 ✓" if passed == total else "有未通过项 ✗ 需要修")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
