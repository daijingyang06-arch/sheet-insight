"""
发现引擎（统计层）—— M1 / M2 的核心
=====================================
这是整个产品的"技术心脏"，也是它区别于"AI 套壳"的地方。

职责：**只做计算，不做表达。**
  输入一份表格 → 输出「数据画像 + 若干条带精确数字的发现」。
  所有数字都由 pandas/scipy 真实算出来，一个字都不靠大模型猜。
  后续（M3）再把这份结构化结果交给 LLM，让它"翻译成人话"。

为什么这样分工？
  大模型的算术不可靠，会一本正经地编出"平均分提高了 23%"，
  而真实数字可能是 12%。所以架构上必须把"算"和"说"彻底隔离：
  LLM 只能引用这里给出的数字，不允许自己计算。

用法：
    python analyze.py                       # 分析默认的模拟数据
    python analyze.py 你的表格.xlsx          # 分析任意表格
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(BASE_DIR, "data", "sample_survey.csv")
OUT_JSON = os.path.join(BASE_DIR, "results", "discoveries.json")

# 参与统计的门槛（阈值太低会产出大量"噪音发现"，太高又会漏掉真实规律）
#
# 【MIN_PAIR_N 为什么从 50 改成 12】
# 50 是在 300 行的问卷数据上调出来的。拿它去跑一份 31 行的地区经济数据
# （31 个省 × GDP × 社会消费品零售总额），结果是：相关性一条都进不来。
# 可那张表里 r = 0.97、p = 5.6e-20，是全表唯一的、也是最重要的结论。
# 引擎什么都没说，用户只会以为"这份数据没规律"——这正是最危险的失败：
# 不报错，只是安静地把真结论丢掉。
#
# 根子上的错在于把两件事混成了一件：p 值门槛本来就随样本量自动调整
# （n 变小时，同样的 r 更难显著），再压一个 n ≥ 50 是重复限制，
# 等于把整类中等规模的表格（几十行，真实场景里极常见）一刀切拒之门外。
#
# 现在 12 只用来挡"3 个点连成直线"这种极端情况，真正的把关交给 p 值和 |r|。
MIN_PAIR_N = 12          # 算相关/检验的硬下限
SMALL_SAMPLE_N = 30      # 低于此行数，报告要主动标注"样本偏少"
MIN_GROUP_N = 8          # 每个分组至少多少人
MIN_R = 0.25             # 相关系数绝对值下限
P_THRESHOLD = 0.01       # 显著性水平
MAX_MISSING_FOR_TEST = 0.40   # 缺失率超过这个值的列不参与组间检验


def corr_strength(r):
    """
    相关系数的强度分级，按统计教材的通用口径：
    |r| ≥ 0.8 高度、≥ 0.5 中度、其余低度。

    这个函数存在的唯一理由是"只留一个真相来源"：
    标题里写"高度正相关"、正文里写"属于较强程度"，两句话必须对得上。
    之前标题是写死的"中等以上"，于是 r=0.97 和 r=0.51 得到一模一样的措辞。
    """
    a = abs(r)
    if a >= 0.8:
        return "高度"
    if a >= 0.5:
        return "中度"
    return "低度"


# ============================================================
# 1. 读取
# ============================================================
def load_table(path):
    """读取 CSV / Excel，并做最基础的清理（列名去空格）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path)
        except ImportError:
            # 原始报错是 "Missing optional dependency 'openpyxl'"，用户看不懂该干什么。
            # 这里把它翻译成一条可以直接照抄的命令——出错信息的第一职责是给出下一步。
            raise SystemExit(
                f"\n读取 {ext} 需要一个额外的引擎库，装上即可：\n"
                f"    python -m pip install openpyxl\n"
                f"（若是老的 .xls 格式，再补一个：python -m pip install xlrd）\n"
            ) from None
    else:
        # utf-8-sig 能吃掉 Excel 导出时带的 BOM，不加的话第一列列名会变成 "\ufeff学号"
        df = pd.read_csv(path, encoding="utf-8-sig")

    # 真实表格的列名经常带前后空格/全角空格，不去掉会导致后面按列名取不到
    df.columns = [str(c).replace("\u3000", " ").strip() for c in df.columns]
    # 去掉完全空白的列
    df = df.dropna(axis=1, how="all")
    return df


# ============================================================
# 2. 列角色识别
# ============================================================
def classify_columns(df):
    """
    判断每一列是「数值 / 分类 / 日期 / 文本 / ID」。

    这是所有分析的前提——如果程序把"学号"当成数值列去算平均分，
    结果会毫无意义（学号的平均值没有语义）。

    注意一个坑：pandas 3.x 读进来的字符串列 dtype 是 `str`，不再是 `object`。
    如果只用 `dtype == object` 判断，会把所有分类列（性别、专业…）全部漏掉，
    导致组间差异和时间趋势两类分析直接空白。所以这里统一用 is_string_dtype。
    """
    roles = {"id": [], "datetime": [], "numeric": [], "categorical": [], "text": []}

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        n_unique = s.nunique(dropna=True)

        is_str = pd.api.types.is_string_dtype(s) or s.dtype == object

        # --- ID 列：名字像 ID，或者每行都不同且是字符串 ---
        name_lower = col.lower()
        looks_id = any(k in name_lower for k in ("学号", "编号", "序号", "id", "工号"))
        if looks_id or (is_str and n_unique == len(df)):
            roles["id"].append(col)
            continue

        # --- 日期列：尝试解析，成功率 > 80% 才算（放在数值判断之前，
        #     因为日期列有时会被读成数字时间戳）---
        if is_str or pd.api.types.is_datetime64_any_dtype(s):
            try:
                parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.8 and n_unique > 5:
                    roles["datetime"].append(col)
                    continue
            except (ValueError, TypeError):
                pass

        # --- 数值列 ---
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if n_unique <= 1:      # 所有值相同 = 没有分析价值
                continue
            roles["numeric"].append(col)
            continue

        # --- 字符串列：区分"文本"和"分类" ---
        if is_str:
            if n_unique <= 1:
                continue
            avg_len = non_null.astype(str).str.len().mean()
            if n_unique > 20 or avg_len > 12:
                roles["text"].append(col)      # 长文本/开放式回答
            else:
                roles["categorical"].append(col)  # 性别、年级、专业…
            continue

    return roles


# ============================================================
# 3. 数据画像
# ============================================================
def build_profile(df, roles):
    """每列的基本情况：缺失率、唯一值、数值列的分布统计。"""
    profile = {"rows": len(df), "cols": len(df.columns), "columns": {}}

    for col in df.columns:
        s = df[col]
        info = {
            "缺失率": round(float(s.isna().mean()), 4),
            "唯一值数": int(s.nunique(dropna=True)),
        }
        if col in roles["numeric"]:
            v = pd.to_numeric(s, errors="coerce").dropna()
            info.update({
                "均值": round(float(v.mean()), 2),
                "中位数": round(float(v.median()), 2),
                "最小值": round(float(v.min()), 2),
                "最大值": round(float(v.max()), 2),
                "标准差": round(float(v.std()), 2),
            })
        elif col in roles["categorical"]:
            vc = s.value_counts()
            if len(vc):
                info["最多类别"] = str(vc.index[0])
                info["最多类别占比"] = round(float(vc.iloc[0] / vc.sum()), 3)
        profile["columns"][col] = info

    return profile


# ============================================================
# 4. 各类发现
# ============================================================
def _make(kind, score, headline, numbers, columns, suggestion):
    """统一的发现数据结构。numbers 就是后面给 LLM 的"数字白名单"。"""
    return {
        "kind": kind,
        "score": round(float(score), 4),
        "headline": headline,
        "numbers": numbers,
        "columns": columns,
        "suggestion": suggestion,   # 建议画什么图
    }


def find_correlations(df, roles):
    """数值列两两相关。"""
    out = []
    cols = roles["numeric"]

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            pair = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            n = len(pair)
            if n < MIN_PAIR_N:
                continue

            r, p = stats.pearsonr(pair[a], pair[b])
            if not np.isfinite(r) or abs(r) < MIN_R or p >= P_THRESHOLD:
                continue

            # 显著性加成：p 越小分越高（上限 1.0）
            sig = min(1.0, -np.log10(max(p, 1e-300)) / 3)
            score = abs(r) * (0.7 + 0.3 * sig)
            direction = "正" if r > 0 else "负"

            out.append(_make(
                kind="correlation",
                score=score,
                headline=f"「{a}」与「{b}」呈{corr_strength(r)}{direction}相关（r={r:.2f}）",
                numbers={"r": round(float(r), 3), "p": float(f"{p:.2e}"), "n": n,
                         "均值_a": round(float(pair[a].mean()), 2),
                         "均值_b": round(float(pair[b].mean()), 2)},
                columns=[a, b],
                suggestion="scatter" if abs(r) > 0.4 else "hexbin",
            ))
    return out


def find_group_differences(df, roles):
    """分类列的各个分组之间，数值列是否有显著差异。"""
    out = []

    for cat in roles["categorical"]:
        if df[cat].isna().mean() > MAX_MISSING_FOR_TEST:
            continue
        vc = df[cat].value_counts()
        groups_all = [g for g in vc.index if vc[g] >= MIN_GROUP_N]
        if not (2 <= len(groups_all) <= 8):
            continue

        for num in roles["numeric"]:
            sub = df[[cat, num]].copy()
            sub[num] = pd.to_numeric(sub[num], errors="coerce")
            sub = sub.dropna()
            sub = sub[sub[cat].isin(groups_all)]
            if len(sub) < MIN_PAIR_N:
                continue

            samples = [sub.loc[sub[cat] == g, num].to_numpy() for g in groups_all]
            if any(len(s) < MIN_GROUP_N for s in samples):
                continue

            # 单因素方差分析：检验"组间差异是否显著大于组内波动"
            f_stat, p = stats.f_oneway(*samples)
            if not np.isfinite(f_stat) or p >= P_THRESHOLD:
                continue

            # 效应量 η²（eta squared）：差异到底有多大，0~1，比 p 值更有实际意义
            #   p 值只回答"有没有差异"，η² 回答"差异有多大"
            grand_mean = sub[num].mean()
            ss_between = sum(len(s) * (s.mean() - grand_mean) ** 2 for s in samples)
            ss_total = ((sub[num] - grand_mean) ** 2).sum()
            eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0
            if eta2 < 0.02:      # 差异太小，说了也没意义
                continue

            means = {g: round(float(s.mean()), 2) for g, s in zip(groups_all, samples)}
            hi = max(means, key=means.get)
            lo = min(means, key=means.get)

            out.append(_make(
                kind="group_diff",
                # 按 Cohen 的效应量标准归一化：η² 0.01 小、0.06 中、0.14 大
                # 这样分数才有"实际意义大小"的含义，而不是随便乘一个系数
                score=min(1.0, eta2 / 0.14),
                headline=(f"不同「{cat}」的「{num}」存在显著差异："
                          f"{hi}最高（{means[hi]}），{lo}最低（{means[lo]}）"),
                numbers={"eta2": round(eta2, 4), "p": float(f"{p:.2e}"),
                         "n": len(sub), "组均值": means,
                         "最高组": hi, "最低组": lo},
                columns=[cat, num],
                suggestion="box",
            ))
    return out


def find_trends(df, roles):
    """
    时间趋势：某个数值指标随时间整体在上升还是下降。

    做法说明（这里踩过一个坑）：
      最初我按"周"聚合再看趋势，但这份表格只跨 30 天，聚完只剩 4~5 个点，
      点数太少根本判断不了趋势，结果一条都没找出来。
      改成**对原始记录做线性回归**——300 个点全都用上，敏感度高得多。
      只有当"全期总变化量"超过数据本身波动（标准差）的 30% 时才算真趋势，
      否则只是噪声。
    """
    out = []

    for date_col in roles["datetime"]:
        d = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
        for num in roles["numeric"]:
            sub = pd.DataFrame({"t": d, "v": pd.to_numeric(df[num], errors="coerce")}).dropna()
            n = len(sub)
            if n < MIN_PAIR_N:
                continue

            span_days = (sub["t"].max() - sub["t"].min()).days
            if span_days < 14:
                continue      # 时间跨度太短，谈不上"趋势"

            x = (sub["t"] - sub["t"].min()).dt.days.to_numpy().astype(float)
            y = sub["v"].to_numpy().astype(float)
            res = stats.linregress(x, y)
            sd = float(np.std(y, ddof=1))
            if sd == 0:
                continue

            total_change = float(res.slope) * float(x.max())   # 全期变化量
            # 效应量：总变化相当于几个标准差。太小的趋势不值得写进报告
            effect = abs(total_change) / sd
            if res.pvalue >= P_THRESHOLD or effect < 0.30:
                continue

            weekly = float(res.slope) * 7
            out.append(_make(
                kind="trend",
                score=min(1.0, effect) * 0.6,
                headline=(f"「{num}」随时间呈{'上升' if res.slope > 0 else '下降'}趋势："
                          f"每周约{'+' if weekly > 0 else ''}{weekly:.2f}，"
                          f"全期累计{'+' if total_change > 0 else ''}{total_change:.1f}"),
                numbers={"每周变化": round(weekly, 3), "全期变化": round(total_change, 2),
                         "p": float(f"{res.pvalue:.2e}"), "n": n,
                         "跨度天数": int(span_days),
                         "起止": f"{sub['t'].min().date()} ~ {sub['t'].max().date()}"},
                columns=[date_col, num],
                suggestion="line",
            ))
    return out


def find_outliers(df, roles):
    """
    异常值：用**稳健 z 分数**（基于中位数和 MAD）找离群点。

    为什么不用常见的 1.5 倍 IQR 规则？
      1.5 IQR 在偏态分布上会把大量正常值判成异常（我第一版就误报了睡眠 4.1 小时）。
      稳健 z 分数衡量的是"离主体有多远"，配合阈值可以把
      "真正的录入错误"（下周自习 60 小时）和"正常的极端值"区分开。
    """
    out = []

    for num in roles["numeric"]:
        v = pd.to_numeric(df[num], errors="coerce").dropna()
        if len(v) < 20:
            continue

        med = float(v.median())
        mad = float((v - med).abs().median())
        if mad == 0:
            continue
        # 1.4826 是让 MAD 在正态分布下与标准差可比的换算常数
        robust_sigma = 1.4826 * mad
        z = (v - med).abs() / robust_sigma

        flagged = v[z >= 3.5]
        if len(flagged) == 0 or len(flagged) / len(v) > 0.05:
            continue      # 超过 5% 说明整体分布就长尾，不算"个别异常"

        max_z = float(z.max())
        extreme = float(v.loc[z.idxmax()])
        out.append(_make(
            kind="outlier",
            # 异常值本质是"数据质量提示"，不是业务洞察，所以分数上限压低（0.45），
            # 否则一条录入错误会挤掉真正有价值的规律
            score=min(0.45, (max_z - 3.0) / 8.0),
            headline=(f"「{num}」有 {len(flagged)} 个明显离群的值，"
                      f"最极端的是 {extreme}（中位数 {med:.2f}，偏离 {max_z:.1f} 倍稳健标准差）"),
            numbers={"离群个数": int(len(flagged)), "最极端值": extreme,
                     "中位数": round(med, 2), "偏离倍数": round(max_z, 2),
                     "样本数": int(len(v))},
            columns=[num],
            suggestion="box",
        ))
    return out


def find_quality_issues(df, roles):
    """数据质量问题。这类不算"发现"，但会直接影响结论可信度，必须提示。"""
    issues = []

    for col in df.columns:
        s = df[col]
        miss = s.isna().mean()
        n_unique = s.nunique(dropna=True)

        if n_unique == 0:
            issues.append({"列": col, "问题": "整列都是空值", "严重度": "高"})
        elif n_unique == 1 and len(s) > 0:
            issues.append({"列": col, "问题": f"所有值都是「{s.dropna().iloc[0]}」，无分析价值", "严重度": "低"})
        elif miss > 0.20:
            issues.append({"列": col, "问题": f"缺失率 {miss*100:.1f}%，相关分析结论需谨慎", "严重度": "中"})

        if col in roles["categorical"] and n_unique >= 2:
            vc = s.value_counts()
            share = vc.iloc[-1] / vc.sum()
            if share < 0.05:
                issues.append({"列": col, "问题": f"「{vc.index[-1]}」只有 {int(vc.iloc[-1])} 条（占 {share*100:.1f}%），样本过少，组间对比不可靠", "严重度": "中"})

    # 重复行
    dup = int(df.duplicated().sum())
    if dup:
        issues.append({"列": "（整行）", "问题": f"存在 {dup} 行完全重复的记录", "严重度": "中"})

    return issues


# ============================================================
# 5. 排序与挑选
# ============================================================
def select_top(discoveries, top_k=6, min_score=0.15):
    """
    从一堆候选发现里挑出最值得写进报告的几条。

    这里的规则是反复试错后定下来的，每条都对应一个具体的坏结果：

    1. **第一轮先保证"类型多样"**，每种类型（组间差异/相关/趋势/异常值）各取最好的一条。
       纯按分数排序时，前面几条会被相关性占满，报告读起来像复读机。

    2. **同一个分类列只出一条组间差异。**
       否则会出现"不同专业的自习时长有差异""不同专业的成绩有差异"
       "不同专业的满意度有差异"——同一件事换个指标说三遍。

    3. **不做"同列打折"。**（这一条我改过两版）
       中间版本给"已经出现过的列"打 7.5 折，结果适得其反：
       「每周自习时长」同时出现在组间差异和趋势里，被打了两次折，
       导致它和成绩之间的相关性（r=0.56，全表最重要的一条）被挤出榜单。
       **压制重复列的副作用，会误伤"核心变量"。** 所以改成不做惩罚，
       只靠规则 2 和 4 控制重复，其余老实按分数排。

    4. **同类最多 3 条**，既允许"多个因素共同影响同一个结果变量"这种真实情况，
       又不至于让报告变成单一类型。
    """
    ordered = sorted(discoveries, key=lambda d: d["score"], reverse=True)

    picked, kind_used, cat_used = [], {}, set()

    def allowed(d):
        if kind_used.get(d["kind"], 0) >= 3:
            return False
        if d["kind"] == "group_diff" and d["columns"][0] in cat_used:
            return False
        return True

    def take(d):
        picked.append(d)
        kind_used[d["kind"]] = kind_used.get(d["kind"], 0) + 1
        if d["kind"] == "group_diff":
            cat_used.add(d["columns"][0])

    # ---------- 第一轮：每种类型先各取一条，保证报告有"层次" ----------
    for kind in ("group_diff", "correlation", "trend", "outlier"):
        if len(picked) >= top_k:
            break
        for d in ordered:
            if d["kind"] == kind and d["score"] >= min_score and d not in picked:
                take(d)
                break

    # ---------- 第二轮：按分数补足 ----------
    for d in ordered:
        if len(picked) >= top_k:
            break
        if d in picked or d["score"] < min_score or not allowed(d):
            continue
        take(d)

    # 最终展示顺序仍按分数从高到低，而不是按挑选顺序
    return sorted(picked, key=lambda d: d["score"], reverse=True)


# ============================================================
# 6. 输出
# ============================================================
KIND_CN = {
    "correlation": "关联关系",
    "group_diff": "组间差异",
    "trend": "时间趋势",
    "outlier": "异常值",
}


def print_report(profile, discoveries, issues):
    w = 62
    print("=" * w)
    print(f"数据画像：{profile['rows']} 行 × {profile['cols']} 列")
    print("=" * w)
    for col, info in profile["columns"].items():
        miss = info["缺失率"] * 100
        line = f"  {col:<22} 唯一值 {info['唯一值数']:>4}  缺失 {miss:>5.1f}%"
        if "均值" in info:
            line += f"   均值 {info['均值']:>7.2f}  范围 [{info['最小值']}, {info['最大值']}]"
        elif "最多类别" in info:
            line += f"   最多：{info['最多类别']}（{info['最多类别占比']*100:.0f}%）"
        print(line)

    print()
    print("=" * w)
    print(f"关键发现（共 {len(discoveries)} 条）")
    print("=" * w)
    for i, d in enumerate(discoveries, 1):
        print(f"\n[{i}] 【{KIND_CN.get(d['kind'], d['kind'])}】{d['headline']}")
        nums = "、".join(f"{k}={v}" for k, v in d["numbers"].items() if not isinstance(v, dict))
        if nums:
            print(f"    精确数字：{nums}")
        if isinstance(d["numbers"].get("组均值"), dict):
            gm = "、".join(f"{k}: {v}" for k, v in d["numbers"]["组均值"].items())
            print(f"    各组均值：{gm}")
        print(f"    建议图表：{d['suggestion']}")

    if issues:
        print()
        print("=" * w)
        print(f"数据质量问题（{len(issues)} 项）")
        print("=" * w)
        for it in issues:
            print(f"  [{it['严重度']}] {it['列']}：{it['问题']}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.exists(path):
        print(f"找不到文件：{path}")
        return

    df = load_table(path)

    # 样本太小时，任何统计结论都不可靠。与其给用户一堆没意义的数字，
    # 不如直接说清楚为什么没有结果——这是"诚实"比"显得有用"更重要的地方。
    if len(df) < MIN_PAIR_N:
        print(f"这份表格只有 {len(df)} 行，样本量太少（至少需要 {MIN_PAIR_N} 行）")
        print("统计结论在样本量这么小时基本不可靠，建议先补充数据再分析。")
        return

    roles = classify_columns(df)
    profile = build_profile(df, roles)

    # 汇总四种发现，然后统一排序挑选
    all_found = (
        find_correlations(df, roles)
        + find_group_differences(df, roles)
        + find_trends(df, roles)
        + find_outliers(df, roles)
    )
    issues = find_quality_issues(df, roles)
    picked = select_top(all_found, top_k=6)

    print_report(profile, picked, issues)

    # 落盘：这份 JSON 就是 M3 交给 LLM 的输入（numbers 字段是数字白名单）
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "source": os.path.basename(path),
            "profile": profile,
            "discoveries": picked,
            "quality_issues": issues,
            "candidates_total": len(all_found),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结构化结果已保存：{OUT_JSON}")
    print(f"（本次共产生 {len(all_found)} 条候选发现，挑选了 {len(picked)} 条展示）")


if __name__ == "__main__":
    main()
