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
import re
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


def time_axis_unit(col, s):
    """
    判断一列数字是不是「时间轴」，返回 "year" / "month" / None。

    为什么需要这个：`find_trends` 原先只认真正的日期列。
    可现实里大量数据的时间轴就是普通数字——「年份 2005-2025」「月份 1-12」。
    这类列被判成普通数值列之后，趋势分析压根不会看它，
    于是一份 2005-2025 的生产数据，最重要的"逐年增长"结论直接消失。

    名字必须**明确指向时间**才认（免得把"年龄"当成"年"）：
    年份 / 年度 / year、月份 / month，并且数值范围要合理。
    """
    name = str(col).lower().strip()
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) < 3 or v.nunique() < 3:
        return None
    # 时间轴一定是整数（2005、1、12），带小数的大概率是普通指标
    if not np.all(np.equal(np.mod(v.to_numpy(dtype=float), 1), 0)):
        return None

    lo, hi = float(v.min()), float(v.max())
    is_year_name = name.endswith(("年份", "年度")) or name in ("年", "year", "years")
    is_month_name = name.endswith(("月份",)) or name in ("月", "month", "months")

    if is_year_name and 1900 <= lo and hi <= 2100:
        return "year"
    if is_month_name and 1 <= lo and hi <= 12:
        return "month"
    return None


def build_time_axis(df, col, unit):
    """
    把时间轴列转成数值横坐标 + 可直接标注的原始值。

    三种轴的单位不同，横坐标也就用不同的单位，这样"每周期变化"才对得上：
      day   → 距首日的天数（每 7 天 = 一周）
      year  → 年份本身（2005、2006…）
      month → 月份本身（1、2…12）
    """
    if unit == "day":
        t = pd.to_datetime(df[col], errors="coerce", format="mixed")
        base = t.min()
        return (t - base).dt.days.astype(float), t
    v = pd.to_numeric(df[col], errors="coerce").astype(float)
    return v, v


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
def _detect_header_row(raw):
    """
    找出真正的表头在第几行（0 基）。

    真实 Excel 极少把表头放在第一行：上面往往压着一行合并的大标题和单位说明，
    比如「某工厂2024年1-12月销售额  单位：（亿元）」，A1:D1 还是合并单元格。
    直接 header=0 的后果是：标题被当成第一个列名，后面全变成 Unnamed: 1/2/3，
    整张表彻底无法分析——而且 pandas 不会报错。

    判定规则：在前 10 行里，挑第一行同时满足
      ① 非空单元格数 ≥ 最宽行的 60%（排除那种只有一格标题的行）
      ② 其中至少一半是字符串（表头文字，而不是数据）
    """
    head = raw.head(10)
    widths = head.notna().sum(axis=1)
    if widths.empty or widths.max() == 0:
        return 0
    widest = int(widths.max())

    for i in range(len(head)):
        row = head.iloc[i].dropna()
        if len(row) < max(2, widest * 0.6):
            continue
        str_ratio = sum(1 for v in row if isinstance(v, str)) / len(row)
        if str_ratio >= 0.5:
            return i
    return 0


def _coerce_dtypes(df):
    """
    还原被 header=None 打乱的数值类型。

    这是个必须补的坑：用 `read_excel(header=None)` 才能自己找表头，
    但代价是表头那一行也被当成数据读了进来，于是**整列都变成 object**——
    "年份 2005"、"净利润 1200" 全成了字符串。
    后果比想象中严重：数值列会被判成"每行都不一样的字符串"，
    直接归类到 ID 列，相关性、趋势、异常值三类分析全部归零，而且不报错。

    这里逐列试着转数值，只有"绝大多数单元格都能转"时才替换，
    避免把"备注"这种真文本列里夹带的数字当成数值。
    """
    for c in df.columns:
        s = df[c]
        if s.dtype != object:
            continue
        conv = pd.to_numeric(s, errors="coerce")
        non_null = int(s.notna().sum())
        if non_null and conv.notna().sum() >= non_null * 0.95:
            df[c] = conv
    return df


_YEAR_COL_RE = re.compile(r"^(19|20)\d{2}\s*年?$")
_LABEL_HINTS = ("指标", "名称", "项目", "地区", "类别", "省份", "品类", "产品")


def _maybe_transpose_wide_years(df):
    """
    识别「年份在列」的宽表，转置成「年份在行」的标准时间序列表。

    国家统计局、海关总署导出的数据几乎都是这个形状：

        序号 | 指标                        | 2024年  | 2023年  | … | 2006年
        1    | 中国同美国进出口总额        | 68833151 | 66400000 | … | 26265946
        2    | 中国同加拿大进出口总额      | 9300957  | 8900000  | … | 2317883

    如果照原样分析，"2024年"会被当成一个普通数值列，它的均值是把
    美国进出口额和"北美洲其他国家"的进出口额加在一起算出来的——
    数字算得出来，但没有任何含义，属于最典型的"沉默地算错"。

    转置之后：列 = 各项指标，行 = 年份。年份变成时间轴，
    每个指标都能各算各的趋势，相关系数也只在同一时间截面上比。
    """
    if df.empty or len(df) < 3:
        return df

    year_cols = [c for c in df.columns if _YEAR_COL_RE.match(str(c).strip())]
    if len(year_cols) < 3:
        return df

    # 找标签列：值应该是"文字"，优先挑名字像标签的
    others = [c for c in df.columns if c not in year_cols]
    label_col = None
    for c in others:
        if any(h in str(c) for h in _LABEL_HINTS):
            label_col = c
            break
    if label_col is None:
        for c in others:
            vals = df[c].dropna().astype(str)
            if len(vals) >= 3 and vals.str.len().mean() >= 4:
                label_col = c
                break
    if label_col is None:
        return df

    # 年份列按真实年份升序排列（原表通常是倒序的）
    ordered = sorted(year_cols, key=lambda c: int(re.match(r"^(19|20)\d{2}", str(c).strip()).group(0)))

    sub = df[[label_col] + ordered].dropna(subset=[label_col])
    # 表尾的"数据来源：xxx""注：xxx"不是指标，先剔掉
    keep = ~sub[label_col].astype(str).str.contains("数据来源|注：|说明|单位：", na=False)
    sub = sub[keep]
    if len(sub) < 2:
        return df

    wide = sub.set_index(label_col)[ordered].T
    wide.index = [int(re.match(r"^(19|20)\d{2}", str(c).strip()).group(0)) for c in wide.index]
    wide.index.name = "年份"

    out = wide.reset_index()
    # 指标名里常见的"0中国同美国…""1中国向美国出口…"前缀是分类序号，去掉更好读
    out.columns = [re.sub(r"^\d+", "", str(c)).strip() if str(c) != "年份" else "年份"
                   for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    out = _coerce_dtypes(out)
    return out


def load_table(path):
    """读取 CSV / Excel，并做最基础的清理（列名去空格）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            raw = pd.read_excel(path, header=None)
        except ImportError:
            # 原始报错是 "Missing optional dependency 'openpyxl'"，用户看不懂该干什么。
            # 这里把它翻译成一条可以直接照抄的命令——出错信息的第一职责是给出下一步。
            raise SystemExit(
                f"\n读取 {ext} 需要一个额外的引擎库，装上即可：\n"
                f"    python -m pip install openpyxl\n"
                f"（若是老的 .xls 格式，再补一个：python -m pip install xlrd）\n"
            ) from None

        hdr = _detect_header_row(raw)
        names = []
        for i, v in enumerate(raw.iloc[hdr]):
            text = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
            text = text.replace("\u3000", " ").strip()
            names.append(text or f"列{i + 1}")
        df = raw.iloc[hdr + 1:].reset_index(drop=True)
        df.columns = names
        df = _coerce_dtypes(df)
    else:
        # utf-8-sig 能吃掉 Excel 导出时带的 BOM，不加的话第一列列名会变成 "\ufeff学号"
        df = pd.read_csv(path, encoding="utf-8-sig")

    # 真实表格的列名经常带前后空格/全角空格，不去掉会导致后面按列名取不到
    df.columns = [str(c).replace("\u3000", " ").strip() for c in df.columns]
    # 去掉完全空白的列（合并单元格撑出来的空列就靠这一步清掉）
    df = df.dropna(axis=1, how="all")
    # 年份在列的宽表先转置，否则会把"2024年"当成一个普通指标来算均值
    df = _maybe_transpose_wide_years(df)
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
    roles = {"id": [], "datetime": [], "time": [], "numeric": [],
             "categorical": [], "text": []}

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
            # 「年份 / 月份」这类数字时间轴单独归到 time：
            # 它不该参与"两个指标有没有相关"（和年份相关其实就是趋势），
            # 但必须是趋势分析的时间基准。
            if time_axis_unit(col, s):
                roles["time"].append(col)
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
def _make(kind, score, headline, numbers, columns, suggestion, axis=None, note=None):
    """统一的发现数据结构。numbers 就是后面给 LLM 的"数字白名单"。"""
    return {
        "kind": kind,
        "score": round(float(score), 4),
        "headline": headline,
        "numbers": numbers,
        "columns": columns,
        "suggestion": suggestion,   # 建议画什么图
        "axis": axis,               # 仅趋势类用：day / year / month
        "note": note,               # 特殊标记，如 near_same（两个指标近乎重合）
    }


def _fmt_axis_end(v, unit):
    """把时间轴的端点转成人能读的写法（日期 / 年份 / 月份）。"""
    if unit == "day":
        return str(pd.Timestamp(v).date())
    return str(int(round(float(v))))


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

            # |r| ≈ 1 往往不是"发现"，而是两个指标本来就是同一个东西：
            # 总量与它的分项、同一指标的不同口径（如"北美洲进出口"包含"美国进出口"）。
            # 这种关系照实写出来，但不该占报告头条——它会挤掉真正值得说的问题。
            # 处理方式：降权 + 换一种说法，让读者知道这是定义造成的必然。
            near_same = abs(r) >= 0.995
            if near_same:
                score *= 0.55
                headline = (f"「{a}」与「{b}」几乎完全重合（r={r:.3f}），"
                            f"很可能是同一指标的不同口径")
            else:
                headline = f"「{a}」与「{b}」呈{corr_strength(r)}{direction}相关（r={r:.2f}）"

            out.append(_make(
                kind="correlation",
                score=score,
                headline=headline,
                numbers={"r": round(float(r), 3), "p": float(f"{p:.2e}"), "n": n,
                         "均值_a": round(float(pair[a].mean()), 2),
                         "均值_b": round(float(pair[b].mean()), 2)},
                columns=[a, b],
                suggestion="scatter" if abs(r) > 0.4 else "hexbin",
                note="near_same" if near_same else None,
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

    # 三种时间轴：真日期、年份、月份。横坐标单位跟着轴走，
    # 这样"每周期变化"才是真的"每年 / 每月 / 每周"变化，而不是一律换算成周。
    axes = [(c, "day") for c in roles["datetime"]] + \
           [(c, time_axis_unit(c, df[c])) for c in roles["time"]]

    for axis_col, unit in axes:
        if unit is None:
            continue
        x_raw, t_raw = build_time_axis(df, axis_col, unit)
        period_label, per_factor = {"day": ("周", 7.0),
                                    "year": ("年", 1.0),
                                    "month": ("月", 1.0)}[unit]

        for num in roles["numeric"]:
            sub = pd.DataFrame({"x": x_raw, "v": pd.to_numeric(df[num], errors="coerce")}).dropna()
            n = len(sub)
            if n < MIN_PAIR_N:
                continue

            x = sub["x"].to_numpy().astype(float)
            lo, hi = float(np.min(x)), float(np.max(x))
            span = hi - lo
            if span <= 0:
                continue
            # 跨度门槛：日期轴至少两周；年/月轴至少 3 个周期
            if (unit == "day" and span < 14) or (unit != "day" and span < 3):
                continue

            y = sub["v"].to_numpy().astype(float)
            res = stats.linregress(x, y)
            sd = float(np.std(y, ddof=1))
            if sd == 0:
                continue

            total_change = float(res.slope) * span          # 全期变化量
            # 效应量：总变化相当于几个标准差。太小的趋势不值得写进报告
            effect = abs(total_change) / sd
            if res.pvalue >= P_THRESHOLD or effect < 0.30:
                continue

            per_change = float(res.slope) * per_factor
            span_text = f"{int(round(span))} 天" if unit == "day" else f"{int(round(span))} {period_label}"
            out.append(_make(
                kind="trend",
                score=min(1.0, effect) * 0.6,
                headline=(f"「{num}」随{axis_col}呈{'上升' if res.slope > 0 else '下降'}趋势："
                          f"每{period_label}约{'+' if per_change > 0 else ''}{per_change:.2f}，"
                          f"全期累计{'+' if total_change > 0 else ''}{total_change:.1f}"),
                numbers={"周期": period_label,
                         "每周期变化": round(per_change, 4),
                         "全期变化": round(total_change, 2),
                         "p": float(f"{res.pvalue:.2e}"), "n": n,
                         "跨度": span_text,
                         "起止": f"{_fmt_axis_end(t_raw.min(), unit)} ~ "
                                f"{_fmt_axis_end(t_raw.max(), unit)}"},
                columns=[axis_col, num],
                suggestion="line",
                axis=unit,
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
