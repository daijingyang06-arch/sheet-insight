"""
图表生成 —— 按发现类型自动选图
================================
PRD 里原本设想的是"让 LLM 写 matplotlib 代码、在沙箱里执行"。
实际做下来我改成了**模板选图**，理由很实在：

  1. 只有 4 种发现类型（相关/组间差异/趋势/异常值），对应 4 种图，
     映射关系是确定的。让 LLM 每次重新发明一遍，只是多了一个出错环节。
  2. 用户要的是"一张能放进报告的图"，不是"图有多花哨"。
     固定模板能保证：字体正常、坐标轴有单位、标题带统计量。
  3. 沙箱执行 LLM 代码是有价值的技术点，但它解决的是"未知图表需求"。
     等做到第二版的"用户自定义绘图"时再引入，那时它才真的必要。

结论：MVP 阶段用固定模板，稳定且快；沙箱留到需要时再加。
"""

import os

import matplotlib

matplotlib.use("Agg")        # 不弹窗，直接存文件（服务器上没有显示器）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

import analyze as A

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "results", "charts")

# 中文字体候选：Windows / macOS / Linux 各来几个，谁先命中用谁
CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "DengXian",
    "PingFang SC", "Heiti SC", "STHeiti",
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
]

PRIMARY = "#2563eb"
ACCENT = "#ea580c"
GRID = "#e5e7eb"
TEXT = "#1f2937"


def _setup_font():
    """检测系统里有没有中文字体，有就全局启用，没有就退回英文（绝不输出方块）。"""
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:            # noqa: BLE001
        return None
    for name in CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False   # 负号不显示成方块
            return name
    return None


FONT = _setup_font()


def L(cn, en):
    """有中文字体就用中文标题，没有就用英文——保证任何机器上都是可读的图。"""
    return cn if FONT else en


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ca3af")
    ax.tick_params(colors="#4b5563", labelsize=9)


# ============================================================
# 四种图
# ============================================================
def _scatter(ax, df, d):
    a, b = d["columns"]
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    pair = pd.DataFrame({"x": x, "y": y}).dropna()

    ax.scatter(pair["x"], pair["y"], s=18, alpha=0.5, color=PRIMARY, edgecolors="none")

    # 加一条趋势线：散点图不画拟合线，读者很难看出方向
    if len(pair) >= 3 and pair["x"].nunique() > 1:
        k, b0 = np.polyfit(pair["x"], pair["y"], 1)
        xs = np.linspace(pair["x"].min(), pair["x"].max(), 50)
        ax.plot(xs, k * xs + b0, color=ACCENT, linewidth=2,
                label=L("线性趋势", "trend"))
        ax.legend(frameon=False, fontsize=9, loc="best")

    r = d["numbers"].get("r")
    ax.set_xlabel(a, color=TEXT, fontsize=10)
    ax.set_ylabel(b, color=TEXT, fontsize=10)
    ax.set_title(L(f"「{a}」与「{b}」的关系（r = {r}）",
                   f"{a} vs {b}  (r = {r})"),
                 fontsize=11, color=TEXT, pad=10)


def _box_by_group(ax, df, d):
    cat, num = d["columns"]
    sub = df[[cat, num]].copy()
    sub[num] = pd.to_numeric(sub[num], errors="coerce")
    sub = sub.dropna()

    means = d["numbers"].get("组均值", {})
    order = list(means.keys()) if means else list(sub[cat].value_counts().index)
    groups = [sub.loc[sub[cat] == g, num].to_numpy() for g in order]

    bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                    medianprops={"color": ACCENT, "linewidth": 2},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": "#9ca3af", "markeredgecolor": "none"})
    # 标签用 set_xticks 设置，而不是 boxplot(labels=...)：
    # matplotlib 3.9 起把 labels 改名成 tick_labels，跨版本写法会直接报错。
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    for patch in bp["boxes"]:
        patch.set_facecolor("#dbeafe")
        patch.set_edgecolor(PRIMARY)

    # 把均值点标出来，读者能直接对上正文里的"最高/最低"
    for i, g in enumerate(groups):
        ax.plot(i + 1, g.mean(), marker="D", color=PRIMARY, markersize=6, zorder=5)

    eta2 = d["numbers"].get("eta2")
    ax.set_ylabel(num, color=TEXT, fontsize=10)
    ax.set_xlabel(cat, color=TEXT, fontsize=10)
    ax.set_title(L(f"不同「{cat}」的「{num}」分布（η² = {eta2}）",
                   f"{num} by {cat}  (eta2 = {eta2})"),
                 fontsize=11, color=TEXT, pad=10)
    ax.text(0.01, 0.98, L("◇ 为组均值", "◇ group mean"), transform=ax.transAxes,
            fontsize=8.5, color="#6b7280", va="top")


def _num(v, nd=2):
    """
    图表里的数字格式：不要科学计数法。

    2502139.72 被写成 2.50214e+06 时，标题读起来像一串乱码；
    这些是给人看的图，宁可写成 2502140。
    """
    x = round(float(v), nd)
    if x == 0:
        return "0"
    if abs(x) >= 10000:
        return f"{x:.0f}"
    text = f"{x:.{nd}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _line_trend(ax, df, d):
    axis_col, num = d["columns"]
    unit = d.get("axis") or "day"
    x_raw, t_raw = A.build_time_axis(df, axis_col, unit)
    v = pd.to_numeric(df[num], errors="coerce")
    sub = pd.DataFrame({"x": x_raw, "t": t_raw, "v": v}).dropna().sort_values("x")

    period = d["numbers"].get("周期", "周")
    per = d["numbers"].get("每周期变化")

    if unit == "day":
        ax.scatter(sub["t"], sub["v"], s=14, alpha=0.35, color="#9ca3af",
                   edgecolors="none", label=L("每条记录", "records"))
        # 按天取均值再连线，否则原始点太密、看不出趋势
        daily = sub.set_index("t")["v"].resample("D").mean().dropna()
        if len(daily) >= 2:
            ax.plot(daily.index, daily.values, color=PRIMARY, linewidth=2.2,
                    marker="o", markersize=3.5, label=L("按天均值", "daily mean"))
        ax.tick_params(axis="x", rotation=30)
    else:
        # 年份 / 月份轴：直接按轴值画散点 + 拟合线，刻度就是年份或月份
        ax.scatter(sub["x"], sub["v"], s=26, alpha=0.75, color=PRIMARY,
                   edgecolors="none", label=L("实际值", "actual"))
        if len(sub) >= 2:
            k, b0 = np.polyfit(sub["x"], sub["v"], 1)
            xs = np.linspace(sub["x"].min(), sub["x"].max(), 50)
            ax.plot(xs, k * xs + b0, color=ACCENT, linewidth=2,
                    label=L("线性趋势", "trend"))
        ax.set_xlabel(axis_col, color=TEXT, fontsize=10)
        # 年份/月份是离散整数，不加这一行 matplotlib 会画出 2007.5 这种刻度
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    ax.set_ylabel(num, color=TEXT, fontsize=10)
    if isinstance(per, (int, float)):
        ax.set_title(L(f"「{num}」随{axis_col}的走势（每{period}约 "
                       f"{'+' if per > 0 else ''}{_num(per, 2)}）",
                       f"{num} over {axis_col}"),
                     fontsize=11, color=TEXT, pad=10)
    else:
        ax.set_title(L(f"「{num}」随{axis_col}的走势", f"{num} over {axis_col}"),
                     fontsize=11, color=TEXT, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="best")


def _box_single(ax, df, d):
    num = d["columns"][0]
    v = pd.to_numeric(df[num], errors="coerce").dropna()

    bp = ax.boxplot([v.to_numpy()], patch_artist=True, widths=0.4,
                    medianprops={"color": ACCENT, "linewidth": 2},
                    flierprops={"marker": "o", "markersize": 7,
                                "markerfacecolor": ACCENT, "markeredgecolor": "none"})
    ax.set_xticks([1])
    ax.set_xticklabels([num])
    for patch in bp["boxes"]:
        patch.set_facecolor("#dbeafe")
        patch.set_edgecolor(PRIMARY)

    extreme = d["numbers"].get("最极端值")
    if isinstance(extreme, (int, float)):
        # 用抖动让标注点落在离群点旁边，而不是压在箱体上
        ax.annotate(L(f"离群：{_num(extreme)}", f"outlier: {_num(extreme)}"),
                    xy=(1 + 0.06, extreme), xytext=(1.18, extreme),
                    fontsize=9, color=ACCENT, va="center",
                    arrowprops={"arrowstyle": "-", "color": ACCENT, "linewidth": 1})

    ax.set_title(L(f"「{num}」的分布与离群点", f"distribution of {num}"),
                 fontsize=11, color=TEXT, pad=10)
    ax.tick_params(axis="x", labelsize=9)


RENDERERS = {
    "correlation": _scatter,
    "group_diff": _box_by_group,
    "trend": _line_trend,
    "outlier": _box_single,
}


def render(discovery, df, index, chart_dir=None):
    """
    给一条发现画图，返回图片相对路径（供 Markdown 引用）；失败返回 None。

    失败不抛异常——一张图画不出来不该让整份报告生不出来。

    chart_dir 可注入：Web 层每次分析要写到自己的任务目录，
    不能都塞进 results/charts（那样两个用户同时分析会互相覆盖）。
    """
    kind = discovery.get("kind")
    fn = RENDERERS.get(kind)
    if fn is None:
        return None

    chart_dir = chart_dir or CHART_DIR
    os.makedirs(chart_dir, exist_ok=True)
    filename = f"fig{index}_{kind}.png"
    full = os.path.join(chart_dir, filename)

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=130)
    fig.patch.set_facecolor("white")
    try:
        fn(ax, df, discovery)
        _style(ax)
        fig.tight_layout()
        fig.savefig(full, bbox_inches="tight", facecolor="white")
        # 报告在 results/ 下，用相对路径引用；这里必须用正斜杠，
        # Windows 的 os.path.join 会给出 "charts\fig1.png"，Markdown 解析不了。
        return f"charts/{filename}"
    except Exception as e:                          # noqa: BLE001
        print(f"    [警告] 第 {index} 张图生成失败：{type(e).__name__}: {e}")
        return None
    finally:
        plt.close(fig)


if __name__ == "__main__":
    # 单独跑一次，确认字体和四种图都能出
    import json

    import analyze as A

    with open(os.path.join(BASE_DIR, "results", "discoveries.json"), encoding="utf-8") as f:
        data = json.load(f)

    df = A.load_table(os.path.join(BASE_DIR, "data", "sample_survey.csv"))
    print(f"中文字体：{FONT or '未找到，图表使用英文标签'}")
    for i, d in enumerate(data["discoveries"], 1):
        p = render(d, df, i)
        print(f"  [{'✓' if p else '✗'}] {d['kind']:<12} -> {p}")
