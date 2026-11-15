"""
结果页的展示层纯函数 —— 不依赖 Web 框架
=======================================
单独拆出来的原因很实际：这些函数决定"页面上看到什么"，是最该被测试的部分，
但一旦放进 webapp.py，导入它就需要 fastapi，验收脚本就得先装一堆东西。
放到这里之后，`python verify_m4.py` 不装任何 Web 依赖也能验展示口径。
"""

import os

KIND_CN = {
    "correlation": "关联关系",
    "group_diff": "组间差异",
    "trend": "时间趋势",
    "outlier": "异常值",
}

# 统计层的键名是给程序用的（r / eta2 / 均值_a），直接显示在页面上像开发调试。
# 这里做一次"翻译"，但数值一个都不改——口径必须和报告完全一致。
METRIC_LABEL = {
    "r": "相关系数 r",
    "p": "显著性 p",
    "n": "样本量 n",
    "eta2": "效应量 η²",
    "均值_a": "前者均值",
    "均值_b": "后者均值",
    "组均值": "各组均值",
    "样本数": "各组样本数",
    "最高组": "最高组",
    "最低组": "最低组",
    "周期": "时间轴单位",
    "每周期变化": "每周期变化",
    "全期变化": "全期累计变化",
    "跨度": "跨度",
    "起止": "起止",
    "离群个数": "离群值个数",
    "最极端值": "最极端值",
    "偏离倍数": "偏离中位数倍数",
    "中位数": "中位数",
}


def fmt_num(v):
    """页面上的数字格式。跟报告口径一致：不出现 2.50214e+06 这种。"""
    if isinstance(v, bool) or v is None:
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        # 组间差异的「组均值」是个字典，直接 str() 会渲染成 {'理工': 16.7, ...}
        return " · ".join(f"{k} {fmt_num(x)}" for k, x in v.items())
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        return f"{x:.2e}"
    if abs(x) >= 100000:
        return f"{x:,.0f}"
    return f"{x:,.3f}".rstrip("0").rstrip(".")


def _char_overlap(a, b):
    """两个字符串共用多少比例的字符（粗粒度相似度，够用来判断"是不是同一句话"）。"""
    if not b:
        return 0.0
    return len(set(a) & set(b)) / len(set(b))


def strip_echo(headline, text):
    """
    卡片上标题和正文并排显示，而模板正文的第一句往往就是标题的另一种说法。
    两句都念一遍会很啰嗦，所以：第一句和标题高度重合时，去掉它。

    阈值 0.66 是量出来的，不是猜的：拿示例数据跑出来的 6 条结论里，
    「回声句」的重合度落在 0.70～0.88（相关 0.70、趋势 0.76、组间 0.88、离群 0.87），
    而内容不同的句子远低于这个区间。只在重合度很高时动手，
    并且至少给正文留一句——宁可重复，不可只剩标题。
    """
    first, sep, rest = text.partition("。")
    if not sep or not rest.strip():
        return text
    if _char_overlap(headline, first) >= 0.66:
        return rest.strip()
    return text


def build_cards(records):
    """把 narration 记录整理成页面要的卡片结构（模板里不做逻辑）。"""
    cards = []
    for r in records:
        nums = r.get("numbers") or {}
        chart = r.get("chart")
        headline = r.get("headline", "")
        metrics = []
        for k, v in nums.items():
            value = fmt_num(v)
            metrics.append({
                "label": METRIC_LABEL.get(k, k),
                "value": value,
                # 值太长（如各组均值）就横跨两格，否则会被挤成两行
                "wide": len(value) > 16,
            })
        cards.append({
            "index": r.get("index"),
            "kind": r.get("kind"),
            "kind_cn": KIND_CN.get(r.get("kind"), r.get("kind")),
            "headline": headline,
            "text": strip_echo(headline, r.get("text", "")),
            "chart": os.path.basename(chart) if chart else None,
            "metrics": metrics,
            "backend": r.get("backend"),
        })
    return cards
