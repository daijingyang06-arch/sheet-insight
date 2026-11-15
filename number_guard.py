"""
数字守卫（Number Guard）—— M3 的生死线
========================================
整个产品最核心的一句承诺是：**所有数字都来自真实计算，不是大模型编的。**

但"承诺"不是机制。大模型天生爱改写数字：
  给它 r=0.56，它可能写成"相关系数 0.78，属于强相关"；
  给它 300 行数据，它可能写成"在 500 份问卷中"。
这些错误看起来毫不起眼，但一旦进了报告，整个产品就报废了。

所以这里做三件事：
  1. build_whitelist(discovery)  给每条发现算出"允许出现的数字"集合
  2. extract_numbers(text)       把模型输出里所有数字抓出来
  3. check_text(text, whitelist) 逐个核对，返回违规数字

设计上有个关键取舍：**白名单要允许"等价写法"，但不能允许"编造"。**
  - 0.619 写成 0.62 / 0.6 都是合理的四舍五入 → 要放过
  - 0.1502 写成 15%（效应量换成百分比）→ 要放过
  - 0.56 写成 0.78 → 必须拦下来

用法：
    from number_guard import build_whitelist, check_text
    bad = check_text(text, build_whitelist(discovery))
"""

import re

# 全角数字/标点是中文输入法的高频产物，先归一化再解析
_FULLWIDTH_MAP = {ord("０") + i: ord("0") + i for i in range(10)}
_FULLWIDTH_MAP.update({ord("．"): ord("."), ord("－"): ord("-"), ord("％"): ord("%")})

# 匹配数字。前面的负向后视 `(?<![\d.])` 很关键：
#   日期 "2026-10-15" 里的 "-10" 不应该被当成负数 -10，
#   因为 "-" 前面是数字 6，正则就不会从这里开始匹配，而是从 1 开始匹配出 "10"。
NUM_RE = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?")

# p 值的"档位"写法。模型很爱写"p<0.001"，而真实 p=3.18e-11，
# 数值上差很远但语义上完全正确，所以这类固定档位单独放行。
P_BUCKETS = {0.05, 0.01, 0.001, 0.0001, 0.00001}

# 一处舍入最多能偏离原值多少（相对）。
# 为什么定 5%：这是"舍入"和"另一个数"的分界线。
#   0.619 → 0.62 偏 0.16%，是舍入，放行
#   0.619 → 0.6  偏 3%，是舍入，放行
#   0.619 → 1    偏 61%，这不是舍入，是把"中等相关"说成"完全相关"，必须拦
# 第一版没有这条限制，于是 "r=0.619" 允许写成 "r=1"、"η²=0.1502" 允许写成 "0"，
# 验收脚本的召回率只有 73%——正是这条规则把漏判补上了。
MAX_ROUNDING_ERROR = 0.05


def normalize(text):
    """把全角数字、全角小数点、全角负号、全角百分号转成半角。"""
    return str(text).translate(_FULLWIDTH_MAP)


def extract_numbers(text):
    """
    从文本里抽出所有数字。

    返回 [(原始写法, 数值, 小数位数), ...]
      小数位数用来决定"允许多大误差"——写成 0.6 时误差容忍 0.05，
      写成 0.619 时只容忍 0.0005。
    """
    out = []
    for m in NUM_RE.finditer(normalize(text)):
        tok = m.group(0)
        try:
            value = float(tok)
        except ValueError:
            continue
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        out.append((tok, value, decimals))
    return out


def _walk(obj):
    """递归遍历任意嵌套结构，产出其中出现的所有数字。"""
    if isinstance(obj, bool):        # True/False 不是数字，别混进来
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        # 字符串里也可能藏数字（例如"起止": "2026-10-15 ~ 2026-11-13"）
        for _, v, _d in extract_numbers(obj):
            yield v
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(k)
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)


def _variants(value):
    """
    一个数值的"合法等价写法"：
      · 各精度的四舍五入（但只保留真实合理的舍入，见 MAX_ROUNDING_ERROR）
      · 绝对值（r = -0.453 常被写成"相关强度 0.45"）
      · 百分比形式（η² = 0.1502 → 15%）

    这不是在放宽标准——这几种写法在语义上等价。
    但"0.56 写成 0.78"仍然会被拦下，因为 0.78 不在任何等价写法里。
    """
    out = set()
    for base in (value, abs(value)):
        if base == 0:
            out.add(0.0)
            continue
        for k in range(0, 5):
            cand = round(base, k)
            # 只把"看起来还是同一个数"的舍入当成等价写法
            if abs(cand - base) <= MAX_ROUNDING_ERROR * abs(base) + 1e-12:
                out.add(cand)
        if abs(base) < 1:
            pct = base * 100          # 0.1502 → 15.02（模型可能写成 15%）
            for k in range(0, 3):
                cand = round(pct, k)
                # 阈值必须用 abs()：pct 可能是负数（r=-0.453 → -45.3%），
                # 负数乘出来是个负阈值，比较永远不成立，负百分比就会被误杀。
                if abs(cand - pct) <= MAX_ROUNDING_ERROR * abs(pct) + 1e-12:
                    out.add(cand)
    return out


def build_whitelist(discovery):
    """
    为一条发现构造允许出现的数字集合。

    来源有三处，缺一不可：
      1. numbers 字段（统计层的真实结果，递归展开嵌套字典如「组均值」）
      2. columns 字段（列名里可能含数字，如「学习满意度_1到5」，正文会引用它）
      3. 上述数字的等价写法（四舍五入、百分比、p 值档位）
    """
    whitelist = set()

    for v in _walk(discovery.get("numbers", {})):
        whitelist |= _variants(v)

    for col in discovery.get("columns", []):
        for _, v, _d in extract_numbers(str(col)):
            whitelist |= _variants(v)

    if "p" in discovery.get("numbers", {}):
        whitelist |= P_BUCKETS

    return whitelist


def _close(x, w):
    """
    判断文本里的数 x 和白名单里的数 w 是否算"同一个数"。

    规则只有一条：**相对偏差不超过 5%**。

    为什么不用"按小数位数给绝对容差"（第一版的做法）？
      那样算的话，写 "0.2" 会被当成 η²=0.1502 的"一位小数舍入"而放行——
      偏差 33%，但绝对值只差 0.0498，正好落在 ±0.05 的容差里。
      于是"效应量 0.15"被说成"0.2"，守卫却看不出来。
    换成相对偏差后，同一个数不管写几位小数，判断标准都一样。
    """
    if w == 0:
        return abs(x) < 1e-9
    return abs(x - w) <= MAX_ROUNDING_ERROR * abs(w) + 1e-12


def check_text(text, whitelist):
    """
    核对文本里的每个数字是否都在白名单里。

    返回违规数字的原始写法列表（去重、保持出现顺序）。
    空列表 = 通过。
    """
    bad, seen = [], set()
    for tok, value, _decimals in extract_numbers(text):
        if any(_close(value, w) for w in whitelist):
            continue
        if tok not in seen:
            seen.add(tok)
            bad.append(tok)
    return bad


def allowed_numbers_hint(discovery):
    """
    给提示词用的"允许使用的数字"，只列原始值（不展开等价写法）。

    为什么不直接把白名单整个丢给模型？因为展开后有两三百个数，
    模型会被淹没，反而更容易写错。给 5~10 个关键值就够了。
    """
    items = []

    def add(value):
        s = f"{value:g}"
        if s not in items:
            items.append(s)

    for v in _walk(discovery.get("numbers", {})):
        add(v)

    return items
