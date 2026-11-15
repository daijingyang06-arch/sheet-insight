"""
引擎验收脚本
=============
用"带已知答案"的模拟数据检验发现引擎：我埋了 7 个规律，看引擎能不能找出来。

这类测试在机器学习项目里叫**回归测试（regression test）**：
  改完代码跑一遍，确认没有把原来能找出来的规律弄丢。
  这个项目里我已经因为"调排序规则"两次弄丢过重要发现，
  所以这个脚本是必需品，不是可选项。

用法：
    python verify.py
"""

import os

import numpy as np
import pandas as pd

import analyze as A

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "data", "sample_survey.csv")

# 埋进去的规律 → (检查函数, 说明)
# 每条检查函数接收"全部候选发现"，返回是否找到
CHECKS = [
    ("自习时长 ↔ 期末成绩 正相关（r≈0.56）",
     lambda ds: _has_corr(ds, "每周自习时长_小时", "期末成绩", sign=+1, min_r=0.45)),
    ("手机使用 ↔ 期末成绩 负相关（r≈-0.45）",
     lambda ds: _has_corr(ds, "日均手机使用时长_小时", "期末成绩", sign=-1, min_r=0.35)),
    ("专业类别 → 自习时长 有差异",
     lambda ds: _has_group(ds, "专业类别", "每周自习时长_小时")),
    ("专业类别 → 期末成绩 有差异",
     lambda ds: _has_group(ds, "专业类别", "期末成绩")),
    ("参加社团 → 期末成绩 有差异（弱效应）",
     lambda ds: _has_group(ds, "参加社团", "期末成绩")),
    ("自习时长 随时间上升",
     lambda ds: _has_trend(ds, "每周自习时长_小时", sign=+1)),
    ("自习时长 有 60 小时的异常值",
     lambda ds: _has_outlier(ds, "每周自习时长_小时", value=60.0)),
]


def _has_corr(ds, a, b, sign, min_r):
    for d in ds:
        if d["kind"] != "correlation" or set(d["columns"]) != {a, b}:
            continue
        r = d["numbers"]["r"]
        if (r > 0) == (sign > 0) and abs(r) >= min_r:
            return True
    return False


def _has_group(ds, cat, num):
    return any(d["kind"] == "group_diff" and d["columns"] == [cat, num] for d in ds)


def _has_trend(ds, num, sign):
    for d in ds:
        if d["kind"] == "trend" and d["columns"][1] == num:
            if (d["numbers"]["每周变化"] > 0) == (sign > 0):
                return True
    return False


def _has_outlier(ds, num, value):
    return any(d["kind"] == "outlier" and d["columns"] == [num]
               and abs(d["numbers"]["最极端值"] - value) < 1e-6 for d in ds)


def smoke_tests():
    """
    边界情况测试：真实用户的表格千奇百怪，不能一遇到怪表就崩。

    这几种是最容易让程序抛异常的：
      1. 只有数值列（没有分类列、没有日期列）
      2. 只有 12 行（样本量极小）
      3. 某列全是空值
      4. 列名带空格和全角字符
    """
    cases = []

    # 1. 纯数值
    rng = np.random.default_rng(0)
    cases.append(("纯数值表", pd.DataFrame({
        "身高": rng.normal(170, 8, 80),
        "体重": rng.normal(65, 10, 80),
    })))

    # 2. 极小样本
    cases.append(("12 行小表", pd.DataFrame({
        "组别": ["A", "B"] * 6,
        "分数": [60, 70, 62, 71, 58, 69, 61, 72, 59, 68, 63, 70],
    })))

    # 3. 有空列 + 全同列
    cases.append(("含空列/常量列", pd.DataFrame({
        "名称": [f"项目{i}" for i in range(50)],
        "空列": [None] * 50,
        "常量": ["X"] * 50,
        "数值": rng.normal(0, 1, 50),
    })))

    # 4. 列名带空格/全角
    cases.append(("列名不规整", pd.DataFrame({
        " 学号 ": [f"S{i}" for i in range(40)],
        "分数\u3000值": rng.normal(75, 5, 40),
        "类型 ": rng.choice(["甲", "乙"], 40),
    })))

    results = []
    for name, df in cases:
        try:
            roles = A.classify_columns(df)
            A.build_profile(df, roles)
            found = (A.find_correlations(df, roles) + A.find_group_differences(df, roles)
                     + A.find_trends(df, roles) + A.find_outliers(df, roles))
            A.select_top(found)
            results.append((name, True, f"{len(found)} 条发现"))
        except Exception as e:      # noqa: BLE001  —— 这里就是要抓住所有异常
            results.append((name, False, f"{type(e).__name__}: {e}"))
    return results


def main():
    df = A.load_table(DATA)
    roles = A.classify_columns(df)
    all_found = (A.find_correlations(df, roles) + A.find_group_differences(df, roles)
                 + A.find_trends(df, roles) + A.find_outliers(df, roles))

    print("=" * 66)
    print(f"召回率验收（候选发现共 {len(all_found)} 条）")
    print("=" * 66)
    passed = 0
    for name, fn in CHECKS:
        ok = fn(all_found)
        passed += ok
        print(f"  [{'✓' if ok else '✗'}] {name}")

    print(f"\n召回：{passed}/{len(CHECKS)} = {passed/len(CHECKS)*100:.0f}%")

    print()
    print("=" * 66)
    print("边界情况测试（真实表格千奇百怪，不能一遇怪表就崩）")
    print("=" * 66)
    smoke_ok = 0
    for name, ok, detail in smoke_tests():
        smoke_ok += ok
        print(f"  [{'✓' if ok else '✗'}] {name:<16} {detail}")
    print(f"\n通过：{smoke_ok}/{len(smoke_tests())}")

    total_ok = (passed == len(CHECKS)) and (smoke_ok == len(smoke_tests()))
    print("\n" + ("全部通过 ✓" if total_ok else "有未通过项 ✗ 需要修"))


if __name__ == "__main__":
    main()
