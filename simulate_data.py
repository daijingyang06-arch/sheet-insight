"""
模拟数据生成器（M1 阶段的"假表格"）
=====================================
作用：在拿到真实表格之前，先造一份"长得像真表"的数据，用来开发和验证发现引擎。

关键设计：这份数据里的规律是我**故意埋进去**的，所以我知道"正确答案"是什么。
  引擎跑出来如果能把埋的规律都找到 → 说明它真的有效；
  如果找不到或者找到了假的 → 说明算法有问题。
  这比拿真表去试更高效（真表我不知道答案，没法判断对错）。

埋进去的规律（用于验收）：
  1. 每周自习时长 ↔ 期末成绩：强正相关（r ≈ 0.6）
  2. 日均手机使用时长 ↔ 期末成绩：中等负相关（r ≈ -0.45）
  3. 是否参加社团 → 期末成绩：显著差异（参加的平均高约 2.5 分）
  4. 专业类别 → 每周自习时长：显著差异（理工 > 文史 > 艺术）
  5. 填表日期越晚 → 自习时长越高（临近考试，学生开始卷）
  6. 数据质量问题：备注缺失率高 / 问卷版本全相同 / 艺术类样本极少
  7. 异常值：有一条自习时长录入成 60 小时

用法：
    python simulate_data.py
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "data", "sample_survey.csv")

N = 300
SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    # ---------- 1. 人口学信息 ----------
    gender = rng.choice(["男", "女"], size=N, p=[0.48, 0.52])

    grade = rng.choice(["大一", "大二", "大三", "大四"], size=N, p=[0.30, 0.28, 0.25, 0.17])

    # 专业类别故意做成不均衡：艺术只有 4%，用来触发"类别极度不均衡"的质量提示
    major = rng.choice(["理工", "文史", "艺术"], size=N, p=[0.55, 0.41, 0.04])
    # 保证艺术类至少有几个人，否则统计检验会失败
    major[:9] = "艺术"

    # ---------- 2. 行为数据 ----------
    # 专业类别 → 自习时长（理工最卷，艺术最少）—— 埋的规律 4
    major_offset = pd.Series(major).map({"理工": 2.5, "文史": -1.0, "艺术": -4.0}).to_numpy()
    study_hours = rng.normal(12, 4.5, size=N) + major_offset
    study_hours = np.clip(study_hours, 0, 40)

    sleep_hours = np.clip(rng.normal(6.8, 0.9, size=N), 3.5, 10)

    breakfast = rng.integers(0, 8, size=N).astype(float)

    # 手机使用时长与自习时长轻微负相关（现实中确实如此）
    phone_hours = np.clip(rng.normal(5.5, 1.8, size=N) - 0.08 * (study_hours - 12), 0.5, 14)

    in_club = rng.choice(["是", "否"], size=N, p=[0.45, 0.55])

    # ---------- 3. 期末成绩（由上面几个变量决定）----------
    # 这是整份数据的核心：成绩是由"自习、睡眠、手机、社团"共同决定的，
    # 系数就是埋进去的规律强度，方便我反推引擎找得对不对。
    score = (
        55
        + 1.20 * (study_hours - 12)          # 规律 1：自习最重要
        + 2.00 * (sleep_hours - 6.8)         # 睡眠也有影响
        - 1.80 * (phone_hours - 5.5)         # 规律 2：手机是负向因素
        + 2.50 * (in_club == "是")           # 规律 3：社团加分
        + 0.60 * (breakfast - 3)             # 早餐影响较弱
        + rng.normal(0, 4.0, size=N)         # 个人差异（噪声）
    )
    score = np.clip(score, 20, 100).round(1)

    # ---------- 4. 满意度（与成绩相关的主观评分）----------
    satisfaction = np.clip(np.round((score - 40) / 15 + rng.normal(0, 0.6, size=N)), 1, 5).astype(int)

    # ---------- 5. 填表日期（越晚越卷）—— 埋的规律 5 ----------
    # 在 30 天内随机分布，日期越晚，自习时长加一点（模拟期末临近）
    days = rng.integers(0, 30, size=N)
    study_hours = np.clip(study_hours + days * 0.12, 0, 40).round(1)
    submit_date = pd.Timestamp("2026-10-15") + pd.to_timedelta(days, unit="D")

    # ---------- 6. 异常值：一条录入错误 ----------
    study_hours[7] = 60.0

    # ---------- 7. 组装 ----------
    df = pd.DataFrame({
        "学号": [f"S2024{i:04d}" for i in range(1, N + 1)],
        "填表日期": submit_date.strftime("%Y-%m-%d"),
        "性别": gender,
        "年级": grade,
        "专业类别": major,
        "每周自习时长_小时": study_hours.round(1),
        "日均睡眠时长_小时": sleep_hours.round(1),
        "每周早餐次数": breakfast.astype(int),
        "日均手机使用时长_小时": phone_hours.round(1),
        "参加社团": in_club,
        "期末成绩": score,
        "学习满意度_1到5": satisfaction,
        # 备注故意做成 62% 缺失（真实问卷里开放式问题都这样）
        "备注": [rng.choice(["", "希望多一些自习室", "作息太乱需要调整", "无", "想参加更多活动"])
                if rng.random() < 0.38 else np.nan for _ in range(N)],
        # 全列相同的"废列"（很多真实表格里都有这种列）
        "问卷版本": "v1",
    })

    # 造一点列名不规整：前后空格（真实表格非常常见，很容易让程序崩）
    df.columns = [c.strip() for c in df.columns]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"已生成模拟数据：{OUT_PATH}")
    print(f"行数 {len(df)}，列数 {len(df.columns)}")
    print("\n各列概览：")
    for c in df.columns:
        miss = df[c].isna().mean() * 100
        print(f"  {c:<22} 缺失 {miss:5.1f}%  唯一值 {df[c].nunique():>3}")


if __name__ == "__main__":
    main()
