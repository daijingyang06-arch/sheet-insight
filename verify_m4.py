"""
M4 验收 —— Web 界面与分享存储
==============================
前三套验收守的是「结论对不对」，这一套守的是「别人能不能用它干坏事」。

界面比命令行多出三个风险面，逐条验：

  1. **分享链接可不可猜。** 结果页可能包含同学的真实数据，id 是自增数字就等于
     让任何人顺着编号翻遍所有人的报告。
  2. **id 会不会被拼成路径。** /r/../../etc/passwd 这种请求必须在校验层就被挡住，
     而不是指望操作系统。
  3. **过期是不是真的删。** 说好 24 小时删，就得能证明文件不在了。

用法：
    python verify_m4.py
"""

import io
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from urllib.parse import quote

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 测试期间把结果目录挪到临时位置，别污染真实 shared/
TEST_SHARE = os.path.join(BASE_DIR, "_verify_share")
shutil.rmtree(TEST_SHARE, ignore_errors=True)
os.environ["SHEET_INSIGHT_SHARE_DIR"] = TEST_SHARE

import share_store                      # noqa: E402
from narrate import run_pipeline        # noqa: E402

SAMPLE = os.path.join(BASE_DIR, "data", "sample_survey.csv")

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  [{'✓' if cond else '✗'}] {name}" + (f"    {detail}" if detail else ""))


def expect_raise(name, fn, exc=Exception):
    try:
        fn()
    except exc as e:
        check(name, True, f"拒绝理由：{e}")
    except Exception as e:                              # noqa: BLE001
        check(name, False, f"抛了意外异常 {type(e).__name__}: {e}")
    else:
        check(name, False, "没有拒绝")


def main():
    if not os.path.isfile(SAMPLE):
        print(f"缺少示例数据 {SAMPLE}，先运行 python simulate_data.py")
        return 1

    # ------------------------------------------------------------
    print("\n【1】分享 id 不可猜")
    ids = {share_store.new_id() for _ in range(200)}
    check("200 次生成无重复", len(ids) == 200)
    check("id 长度 ≥ 12", all(len(i) >= 12 for i in ids), f"样例 {list(ids)[0]}")

    # ------------------------------------------------------------
    print("\n【2】非法 id 不得拼进路径")
    for bad in ["../../etc/passwd", "..%2f..%2fetc", "a" * 3, "", "abc/def", "abc def"]:
        expect_raise(f"拒绝 {bad!r}", lambda b=bad: share_store.load(b), share_store.Expired)

    # ------------------------------------------------------------
    print("\n【3】完整流程：分析 → 存 → 读")
    share_id, work_dir = share_store.reserve()
    t0 = time.time()
    result = run_pipeline(SAMPLE, out_dir=work_dir, backend="template", quiet=True)
    elapsed = round(time.time() - t0, 1)
    check("流水线产出报告", len(result["markdown"]) > 500, f"{len(result['markdown'])} 字，{elapsed}s")
    check("产出发现", len(result["records"]) > 0, f"{len(result['records'])} 条")
    check("数字零违规", result["stats"]["final_unauthorized"] == 0)

    share_store.save("示例问卷（300 份）.csv", result, share_id=share_id, extra={"elapsed_sec": elapsed})
    meta = share_store.load(share_id)
    check("meta 可读回", meta["id"] == share_id and meta["n_findings"] == len(result["records"]))
    check("记录有效期", 0 < meta["remaining_hours"] <= share_store.TTL_HOURS, f"{meta['remaining_hours']}h")

    md = share_store.report_text(share_id)
    check("报告可取出且内容一致", md[:40] == result["markdown"][:40])

    chart_files = sorted(os.listdir(meta["chart_dir"]))
    check("图表已落盘", len(chart_files) > 0, f"{len(chart_files)} 张")

    # 图表文件名白名单：只允许 fig<数字>_<类型>.png
    if chart_files:
        real = chart_files[0]
        check("白名单放行正常图名", os.path.isfile(share_store.chart_path(share_id, real)), real)
    for bad in ["../report.md", "..%2f..%2fmeta.json", "meta.json", "fig1.png", "/etc/passwd"]:
        expect_raise(f"拒绝图表名 {bad!r}", lambda b=bad: share_store.chart_path(share_id, b),
                     share_store.Expired)

    # ------------------------------------------------------------
    print("\n【4】过期即删除")
    expired_id, expired_dir = share_store.reserve()
    with open(os.path.join(expired_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"id": expired_id, "source": "x.csv",
                   "created_at": "2020-01-01T00:00:00",
                   "expires_at": (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds"),
                   "records": [], "stats": {}, "profile": {"rows": 1, "cols": 1},
                   "issues": [], "n_findings": 0}, f, ensure_ascii=False)
    open(os.path.join(expired_dir, "report.md"), "w").write("old")
    expect_raise("过期结果不可读", lambda: share_store.load(expired_id), share_store.Expired)
    check("过期文件已被删除", not os.path.exists(expired_dir))

    # 一个正常结果不该被误清
    removed = share_store.purge_expired()
    check("清理不误伤有效结果", os.path.isdir(os.path.join(TEST_SHARE, share_id)), f"清掉 {removed} 个")

    # ------------------------------------------------------------
    print("\n【5】展示层口径（不装 Web 依赖也能验）")
    from web.present import build_cards, fmt_num, strip_echo

    check("大数字不用科学计数法", fmt_num(2502139.72) == "2,502,140", fmt_num(2502139.72))
    check("小数去掉多余零", fmt_num(0.973) == "0.973", fmt_num(0.973))
    check("整数不带小数点", fmt_num(300) == "300", fmt_num(300))
    check("极小 p 值用科学计数法", "e-" in fmt_num(5.65e-20), fmt_num(5.65e-20))
    check("字典值不被 str() 化",
          "{" not in fmt_num({"理工": 16.7, "文史": 12.27}),
          fmt_num({"理工": 16.7, "文史": 12.27}))

    # 卡片上标题 + 正文并排，正文第一句若只是标题的另一种说法就该去掉
    head = "不同「专业类别」的「每周自习时长」存在显著差异：理工最高（16.7），艺术最低（11.9）"
    body = "不同「专业类别」的「每周自习时长」存在显著差异：「理工」组最高（平均 16.7），「艺术」组最低（11.9）。效应量 η² = 0.1502。"
    trimmed = strip_echo(head, body)
    check("重复的正文首句被去掉", trimmed.startswith("效应量") or trimmed.startswith("「理工」"),
          trimmed[:26])
    check("去重后正文没被清空", len(trimmed) > 10)
    check("内容不重复时不动正文",
          strip_echo("完全无关的标题", "这句话讲的是另一件事。这里是第二句。") ==
          "这句话讲的是另一件事。这里是第二句。")
    # 相关类模板：标题说"呈中度正相关（r=0.62）"，首句说"之间存在正相关（r = 0.619）"，
    # 实测重合度 0.70 —— 也属于回声句，必须去掉，否则卡片上同一句话念两遍
    head2 = "「期末成绩」与「学习满意度_1到5」呈中度正相关（r=0.62）"
    body2 = "「期末成绩」与「学习满意度_1到5」之间存在正相关（r = 0.619，样本 300 条）。相关系数 0.62 属于中等程度。"
    check("相关类的回声句也能识别", not strip_echo(head2, body2).startswith("「期末成绩」与"),
          strip_echo(head2, body2)[:24])

    cards = build_cards(result["records"])
    check("每条记录都生成卡片", len(cards) == len(result["records"]))
    check("卡片带类型中文名", all(c["kind_cn"] for c in cards))
    check("卡片带关键指标", all(len(c["metrics"]) > 0 for c in cards),
          f"首卡 {len(cards[0]['metrics'])} 项")
    check("指标标签已翻译成中文",
          any(m["label"] == "相关系数 r" for c in cards for m in c["metrics"]),
          str([m["label"] for m in cards[0]["metrics"]][:3]))
    check("长指标值自动占两格",
          any(m["wide"] for c in cards for m in c["metrics"]),
          "各组均值" if any(m["wide"] for c in cards for m in c["metrics"]) else "无")
    check("图表名不含路径", all((c["chart"] or "").find("/") < 0 for c in cards))

    # ------------------------------------------------------------
    print("\n【6】Web 层（需要 fastapi）")
    try:
        from fastapi.testclient import TestClient
        import webapp
    except ImportError as e:
        print(f"  [–] 跳过：{e}")
        print("      装好后再跑一次：python -m pip install fastapi uvicorn httpx")
    else:
        client = TestClient(webapp.app)

        r = client.get("/")
        check("首页可访问", r.status_code == 200 and "表格分析助手" in r.text)
        check("首页含上传入口", 'id="file"' in r.text)

        with open(SAMPLE, "rb") as f:
            raw = f.read()
        r = client.post("/analyze", content=raw,
                        headers={"X-Filename": quote("示例问卷.csv"),
                                 "Content-Type": "application/octet-stream"})
        check("上传并分析成功", r.status_code == 200, r.text[:120])
        jid = r.json().get("id") if r.status_code == 200 else ""
        check("返回不可猜的 id", bool(jid) and len(jid) >= 12, jid)

        if jid:
            r = client.get(f"/r/{jid}")
            page = r.text
            check("结果页可访问（即分享链接）", r.status_code == 200)
            check("结果页含结论卡片", "card-title" in page)
            check("结果页声明数字校验", "数字校验通过" in page)
            check("结果页标注过期时间", "小时后自动失效" in page)

            r = client.get(f"/r/{jid}/report.md")
            check("报告可下载", r.status_code == 200 and "# 数据分析报告" in r.text)
            check("下载带中文文件名", "filename*=utf-8''" in r.headers.get("content-disposition", "").lower())

            figs = re.findall(r'/r/[A-Za-z0-9_-]+/charts/(fig\d+_[a-z_]+\.png)', page)
            check("页面引用了图表", len(figs) > 0, f"{len(figs)} 张")
            if figs:
                r = client.get(f"/r/{jid}/charts/{figs[0]}")
                check("图表可访问且是 PNG", r.status_code == 200 and r.content[:4] == b"\x89PNG")

            r = client.get(f"/r/{jid}/charts/../../meta.json")
            check("穿越请求被拒", r.status_code == 404)

        # 非法输入
        r = client.post("/analyze", content=b"hello", headers={"X-Filename": quote("a.txt")})
        check("拒绝非表格扩展名", r.status_code == 400, r.json().get("error", "")[:40])

        r = client.post("/analyze", content=b"a,b\n1,2\n", headers={"X-Filename": quote("空的.csv")})
        check("2 行 CSV 不崩（能出结论或友好提示）", r.status_code in (200, 400),
              f"HTTP {r.status_code}")

        r = client.post("/analyze", content=b"x", headers={})
        check("缺文件名被拒", r.status_code == 400)

        r = client.get("/r/notarealid12345")
        check("未知 id 返回 404", r.status_code == 404)

        r = client.get("/healthz")
        check("健康检查可用", r.status_code == 200 and r.json().get("ok") is True)

        r = client.get("/demo", follow_redirects=False)
        check("示例数据一键可跑", r.status_code in (303, 307), f"HTTP {r.status_code}")

    # ------------------------------------------------------------
    print("\n" + "=" * 62)
    total = len(passed) + len(failed)
    print(f"共 {total} 项检查，通过 {len(passed)} 项")
    if failed:
        print("失败：")
        for f in failed:
            print(f"  - {f}")
        print("存在失败 ✗")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    code = main()
    shutil.rmtree(TEST_SHARE, ignore_errors=True)
    sys.exit(code)
