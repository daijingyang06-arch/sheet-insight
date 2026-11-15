"""
M4 · Web 界面 —— 上传表格，拿到一份能分享的分析报告
=====================================================
设计上刻意做了三个取舍，都跟"同学真的会用"有关：

1. **不做账号。** 打开链接就能用。没有账号，就不存在"谁的表格存在我的服务器上"
   这个责任问题——所以原始表格分析完立刻删，只留报告，且 24 小时过期。

2. **结果页 URL 就是分享链接。** 不用点"分享"按钮生成什么，复制地址栏即可。
   id 是 12 位随机串（不可猜），因为这个页面可能包含同学的真实数据。

3. **分析与界面彻底分开。** 这里只负责"收文件 → 调 narrate.run_pipeline → 存结果"，
   一个字的统计逻辑都没有。否则就会出现"命令行结论对、网页上结论不一样"。

启动：
    python webapp.py                 # 默认 http://127.0.0.1:8000
    python webapp.py --port 8080
    uvicorn webapp:app --host 0.0.0.0 --port 8000
"""

import argparse
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
from urllib.parse import unquote

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

import share_store
from narrate import run_pipeline
from web.present import build_cards

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

MAX_UPLOAD_MB = float(os.environ.get("SHEET_INSIGHT_MAX_MB", "20"))
ALLOWED_EXT = {".csv", ".xlsx", ".xls"}
BACKEND = os.environ.get("SHEET_INSIGHT_BACKEND", "auto")

# matplotlib 的 pyplot 是全局状态机，多线程同时画图会串图（甚至崩）。
# 所以分析默认串行：一个人等 3 秒，好过两个人拿到对方的图。
_analysis_lock = threading.Semaphore(int(os.environ.get("SHEET_INSIGHT_CONCURRENCY", "1")))

app = FastAPI(title="表格分析助手", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(WEB_DIR, "templates"))


class BusinessError(Exception):
    """能直接展示给用户看的错误（说人话，不带堆栈）。"""


# ============================================================
# 工具
# ============================================================
def clean_filename(raw):
    """
    只留文件名本身，去掉路径、控制字符，限长 80。

    这不是洁癖：文件名会出现在报告标题和页面里，还会决定临时文件的扩展名。
    带 ../ 的名字等于把服务器目录结构交出去。
    """
    name = unquote(raw or "").strip()
    name = name.replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable())
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:80]


def _analyze_blocking(data_path, work_dir):
    """在后台线程里跑完整流水线。拿不到分析名额就排队，不拒绝。"""
    if not _analysis_lock.acquire(timeout=120):
        raise TimeoutError("前面还有分析在排队，稍后再试。")
    try:
        return run_pipeline(data_path, out_dir=work_dir, backend=BACKEND, quiet=True)
    finally:
        _analysis_lock.release()


async def analyze_bytes(raw, filename):
    """
    收下文件 → 跑分析 → 存成可分享的结果。返回 (share_id, meta) 或抛 BusinessError。

    原始文件写在系统临时目录，跑完立刻删；它不会进入 shared/，
    也不会被任何日志记录下来。
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise BusinessError(f"只支持 .csv / .xlsx / .xls，收到的是「{ext or '无扩展名'}」。")
    if not raw:
        raise BusinessError("文件是空的，没有内容可以分析。")
    if len(raw) > MAX_UPLOAD_MB * 1e6:
        raise BusinessError(f"文件超过 {MAX_UPLOAD_MB:.0f} MB。请先删除无关列或抽样后再上传。")

    share_store.purge_expired()
    share_id, work_dir = share_store.reserve()

    tmpdir = tempfile.mkdtemp(prefix="sheet-insight-")
    try:
        tmp_path = os.path.join(tmpdir, "upload" + ext)
        with open(tmp_path, "wb") as f:
            f.write(raw)

        t0 = time.time()
        result = await run_in_threadpool(_analyze_blocking, tmp_path, work_dir)
        elapsed = round(time.time() - t0, 1)

        share_store.save(filename, result, share_id=share_id,
                         extra={"elapsed_sec": elapsed})
        return share_id, share_store.load(share_id)
    except BusinessError:
        share_store.discard(share_id)
        raise
    except ValueError as e:
        share_store.discard(share_id)
        raise BusinessError(str(e)) from None
    except Exception as e:                                  # noqa: BLE001
        share_store.discard(share_id)
        traceback.print_exc()
        raise BusinessError(
            f"分析失败（{type(e).__name__}）。这份表格可能缺少可分析的数值列，"
            f"或者结构比较特殊——换一份试试，或把报错反馈给我。"
        ) from None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# 路由
# ============================================================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "max_mb": f"{MAX_UPLOAD_MB:.0f}",
        "ttl_hours": f"{share_store.TTL_HOURS:.0f}",
    })


@app.post("/analyze")
async def analyze(request: Request):
    filename = clean_filename(request.headers.get("x-filename", ""))
    if not filename:
        return JSONResponse({"error": "没有收到文件名，请重新选择文件。"}, status_code=400)

    raw = await request.body()
    try:
        share_id, meta = await analyze_bytes(raw, filename)
    except BusinessError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return JSONResponse({"id": share_id, "url": f"/r/{share_id}",
                         "findings": meta["n_findings"]})


@app.get("/demo")
async def demo(request: Request):
    """示例数据：访客不传文件也能看到产出长什么样。"""
    sample = os.path.join(BASE_DIR, "data", "sample_survey.csv")
    if not os.path.isfile(sample):
        return templates.TemplateResponse(request, "message.html", {
            "title": "示例数据缺失",
            "message": "仓库里没有 data/sample_survey.csv，先运行 python simulate_data.py 生成。",
        }, status_code=500)

    with open(sample, "rb") as f:
        raw = f.read()
    try:
        share_id, _meta = await analyze_bytes(raw, "示例问卷（300 份）.csv")
    except BusinessError as e:
        return templates.TemplateResponse(request, "message.html", {
            "title": "示例分析失败", "message": str(e),
        }, status_code=500)
    return Response(status_code=303, headers={"Location": f"/r/{share_id}"})


@app.get("/r/{share_id}", response_class=HTMLResponse)
def result_page(request: Request, share_id: str):
    try:
        meta = share_store.load(share_id)
    except share_store.Expired:
        return templates.TemplateResponse(request, "message.html", {
            "title": "链接已失效",
            "message": f"这份结果已经过期了。为了不长期保存别人的数据，"
                       f"所有结果只保留 {share_store.TTL_HOURS:.0f} 小时，之后自动删除。"
                       f"重新上传一次表格即可。",
        }, status_code=404)

    return templates.TemplateResponse(request, "result.html", {
        "meta": meta,
        "cards": build_cards(meta["records"]),
        "ttl_hours": f"{share_store.TTL_HOURS:.0f}",
    })


@app.get("/r/{share_id}/report.md")
def download_report(share_id: str):
    try:
        meta = share_store.load(share_id)
    except share_store.Expired:
        return JSONResponse({"error": "链接已失效"}, status_code=404)
    stem = os.path.splitext(meta["source"])[0] or "分析报告"
    # 中文文件名交给 Starlette 走 RFC 5987（filename*=UTF-8''），自己拼会乱码
    return FileResponse(meta["report_path"], media_type="text/markdown; charset=utf-8",
                        filename=f"{stem}_分析报告.md")


@app.get("/r/{share_id}/charts/{filename}")
def chart(share_id: str, filename: str):
    try:
        path = share_store.chart_path(share_id, filename)
    except share_store.Expired:
        return JSONResponse({"error": "图表不存在或已过期"}, status_code=404)
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600"})


@app.get("/healthz")
def healthz():
    return {"ok": True, **share_store.usage(), "ttl_hours": share_store.TTL_HOURS}


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# ============================================================
# 本地启动
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="表格分析助手 · Web 界面")
    parser.add_argument("--host", default="127.0.0.1", help="默认只监听本机")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="改代码自动重启（开发用）")
    args = parser.parse_args()

    import uvicorn
    print(f"\n  表格分析助手  →  http://{args.host}:{args.port}\n")
    print(f"  结果保留 {share_store.TTL_HOURS:.0f} 小时，存在 {share_store.SHARE_DIR}")
    print(f"  单文件上限 {MAX_UPLOAD_MB:.0f} MB\n")
    uvicorn.run("webapp:app" if args.reload else app,
                host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
