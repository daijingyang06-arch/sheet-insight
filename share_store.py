"""
分享存储 —— 无账号，但结果能给别人看
====================================
产品需求是「同学之间传个链接就能看」，不要账号。这带来三个必须自己解决的问题：

1. **id 要不可猜。** 结果里可能有同学的实验数据，用自增数字（/r/1、/r/2）
   等于让任何人顺着编号翻遍所有人的报告。这里用 `secrets` 生成的随机串。

2. **不能永久存。** 没有账号就没有"这是谁的数据"的依据，长期保存既无必要也不负责任。
   所有结果 24 小时后过期，读取时顺手清理。

3. **原始表格不存。** 只保留报告和图表。表格是用户的原始资产，
   分析完立刻删除——这是"数据最小化"，也是这个产品敢公开让人试的前提。

目录结构（每个 id 一个文件夹，互不干扰）：
    shared/<id>/meta.json     元信息（文件名、行列数、发现数、过期时间）
    shared/<id>/report.md     成品报告
    shared/<id>/charts/*.png  图表
"""

import json
import os
import re
import secrets
import shutil
import time
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SHARE_DIR = os.environ.get("SHEET_INSIGHT_SHARE_DIR") or os.path.join(BASE_DIR, "shared")
TTL_HOURS = float(os.environ.get("SHEET_INSIGHT_TTL_HOURS", "24"))

# 只接受这种形状的 id。校验不是"防君子"，是因为 id 会直接拼进文件路径，
# 不校验就等于把 ../ 送给任何人。
ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,32}$")


class Expired(Exception):
    """结果过期或从未存在——对外统一说"链接已失效"，不区分两者。"""


def _now():
    return time.time()


def new_id():
    """12 位随机串（token_urlsafe 自带 URL 安全字符集，不用再转义）。"""
    return secrets.token_urlsafe(9)


def _dir(share_id):
    if not ID_RE.match(share_id or ""):
        raise Expired("非法 id")
    return os.path.join(SHARE_DIR, share_id)


def reserve():
    """
    先占一个 id 并把目录建好，再跑分析。

    顺序反了会有个麻烦：图表必须直接画进这个 id 的目录里，
    否则跑完还得把一堆 png 搬家一次。先占位、失败就 discard，更干净。
    """
    share_id = new_id()
    d = _dir(share_id)
    os.makedirs(d, exist_ok=True)
    return share_id, d


def save(source_name, result, extra=None, share_id=None):
    """
    把一次分析的结果落盘，返回 share_id。

    result 是 narrate.run_pipeline() 的返回值——报告、记录、统计都在里面。
    """
    share_id = share_id or new_id()
    d = _dir(share_id)
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
        f.write(result["markdown"])

    created = _now()
    meta = {
        "id": share_id,
        "source": source_name,
        "created_at": datetime.fromtimestamp(created).isoformat(timespec="seconds"),
        "expires_at": (datetime.fromtimestamp(created) + timedelta(hours=TTL_HOURS))
                      .isoformat(timespec="seconds"),
        "rows": result["profile"]["rows"],
        "cols": result["profile"]["cols"],
        "n_findings": len(result["records"]),
        "stats": result["stats"],
        "backend": result["backend"],
        "backend_label": result["backend_label"],
        "issues": result["issues"],
        "records": result["records"],
    }
    if extra:
        meta.update(extra)

    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return share_id


def load(share_id):
    """读取一份结果。过期/不存在/被篡改 id 都抛 Expired。"""
    d = _dir(share_id)
    meta_path = os.path.join(d, "meta.json")
    if not os.path.isfile(meta_path):
        raise Expired("不存在")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    try:
        expires = datetime.fromisoformat(meta["expires_at"]).timestamp()
    except (KeyError, ValueError):
        raise Expired("元信息损坏") from None
    if _now() > expires:
        discard(share_id)
        raise Expired("已过期")

    meta["report_path"] = os.path.join(d, "report.md")
    meta["chart_dir"] = os.path.join(d, "charts")
    meta["remaining_hours"] = round((expires - _now()) / 3600, 1)
    return meta


def report_text(share_id):
    meta = load(share_id)
    with open(meta["report_path"], encoding="utf-8") as f:
        return f.read()


def chart_path(share_id, filename):
    """
    取图表文件。filename 只允许 fig<数字>_<类型>.png 这一种形状，
    白名单式校验比"过滤 .."可靠——不给自己出错的余地。
    """
    meta = load(share_id)
    if not re.match(r"^fig\d+_[a-z_]+\.png$", filename or ""):
        raise Expired("非法文件名")
    p = os.path.join(meta["chart_dir"], filename)
    if not os.path.isfile(p):
        raise Expired("图表不存在")
    return p


def discard(share_id):
    try:
        shutil.rmtree(_dir(share_id), ignore_errors=True)
    except Expired:
        pass


def purge_expired():
    """
    清掉过期结果。每次有人上传前调用一次（惰性清理）。

    为什么不做定时任务：这个产品没有常驻的调度器，宿主也可能休眠。
    "有人来的时候顺手扫地"是最省事又不会积压的做法。
    """
    if not os.path.isdir(SHARE_DIR):
        return 0
    removed = 0
    now = _now()
    for name in os.listdir(SHARE_DIR):
        path = os.path.join(SHARE_DIR, name)
        if not os.path.isdir(path) or not ID_RE.match(name):
            continue
        meta_path = os.path.join(path, "meta.json")
        expired = False
        try:
            with open(meta_path, encoding="utf-8") as f:
                expires = datetime.fromisoformat(json.load(f)["expires_at"]).timestamp()
            expired = now > expires
        except (OSError, KeyError, ValueError):
            # 没有 meta 或读不出来 = 上一次分析中途崩了留下的残骸，同样该清
            expired = True
        if expired:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


def usage():
    """当前占用了多少空间/多少个结果，给健康检查用。"""
    if not os.path.isdir(SHARE_DIR):
        return {"shares": 0, "mb": 0.0}
    n, size = 0, 0
    for name in os.listdir(SHARE_DIR):
        path = os.path.join(SHARE_DIR, name)
        if os.path.isdir(path):
            n += 1
            for root, _dirs, files in os.walk(path):
                for fn in files:
                    try:
                        size += os.path.getsize(os.path.join(root, fn))
                    except OSError:
                        pass
    return {"shares": n, "mb": round(size / 1e6, 2)}
