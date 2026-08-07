# -*- coding: utf-8 -*-
"""下载知乎用户 pazzo-2 的全部文章到 E:\\程明杰公司内容\\参考资料\\知乎.

流程:
1. 分页拉取用户文章列表 (get_user_articles_v1, offset 步进 20) 收集元信息
2. 逐篇调用详情接口 (get_column_article_detail_v1) 拉完整 HTML 正文
3. 保存为 .html (含标题/时间/点赞/原文链接/正文), 文件名 = <id>_<序号>.html

特性:
- 限速 1.2s/请求, 防止触发 API 风控
- 断点续传: 已存在的文件跳过
- 进度日志输出到 stdout
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from justoneapi import JustOneAPIClient

TOKEN = "vnP3yDQPrS5FBmQr"
USER_URL_TOKEN = "pazzo-2"
OUT_DIR = r"E:\程明杰公司内容\参考资料\知乎"
PAGE_SIZE = 20
SLEEP = 1.2

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ max-width: 900px; margin: 2em auto; padding: 0 1em; font-family: -apple-system, "Microsoft YaHei", sans-serif; line-height: 1.7; }}
h1 {{ font-size: 1.6em; }}
.meta {{ color: #666; font-size: 0.9em; margin-bottom: 1.5em; }}
.meta a {{ color: #175199; }}
img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">
<p>发布时间: {created} &nbsp;|&nbsp; 赞同: {voteup} &nbsp;|&nbsp; 评论: {comment}</p>
<p>原文: <a href="{url}">{url}</a></p>
</div>
<hr>
{content}
</body>
</html>
"""


def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\r\n\t]', "_", name)[:120]


def fetch_article_ids(client: JustOneAPIClient) -> list[dict]:
    """分页拉取全部文章元信息."""
    articles = []
    offset = 0
    while True:
        resp = client.zhihu.get_user_articles_v1(
            user_url_token=USER_URL_TOKEN, offset=offset, sort_type="created")
        if not resp.success or not resp.data:
            print(f"!! 列表拉取失败 offset={offset}: {resp.message}", flush=True)
            break
        data = resp.data
        batch = data.get("data", [])
        articles.extend(batch)
        totals = data.get("paging", {}).get("totals", 0)
        print(f"列表 offset={offset} +{len(batch)} (共 {len(articles)}/{totals})", flush=True)
        if data.get("paging", {}).get("is_end") or not batch:
            break
        offset += len(batch)
        time.sleep(SLEEP)
    return articles


def fetch_detail(client: JustOneAPIClient, article_id: str) -> dict | None:
    """拉取单篇完整正文."""
    resp = client.zhihu.get_column_article_detail_v1(id_=article_id)
    if resp.success and resp.data:
        return resp.data
    print(f"!! 详情拉取失败 id={article_id}: {resp.message}", flush=True)
    return None


def save_article(meta: dict, content_html: str, index: int) -> str:
    """保存单篇文章为 HTML, 返回文件路径."""
    os.makedirs(OUT_DIR, exist_ok=True)
    aid = str(meta.get("id", ""))
    title = sanitize(meta.get("title") or "untitled")
    created = meta.get("created")
    if created:
        created_str = datetime.fromtimestamp(created, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    else:
        created_str = "unknown"
    url = meta.get("url") or f"https://zhuanlan.zhihu.com/p/{aid}"
    html = HTML_TEMPLATE.format(
        title=title,
        created=created_str,
        voteup=meta.get("voteup_count", "?"),
        comment=meta.get("comment_count", "?"),
        url=url,
        content=content_html,
    )
    fname = f"{index:03d}_{aid}_{title}.html"
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)
    return fpath


def main() -> None:
    print("开始拉取用户文章列表...", flush=True)
    with JustOneAPIClient(token=TOKEN) as client:
        articles = fetch_article_ids(client)
        if not articles:
            print("无文章, 退出", flush=True)
            return
        print(f"共 {len(articles)} 篇, 开始逐篇下载...", flush=True)
        ok = skipped = fail = 0
        for i, meta in enumerate(articles, 1):
            aid = str(meta.get("id", ""))
            title = sanitize(meta.get("title") or "untitled")
            # 断点续传: 检查是否已存在同名文件
            existing = [f for f in os.listdir(OUT_DIR) if f.startswith(f"{i:03d}_") and aid in f]
            if existing:
                print(f"[{i}/{len(articles)}] 跳过(已存在): {title[:40]}", flush=True)
                skipped += 1
                continue
            detail = fetch_detail(client, aid)
            if detail is None:
                fail += 1
                continue
            content = detail.get("content", "")
            # 用列表里的 voteup/comment 补全 meta (详情接口不一定有)
            meta.setdefault("voteup_count", detail.get("voteup_count"))
            meta.setdefault("comment_count", detail.get("comment_count"))
            meta.setdefault("created", detail.get("created"))
            meta.setdefault("url", detail.get("url") or meta.get("url"))
            fpath = save_article(meta, content, i)
            ok += 1
            print(f"[{i}/{len(articles)}] OK: {title[:40]} -> {os.path.basename(fpath)}", flush=True)
            time.sleep(SLEEP)
        print(f"\n完成: 新增 {ok}, 跳过 {skipped}, 失败 {fail}, 共 {len(articles)}", flush=True)


if __name__ == "__main__":
    main()
