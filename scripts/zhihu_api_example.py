# -*- coding: utf-8 -*-
"""知乎内容获取 — justoneapi SDK 用法示例 (data-api / justoneapi-python).

背景:
- justoneapi (PyPI 3.0.45) 是 "Just One API" 第三方聚合数据服务, 已安装于 E:\\Python\\Pythonvenv
- 知乎接口 15 个: 搜索 / 专栏文章列表 / 专栏文章详情 / 问题回答列表 / 用户信息与内容 / 评论
- 需要 token (https://justoneapi.com 申请, 付费/审核制), 填入下方 TOKEN 即可使用
- 知乎官方直连被反爬拦截 (web_fetch 403), GitHub 直连被网络阻断; PyPI 可达所以 SDK 可正常安装

对应 GitHub 仓库:
- justoneapi/justoneapi-python: 本 SDK 的源码 (openapi 生成)
- justoneapi/data-api: 后端服务 (api.justoneapi.com)

用法:
    python scripts/zhihu_api_example.py "你的token" "文章ID或关键词"
"""
from __future__ import annotations

import sys

from justoneapi import JustOneAPIClient


def fetch_article_by_id(client: JustOneAPIClient, article_id: str) -> None:
    """按专栏文章 ID 拉取正文 (ID 取自 URL: zhuanlan.zhihu.com/p/<ID>)."""
    print(f"=== 拉取专栏文章详情: {article_id} ===")
    resp = client.zhihu.get_column_article_detail_v1(id_=article_id)
    print("status:", resp.status_code)
    # 结构化正文通常在 data 内; 字段名以实际返回为准
    data = resp.data if hasattr(resp, "data") else resp
    print(data)


def search(client: JustOneAPIClient, keyword: str) -> None:
    """关键词搜索 (可限定 vertical=article 只搜文章)."""
    print(f"=== 搜索: {keyword} (文章) ===")
    resp = client.zhihu.search_v1(keyword=keyword, vertical="article", sort="created_time")
    print("status:", resp.status_code)
    data = resp.data if hasattr(resp, "data") else resp
    print(data)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        print("示例: python zhihu_api_example.py <token> 2068626750096126624")
        print("      python zhihu_api_example.py <token> --search 量化因子")
        return
    token = sys.argv[1]
    with JustOneAPIClient(token=token) as client:
        if len(sys.argv) >= 4 and sys.argv[2] == "--search":
            search(client, sys.argv[3])
        elif len(sys.argv) >= 3:
            fetch_article_by_id(client, sys.argv[2])
        else:
            print(__doc__)


if __name__ == "__main__":
    main()
