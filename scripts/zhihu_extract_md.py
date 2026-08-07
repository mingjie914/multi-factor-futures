# -*- coding: utf-8 -*-
"""知乎 pazzo-2 文章 HTML → 结构化 markdown 提取 (保留正文/公式).

遍历 E:\\程明杰公司内容\\参考资料\\知乎\\*.html:
- 提取标题/发布时间/点赞/原文链接/正文(去HTML标签, 保留公式/表格/代码)
- 每篇输出 知乎整理\\NNN_标题.md
- 生成 知乎整理\\_索引.md (标题清单 + 原文链接)
"""
from __future__ import annotations

import html
import os
import re
from datetime import datetime, timezone

SRC_DIR = r"E:\程明杰公司内容\参考资料\知乎"
OUT_DIR = os.path.join(SRC_DIR, "知乎整理")
os.makedirs(OUT_DIR, exist_ok=True)


def strip_tags(s: str) -> str:
    """HTML → 纯文本, 保留代码块与表格结构."""
    # 保护代码块
    code_blocks = []
    def save_code(m):
        code_blocks.append(m.group(1))
        return f"\n@@CODE{len(code_blocks)-1}@@\n"
    s = re.sub(r"<pre[^>]*>(.*?)</pre>", save_code, s, flags=re.S)
    # 表格单元格转 markdown 风格
    s = re.sub(r"</tr>", "\n", s)
    s = re.sub(r"</t[dh]>", " | ", s)
    # 段落/块
    s = re.sub(r"</(p|div|h[1-6]|li|br)>", "\n", s)
    # 链接保留文本
    s = re.sub(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', lambda m: f"{m.group(2)} ({m.group(1)})", s, flags=re.S)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    for i, code in enumerate(code_blocks):
        s = s.replace(f"@@CODE{i}@@", "\n```\n" + html.unescape(code) + "\n```\n")
    # 清理多余空行
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract(fpath: str) -> dict:
    with open(fpath, encoding="utf-8") as f:
        raw = f.read()
    title = re.search(r"<h1>(.*?)</h1>", raw, re.S)
    title = strip_tags(title.group(1)) if title else os.path.basename(fpath)
    meta = re.search(r'<div class="meta">(.*?)</div>', raw, re.S)
    meta_txt = strip_tags(meta.group(1)) if meta else ""
    body = re.search(r"<body>(.*?)</body>", raw, re.S)
    content = strip_tags(body.group(1)) if body else strip_tags(raw)
    return {"title": title, "meta": meta_txt, "content": content, "file": os.path.basename(fpath)}


def main() -> None:
    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".html"))
    print(f"共 {len(files)} 篇 HTML")
    index_lines = ["# pazzo-2 知乎文章索引\n", f"> 共 {len(files)} 篇, 由 HTML 自动提取\n"]
    n = 0
    for fname in files:
        try:
            art = extract(os.path.join(SRC_DIR, fname))
        except Exception as e:
            print(f"!! {fname}: {e}", flush=True)
            continue
        seq = fname.split("_")[0]
        out_name = f"{seq}_{art['title'][:60].replace(chr(92),'_').replace('/','_')}.md"
        out_path = os.path.join(OUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# {art['title']}\n\n")
            f.write(f"> 来源文件: {art['file']}\n")
            if art["meta"]:
                f.write(f"> {art['meta']}\n")
            f.write("\n---\n\n")
            f.write(art["content"])
        index_lines.append(f"- **{seq}** [{art['title']}]({out_name})")
        n += 1
    with open(os.path.join(OUT_DIR, "_索引.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))
    print(f"完成: {n} 篇 → {OUT_DIR}")


if __name__ == "__main__":
    main()
