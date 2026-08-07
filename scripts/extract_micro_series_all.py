# -*- coding: utf-8 -*-
"""批量提取目录下所有研报 PDF 文本(排除已提取的 31-33 与笔记)."""
import fitz
import os
import re

SRC = r"E:\程明杰公司内容\参考资料\看研报\开源证券_市场微观结构研究系列"
OUT = r"E:\程明杰公司内容\研报整理\02_Markdown源\开源微观结构_批量提取"
os.makedirs(OUT, exist_ok=True)

skip = {"研报笔记_开源证券市场微观结构系列31-33.md"}
for fn in sorted(os.listdir(SRC)):
    if not fn.lower().endswith(".pdf"):
        continue
    path = os.path.join(SRC, fn)
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"打开失败: {fn}: {e}")
        continue
    pages = len(doc)
    total_chars = 0
    parts = []
    for i, page in enumerate(doc, 1):
        t = page.get_text("text")
        total_chars += len(t)
        parts.append(f"\n===== [第 {i} 页 / 共 {pages} 页] =====\n{t}")
    doc.close()
    # 输出文件名: 期号_原文件名去掉.pdf
    m = re.search(r"系列\((\d+)\)", fn)
    no = m.group(1) if m else "xx"
    out = os.path.join(OUT, f"{no}_{fn[:-4]}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"#{no} {fn[:40]}... | {pages}页 | {total_chars}字符 | {'文本OK' if total_chars>500 else '!!!文本过少(可能扫描图)'}")
