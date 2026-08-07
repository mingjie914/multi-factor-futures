# -*- coding: utf-8 -*-
"""提取三篇开源证券市场微观结构系列 PDF 全文(带页码)."""
import fitz
import os

SRC = r"E:\程明杰公司内容\参考资料\看研报\开源证券_市场微观结构研究系列"
OUT = r"E:\程明杰公司内容\研报整理\02_Markdown源"

files = {
    "31": "开源证券_分钟资金流因子的构建方法——市场微观结构系列（31）_63775916.pdf",
    "32": "开源证券_深度学习赋能因子挖掘2.0：综合应用方案——市场微观结构系列（32）_63876280.pdf",
    "33": "开源证券_高频价格跳跃的峰、岭、谷信息——市场微观结构研究系列（33）_64265610.pdf",
}
for no, fn in files.items():
    path = os.path.join(SRC, fn)
    doc = fitz.open(path)
    parts = []
    for i, page in enumerate(doc, 1):
        txt = page.get_text("text")
        parts.append(f"\n===== [第 {i} 页 / 共 {len(doc)} 页] =====\n")
        parts.append(txt)
    full = "".join(parts)
    out = os.path.join(OUT, f"开源_微观结构系列{no}_提取.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"{no}: {len(doc)} 页, 文本 {len(full)} 字符 → {os.path.basename(out)}")
    doc.close()
