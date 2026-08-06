# -*- coding: utf-8 -*-
"""勘察: 当前因子库编号与 term 面板调用者"""
import re
p = r"E:\程明杰公司内容\multi_factor\factors\library\intraday.py"
src = open(p, encoding="utf-8").read()
factors = re.findall(r"# (\d+)\.\s*([a-z0-9_]+)", src)
regs = re.findall(r'@register_factor\("([a-z0-9_]+)"', src)
print("编号注释数:", len(factors), "| register_factor 数:", len(regs))
nums = [int(f[0]) for f in factors]
print("编号范围:", min(nums), "-", max(nums), "| 唯一:", len(set(nums)) == len(nums),
      "| 连续:", sorted(nums) == list(range(min(nums), max(nums) + 1)))
print("\n#380 之后因子:")
for f in factors:
    if int(f[0]) >= 380:
        print(f"  {f[0]}. {f[1]}")
# _get_term_structure_panel 调用者
print("\n_get_term_structure_panel 调用者:")
for m in re.finditer(r"class\s+(\w+).*?panel = _get_term_structure_panel", src, re.S):
    print("  ", m.group(1))
print("\n调用 _read_local_term 的行:")
for m in re.finditer(r"_read_local_term|_get_term_structure_panel", src):
    line = src[:m.start()].count("\n") + 1
    print(f"  L{line}")
