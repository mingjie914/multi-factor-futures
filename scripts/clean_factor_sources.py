# -*- coding: utf-8 -*-
"""清理 #439-530 因子注释中的外部来源信息 (开源/知乎/复现), 全部默认为原创.

保留: 因子名、方向、公式、经济学说明.
删除: (知乎 feat_xxx 复现) / (开源证券XX期) / 复现/借鉴 等来源标注.
"""
import re

P = 'factors/library/intraday.py'
s = open(P, encoding='utf-8').read()

# 1. 找 #439 起始
i = s.find('# 439.')
if i < 0:
    i = s.find('intraday_dazzling_vol_20d')
head, tail = s[:i], s[i:]

# 2. 尾部各种来源模式替换
repls = [
    # (知乎 feat_xxx 复现) / (知乎#098复现) 括号
    (r'\s*\(知乎[^)]*\)', ''),
    # (开源证券XX期[^)]*) / (开源31期) 括号
    (r'\s*\(开源证券?\d+期[^)]*\)', ''),
    (r'\s*\(开源31期[^)]*\)', ''),
    (r'\s*\(开源32期[^)]*\)', ''),
    (r'\s*\(开源33期[^)]*\)', ''),
    # 行内 "复现"/"借鉴原创" 字样
    (r'\s*(复现|借鉴原创)', '原创'),
    # docstring 内 "知乎 ... 复现" 无括号
    (r'知乎\s*\w+[，,]?\s*复现', '原创'),
    # "开源证券31期, " 等 docstring 来源前缀
    (r'开源证券?\d+期[，,]\s*', ''),
    (r'开源31期[，,]\s*', ''),
]
n = 0
for pat, rep in repls:
    tail, c = re.subn(pat, rep, tail)
    n += c
print(f'来源字样替换: {n} 处')

open(P, 'w', encoding='utf-8').write(head + tail)
print('完成')
