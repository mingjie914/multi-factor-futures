"""收集所有通过检验的有效因子 (组合 + 候选池 + validated), 生成待重检清单."""
import re

s = open('docs/有效因子库.md', encoding='utf-8').read()
head = s.split('## 第二层')[0]
tail = s.split('## 第二层')[1] if '## 第二层' in s else ''

combo = re.findall(r'`(intraday_\w+_20d)`', head)
cand = re.findall(r'\|\s*`(intraday_\w+_20d)`', tail)

all_f = sorted(set(combo + cand))
print(f'组合因子: {sorted(set(combo))}')
print(f'候选池因子 ({len(cand)}): {sorted(set(cand))}')
print(f'合计去重: {len(all_f)}')
print('---')
print(','.join(all_f))
