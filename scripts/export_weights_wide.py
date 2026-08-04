"""生成外部程序可直接回测的 wide 权重文件.

格式: date | instrument1 | instrument2 | ... (每行一个执行日, 单元格=带符号权重, 空头为负)
日期 = 执行日 (T 日持有), 不 shift, 外部用 weight_T × return_T.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd

w = pd.read_csv('weights/daily_weights.csv', parse_dates=['date'])
# 转 wide: date × instrument
wide = w.pivot(index='date', columns='symbol', values='weight').fillna(0.0).sort_index()
# 重命名列
wide.columns.name = 'instrument'
wide = wide.reset_index()
out = 'weights/daily_weights_wide.csv'
wide.to_csv(out, index=False, encoding='utf-8')
print(f'已导出 wide 格式: {out}')
print(f'形状: {wide.shape[0]} 行(交易日) × {wide.shape[1]-1} 列(品种)')
print(f'日期范围: {wide.date.min().date()} ~ {wide.date.max().date()}')
print()
print('前3行样例:')
print(wide.head(3).to_string(index=False, max_cols=8))
print()
# 验证: 每行 20 个非零, 多头+1 空头-1
nz = (wide.iloc[:, 1:] != 0).sum(axis=1)
print(f'每行非零品种: min={nz.min()} max={nz.max()} (应为20)')
long_sum = wide.iloc[:, 1:].clip(lower=0).sum(axis=1)
short_sum = wide.iloc[:, 1:].clip(upper=0).sum(axis=1)
print(f'多头合计: {long_sum.mean():.4f}±{long_sum.std():.4f} (应≈1.0)')
print(f'空头合计: {short_sum.mean():.4f}±{short_sum.std():.4f} (应≈-1.0)')
