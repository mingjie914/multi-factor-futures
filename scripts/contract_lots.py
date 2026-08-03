"""contract_lots — 权重 → 手数换算工具 (备用).

用法:
    python scripts/contract_lots.py --date 2026-08-03 --capital 1000000
    python scripts/contract_lots.py --weights "TL:0.246,A:0.1891" --capital 500000

输出: 每品种 权重 / 名义价值 / 最新价 / 乘数 / 手数(取整) / 实际名义 / 偏差.
数据源: data/contract_specs.py (RDS 导出) + 本地日度收盘价 (最新交易日).
"""
import sys
import json
import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data.contract_specs import CONTRACT_SPECS


def _latest_close() -> dict:
    """读本地日度 parquet 最新交易日收盘价 {root: close}."""
    import os
    import re
    import glob
    base = r'E:\程明杰公司内容\期货行情数据\本地表\futureshistoryprices1d'
    months = sorted(os.listdir(base))
    files = glob.glob(os.path.join(base, months[-1], '*.parquet'))
    if not files:
        return {}
    df = pd.read_parquet(files[0])
    dt_col = 'trade_datetime' if 'trade_datetime' in df.columns else df.columns[0]
    latest = df[dt_col].max()
    df = df[df[dt_col] == latest]
    closes = {}
    for sym, grp in df.groupby('symbol'):
        m = re.match(r'^([A-Za-z]+)', str(sym))
        if not m:
            continue
        root = m.group(1).upper()
        c = grp['close'].dropna()
        if not c.empty:
            closes[root] = float(c.iloc[0])
    return closes


def _weights_from_signal(date: str) -> dict:
    from strategies.combined import CombinedStrategy
    strat = CombinedStrategy('config/intraday_backtest.yaml')
    w = strat.signal(date)
    return {k: v for k, v in w.items() if abs(v) > 1e-12}


def lots_for(weights: dict, capital: float, closes: dict):
    rows = []
    for sym, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
        spec = CONTRACT_SPECS.get(sym)
        price = closes.get(sym)
        if not spec or not price:
            rows.append({'symbol': sym, 'weight': w, 'note': '无规格/价格'})
            continue
        mult = spec['multiplier']
        notional = w * capital
        raw_lots = abs(notional) / (price * mult)
        lots = int(raw_lots)  # 向下取整 (保守)
        actual_notional = lots * price * mult * (1 if w > 0 else -1)
        margin = spec.get('margin')
        rows.append({
            'symbol': sym, 'direction': '多' if w > 0 else '空',
            'weight': w, 'notional': notional, 'price': price,
            'multiplier': mult, 'unit': spec['unit'],
            'raw_lots': raw_lots, 'lots': lots,
            'actual_notional': actual_notional,
            'margin': margin,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description='权重→手数换算')
    ap.add_argument('--date', help='信号日期 (用 combined 信号)')
    ap.add_argument('--weights', help='手动权重 "TL:0.246,A:0.1891"')
    ap.add_argument('--capital', type=float, default=1_000_000, help='总资金 (元)')
    args = ap.parse_args()

    closes = _latest_close()
    print(f'最新收盘日: 数据最新月份 {sorted(__import__("os").listdir(r"E:\\程明杰公司内容\\期货行情数据\\本地表\\futureshistoryprices1d"))[-1]}')
    if args.weights:
        weights = {k.strip(): float(v) for k, v in
                   (pair.split(':') for pair in args.weights.split(','))}
    elif args.date:
        weights = _weights_from_signal(args.date)
    else:
        print('需 --date 或 --weights')
        sys.exit(1)

    rows = lots_for(weights, args.capital, closes)
    print(f'\n资金 {args.capital:,.0f} 元 | 总杠杆 {sum(abs(w) for w in weights.values()):.2f}\n')
    print(f'{"品种":<4} {"向":<2} {"权重":>7} {"名义(元)":>12} {"价格":>10} {"乘数":>7} '
          f'{"单位":<6} {"理论手":>7} {"实手":>4} {"保证金率":>6} {"保证金(元)":>11} {"偏差%":>7}')
    print('-' * 120)
    total_notional = 0
    total_margin = 0
    for r in rows:
        if 'note' in r:
            print(f"{r['symbol']:<4}  {r['note']}")
            continue
        dev = (r['actual_notional'] - r['notional']) / r['notional'] * 100 if r['notional'] else 0
        total_notional += abs(r['actual_notional'])
        margin_amt = abs(r['actual_notional']) * r['margin'] if r['margin'] and r['lots'] > 0 else 0
        total_margin += margin_amt
        print(f"{r['symbol']:<4} {r['direction']:<2} {r['weight']:>7.4f} {r['notional']:>12,.0f} "
              f"{r['price']:>10.2f} {r['multiplier']:>7,.0f} {r['unit']:<6} "
              f"{r['raw_lots']:>7.2f} {r['lots']:>4} {r['margin'] or 0:>5.1%} "
              f"{margin_amt:>11,.0f} {dev:>6.1f}%")
    print(f"\n实际总名义(多空绝对值): {total_notional:,.0f} 元 (目标 {args.capital * sum(abs(w) for w in weights.values()):,.0f})")
    print(f"实际保证金占用: {total_margin:,.0f} 元 (占总资金 {total_margin/args.capital:.1%})")


if __name__ == '__main__':
    main()
