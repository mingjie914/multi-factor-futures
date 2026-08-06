# -*- coding: utf-8 -*-
"""从 RDS Wind EOD (ccommodityfutureseodprices) 补全本地 1d parquet 缺失数据.

背景: 本地 futureshistoryprices1d 与 RDS futureshistoryprices1d 表一致,
但该表对低流动性品种 (如 FU 2016) 记录稀疏; Wind EOD 表 (ccommodity
futureseodprices) 有完整逐日逐合约数据. 本脚本用 Wind EOD 补本地缺失的
(symbol, trade_date).

格式映射 (踩坑核对):
- symbol:  Wind 'FU1608.SHF' -> 大写 'FU1608' (本地已全大写)
- trade_datetime: trade_date 当天 00:00 (与本地一致)
- amount:   Wind EOD 是万元 -> 本地元 (x10000)
- settle:   Wind EOD S_DQ_SETTLE -> settle_price
- position: Wind EOD S_DQ_OI -> position
- pre_settle_price: Wind EOD 无, 用前一日 settle 推导 (fillna)
- type/sequence: 本地样例 type=14, sequence=1; 用固定值
- exchange: 从 Wind 代码后缀推导 (SHF/CZC/DCE/CFE)

写入: 追加到对应 year_month 分区的 data_N.parquet (保持多分片约定).
备份: 写入前对目标分区做备份.
"""
import os
import glob
import shutil
import datetime
import argparse
import yaml
import pymysql
import pandas as pd
import numpy as np

LOCAL_BASE = r"E:\程明杰公司内容\期货行情数据\本地表\futureshistoryprices1d"
CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'config', 'local.yaml'), encoding='utf-8'))
MYSQL = CFG['data']['mysql']

EXCH_SUFFIX = {'.SHF': 'SHFE', '.DCE': 'DCE', '.CZC': 'CZCE', '.CFE': 'CFFEX'}


def _f(v):
    """None -> NaN (Wind EOD 有缺失值)."""
    return float('nan') if v is None else float(v)


def wind_to_local(row):
    """Wind EOD 行 -> 本地 1d 行 (格式映射)."""
    code = row['S_INFO_WINDCODE']
    suffix = next((s for s in EXCH_SUFFIX if code.endswith(s)), '.SHF')
    exchange = EXCH_SUFFIX[suffix]
    symbol = code[:-len(suffix)].upper()
    td = str(row['TRADE_DT'])
    trade_datetime = pd.Timestamp(td)
    return {
        'exchange': exchange,
        'symbol': symbol,
        'trade_datetime': trade_datetime,
        'open': _f(row['S_DQ_OPEN']),
        'high': _f(row['S_DQ_HIGH']),
        'low': _f(row['S_DQ_LOW']),
        'close': _f(row['S_DQ_CLOSE']),
        'volume': _f(row['S_DQ_VOLUME']),
        'amount': _f(row['S_DQ_AMOUNT']) * 10000.0,  # 万元 -> 元
        'position': _f(row['S_DQ_OI']),
        'type': 14,
        'sequence': 1,
        'trade_date': datetime.date(int(td[:4]), int(td[4:6]), int(td[6:8])),  # date 对象 (与本地一致)
        'settle_price': _f(row['S_DQ_SETTLE']),
        'pre_settle_price': np.nan,  # 后处理: 用前一日 settle
    }


def local_existing_keys():
    """本地 1d 全部 (symbol, trade_date) 集合."""
    keys = set()
    for d in sorted(os.listdir(LOCAL_BASE)):
        if not d.startswith('year_month='):
            continue
        pdir = os.path.join(LOCAL_BASE, d)
        for f in glob.glob(os.path.join(pdir, '*.parquet')):
            try:
                df = pd.read_parquet(f, columns=['symbol', 'trade_date'])
                syms = df['symbol'].astype(str).str.upper()
                keys.update(zip(syms, df['trade_date'].astype(str)))
            except Exception:
                continue
    return keys


def fetch_wind_missing(roots, y0, y1, existing):
    """从 Wind EOD 拉指定品种+年份, 返回本地缺失的行."""
    conn = pymysql.connect(host=MYSQL['host'], port=MYSQL['port'], user=MYSQL['user'],
                           password=MYSQL['password'], database=MYSQL['database'],
                           charset='utf8mb4', connect_timeout=15)
    cur = conn.cursor()
    rows_all = []
    for root in roots:
        cur.execute("""
            SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW,
                   S_DQ_CLOSE, S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_OI, S_DQ_SETTLE
            FROM ccommodityfutureseodprices
            WHERE TRADE_DT >= %s AND TRADE_DT <= %s
              AND UPPER(S_INFO_WINDCODE) LIKE %s
              AND S_INFO_WINDCODE REGEXP '^[A-Z]+[0-9]{4}\\.'
        """, (f'{y0}0101', f'{y1}1231', root + '%'))
        rows_all.extend(cur.fetchall())
    conn.close()
    cols = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW',
            'S_DQ_CLOSE', 'S_DQ_VOLUME', 'S_DQ_AMOUNT', 'S_DQ_OI', 'S_DQ_SETTLE']
    wind = pd.DataFrame(rows_all, columns=cols)
    if wind.empty:
        return pd.DataFrame()
    mapped = wind.apply(wind_to_local, axis=1, result_type='expand')
    mapped['_key'] = list(zip(mapped['symbol'], mapped['trade_date'].astype(str)))
    # 只保留本地缺失的
    missing = mapped[~mapped['_key'].isin(existing)].drop(columns=['_key'])
    return missing


def fill_pre_settle(df):
    """用该品种前一日 settle 填充 pre_settle_price (需先读本地历史 settle)."""
    # 读本地全部历史 settle (symbol, trade_date -> settle_price)
    hist = {}
    for d in sorted(os.listdir(LOCAL_BASE)):
        if not d.startswith('year_month='):
            continue
        pdir = os.path.join(LOCAL_BASE, d)
        for f in glob.glob(os.path.join(pdir, '*.parquet')):
            try:
                h = pd.read_parquet(f, columns=['symbol', 'trade_date', 'settle_price'])
                for r in h.itertuples(index=False):
                    hist[(str(r.symbol).upper(), str(r.trade_date))] = r.settle_price
            except Exception:
                continue
    # 本次新增的 settle 也加入 (同一品种前一日可能在本次数据中)
    for r in df.itertuples(index=False):
        hist[(r.symbol, str(r.trade_date))] = r.settle_price
    out = df.copy()
    prev = []
    for r in out.itertuples(index=False):
        # 找前一交易日: 用本地日历 (简化: 前一个存在的 (symbol, date))
        d = pd.Timestamp(r.trade_date)
        best = None
        for days in range(1, 15):
            prev_d = (d - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
            if (r.symbol, prev_d) in hist:
                best = hist[(r.symbol, prev_d)]
                break
        prev.append(best if best is not None else np.nan)
    out['pre_settle_price'] = prev
    return out


def write_partitions(df, dry_run=True):
    """按 year_month 分区写入本地 (追加到 data_N.parquet)."""
    df = df.copy()
    df['_ym'] = df['trade_datetime'].dt.strftime('%Y-%m')
    for ym, grp in df.groupby('_ym'):
        pdir = os.path.join(LOCAL_BASE, f'year_month={ym}')
        if not os.path.isdir(pdir):
            print(f'  ⚠️ 分区不存在: {pdir} (跳过)')
            continue
        files = sorted(glob.glob(os.path.join(pdir, 'data_*.parquet')))
        if not files:
            files = sorted(glob.glob(os.path.join(pdir, 'part.parquet')))
        if not files:
            print(f'  ⚠️ 分区无 parquet: {pdir} (跳过)')
            continue
        target = files[-1]  # 追加到最后一个分片
        out = grp.drop(columns=['_ym'])
        # 保持列顺序与本地一致
        col_order = ['exchange', 'symbol', 'trade_datetime', 'open', 'high', 'low',
                     'close', 'volume', 'amount', 'position', 'type', 'sequence',
                     'trade_date', 'settle_price', 'pre_settle_price']
        out = out[col_order]
        # pre_settle_price: 用该品种前一日 settle 填充 (跨分区需先读本地历史)
        if not dry_run:
            out = fill_pre_settle(out)
        if dry_run:
            print(f'  [dry-run] {ym}: 追加 {len(out)} 行 -> {target}')
        else:
            # 备份
            bak = target + f'.bak_{datetime.date.today().strftime("%Y%m%d")}'
            if not os.path.exists(bak):
                shutil.copy2(target, bak)
            existing = pd.read_parquet(target)
            merged = pd.concat([existing, out], ignore_index=True)
            merged.to_parquet(target, index=False)
            print(f'  [write] {ym}: {len(existing)} + {len(out)} = {len(merged)} 行 -> {target}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', default=['FU'], help='品种列表')
    parser.add_argument('--y0', type=int, default=2015)
    parser.add_argument('--y1', type=int, default=2018)
    parser.add_argument('--apply', action='store_true', help='实际写入 (默认 dry-run)')
    args = parser.parse_args()

    print(f'=== 补数: {args.roots} {args.y0}-{args.y1} ===')
    print('扫描本地已有 (symbol, date)...')
    existing = local_existing_keys()
    print(f'  本地唯一 (symbol, date): {len(existing)}')

    print('拉取 Wind EOD + 找缺失...')
    missing = fetch_wind_missing(args.roots, args.y0, args.y1, existing)
    print(f'  缺失 (本地无, Wind 有): {len(missing)} 行')
    if missing.empty:
        print('无缺失, 无需补数')
        return
    print('  按品种:')
    print(missing.groupby('symbol').size().to_string())
    print(f'\n写入 ({"apply" if args.apply else "dry-run"}):')
    write_partitions(missing, dry_run=not args.apply)


if __name__ == '__main__':
    main()
