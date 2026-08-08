"""生产基准 (泄漏修复后严格口径): 6因子 IC_IR 全历史回测 + 分段.

用 combined.py 的 factor_scores (IC_IR 修复后) 生成信号, 池内 ERC,
逐日回测, 统计全段/OOS/实盘/逐年.
"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from strategies.combined import CombinedStrategy


def main():
    s = CombinedStrategy()
    dm = s.runner.data_manager
    cal = pd.DatetimeIndex(dm.get_calendar(pd.Timestamp('2015-12-01'), pd.Timestamp('2026-08-07')))
    print(f'回测区间: {cal[0].date()} ~ {cal[-1].date()} ({len(cal)} 天)', flush=True)

    # 用 combined 的 IC_IR signal 逐日生成权重
    rets = []
    for i, t in enumerate(cal):
        if i == 0:
            continue
        try:
            w = s.signal(t.strftime('%Y-%m-%d'))
        except Exception:
            continue
        if w is None or len(w) == 0:
            continue
        # T 日收益 = T-1 信号持仓 × T 日价格变动
        prev_close = dm.get('close', pd.DatetimeIndex([cal[i-1]]), s._universe)
        cur_close = dm.get('close', pd.DatetimeIndex([t]), s._universe)
        if prev_close is None or cur_close is None or prev_close.empty or cur_close.empty:
            continue
        r = (cur_close.iloc[0] / prev_close.iloc[0] - 1).fillna(0.0)
        port_ret = sum(w.get(sym, 0.0) * r.get(sym, 0.0) for sym in w.index)
        rets.append((t, port_ret))
    s_ret = pd.Series({d: v for d, v in rets}).sort_index()

    def seg(name, sr):
        ann = sr.mean() * 252
        vol = sr.std() * np.sqrt(252)
        nav = (1 + sr).cumprod()
        mdd = (nav / nav.cummax() - 1).min()
        oos = sr[(sr.index >= pd.Timestamp('2026-03-01')) & (sr.index <= pd.Timestamp('2026-05-15'))]
        live = sr[sr.index > pd.Timestamp('2026-05-15')]
        osh = oos.mean() * 252 / (oos.std() * np.sqrt(252)) if len(oos) > 2 and oos.std() > 0 else 0
        lsh = live.mean() * 252 / (live.std() * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
        print(f'{name}: 夏普={ann/vol:.2f} 年化={ann:.1%} 回撤={mdd:.1%} OOS={osh:.2f} 实盘={lsh:.2f} ({len(sr)}天)')
        return ann / vol if vol > 0 else 0

    print('=== 生产 6因子 IC_IR (泄漏修复后) ===')
    seg('全段', s_ret)
    # 逐年
    for y in range(2016, 2027):
        yr = s_ret[s_ret.index.year == y]
        if len(yr) > 20:
            ann = yr.mean() * 252
            vol = yr.std() * np.sqrt(252)
            print(f'  {y}: 夏普={ann/vol if vol>0 else 0:.2f} 收益={yr.sum():.1%}')


if __name__ == '__main__':
    main()
