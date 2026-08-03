"""diag_pool_sector_icir — 只读对照实验: 品种池 / 板块中性 / ICIR 加权 vs 基准.

对照组 (全部基于 B3 6因子 + ERC 池内权重 + 日度调仓 D, 2025-01~2026-05):
  A0 基准     manual29 (含金融6: IC/IF/IH/T/TL/TS) 全市场截面排名
  A1 商品23   去金融 (23商品)                       全市场截面排名
  B1 板块中性 manual29                              板块内 z-score 后全市场排名
  C1 ICIR加权 manual29                              滚动60日 ICIR 加权合成 (替代等权)
  D1 商品+板块 商品23 + 板块中性 + ICIR加权 (组合最优主张)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from core.config import load_config
from pipeline.runner import PipelineRunner
from factors.engine import FactorEngine
from optimization.risk_budgeting import RiskBudgetingOptimizer

B3 = {
    "intraday_jump_intensity_20d": -1,
    "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1,
    "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1,
    "intraday_peak_ridge_ratio_20d": -1,
}
FIN = ["IC", "IF", "IH", "T", "TL", "TS"]
SECTORS = {  # manual29 商品板块映射 (23 商品)
    "有色": ["CU", "AL", "ZN", "NI", "SN", "AG", "AU"],
    "黑色": ["RB", "HC", "I", "J", "JM"],
    "能化": ["FU", "MA", "RU", "SA", "TA"],
    "农产品": ["A", "M", "P", "RM", "Y", "SR"],
}


def build_runner():
    cfg = load_config("config/intraday_backtest.yaml")
    runner = PipelineRunner(config=cfg)
    return runner, cfg
def backtest(runner, factors, universe, start, end,
             sector_neutral=False, icir_window=0, sector_cap=0):
    """周度/日度可调? 固定日度 D. sector_cap>0: 每板块多/空配额上限."""
    cal = pd.DatetimeIndex(runner.data_manager.get_calendar(
        pd.Timestamp(start), pd.Timestamp(end)))
    engine = FactorEngine(runner.data_manager)
    names = list(factors)
    computed = engine.compute_factors(names, cal, universe, parallel=True)
    close = runner.data_manager.get("close", cal, universe)
    daily_ret = close.pct_change()
    fwd = daily_ret.shift(-1)

    sector_of = {}
    for sec, mem in SECTORS.items():
        for m in mem:
            if m in universe:
                sector_of[m] = sec
    for m in FIN:
        if m in universe:
            sector_of[m] = "金融"

    # 因子截面 rank (板块内 or 全市场) → 打分
    scores = {}
    for name, direction in factors.items():
        raw = computed[name]
        if sector_neutral:
            z = pd.DataFrame(index=cal, columns=universe, dtype=float)
            for sec, mem in SECTORS.items():
                mem = [m for m in mem if m in universe]
                if not mem:
                    continue
                sub = raw[mem]
                m = sub.mean(axis=1)
                s = sub.std(axis=1).replace(0, np.nan)
                z[mem] = (sub.sub(m, axis=0)).div(s, axis=0)
            r = z.rank(axis=1, pct=True)
        else:
            r = raw.rank(axis=1, pct=True)
        scores[name] = r if direction == 1 else (1 - r)

    if icir_window > 0:
        # 滚动 ICIR 加权: 每个因子每日截面 RankIC vs 次日收益, 过去 icir_window 日均值/std
        ic = {}
        for name in names:
            r = scores[name]
            # 日频 IC: 截面 spearman (用 rank 线性等价)
            fwd_rank = fwd.rank(axis=1)
            ic[name] = r.corrwith(fwd_rank, axis=1)
        ic_df = pd.DataFrame(ic)
        icir = ic_df.rolling(icir_window, min_periods=max(20, icir_window // 3)).mean() / \
               ic_df.rolling(icir_window, min_periods=max(20, icir_window // 3)).std().replace(0, np.nan)
        icir = icir.abs()
        # 用 t-1 的 ICIR 加权 (防前视)
        icir = icir.shift(1).fillna(1.0 / len(names))
        wsum = icir.sum(axis=1).replace(0, np.nan)
        w = icir.div(wsum, axis=0)
        score = pd.DataFrame(0.0, index=cal, columns=universe)
        for name in names:
            score = score.add(scores[name].mul(w[name], axis=0), fill_value=0)
        # 综合分归一化到 [0,1]
        score = score.div(score.sum(axis=1).replace(0, np.nan), axis=0)
    else:
        score = pd.DataFrame(index=cal, columns=universe, dtype=float)
        for name in names:
            score = score.add(scores[name], fill_value=0)
        score = score.div(len(names))

    vol20 = close.pct_change().rolling(20, min_periods=10).std(ddof=0)
    rebal = score.resample("D").last()
    rets = []
    prev_l, prev_s = None, None

    def capped_picks(order_desc, cap):
        """按得分降序取, 但每板块最多 cap 个, 直到取满 10 个."""
        picks, counts = [], {}
        for s in order_desc:
            sec = sector_of.get(s, "其他")
            if counts.get(sec, 0) >= cap:
                continue
            picks.append(s)
            counts[sec] = counts.get(sec, 0) + 1
            if len(picks) >= 10:
                break
        return picks

    for t in rebal.index:
        row = rebal.loc[t].dropna()
        if len(row) < 20:
            continue
        rk = row.rank(ascending=False)
        order_desc = list(row.sort_values(ascending=False).index)
        order_asc = list(row.sort_values(ascending=True).index)
        if sector_cap > 0:
            top = capped_picks(order_desc, sector_cap)
            bot = capped_picks(order_asc, sector_cap)
        else:
            top = rk[rk <= 10].index
            bot = rk[rk > len(rk) - 10].index
        vt = vol20.loc[t] if t in vol20.index else vol20.asof(t)

        def erc_w(pool):
            if len(pool) < 2:
                return None
            start_d = t - pd.Timedelta(days=90)
            c = pd.DatetimeIndex(runner.data_manager.get_calendar(start_d, t))
            ret_sub = daily_ret.reindex(c)[list(pool)].dropna()
            if ret_sub.shape[0] < 10:
                return None
            cov_raw = ret_sub.cov().values
            target = np.diag(np.diag(cov_raw))
            cov = 0.7 * cov_raw + 0.3 * target
            try:
                w = RiskBudgetingOptimizer._erc_weights(cov, np.ones(len(pool)))
                return dict(zip(pool, w))
            except (RuntimeError, ValueError):
                v = ret_sub.std(ddof=0).replace(0, np.nan).dropna()
                if v.empty:
                    return None
                w = (1.0 / v).values
                return dict(zip(pool, w / w.sum()))
        wl = erc_w(top)
        ws = erc_w(bot)
        nxt = cal[(cal > t) & (cal <= t + pd.Timedelta(days=1))]
        for d in nxt:
            if d in fwd.index:
                r = fwd.loc[d]
                lr = sum(r[c] * wi for c, wi in wl.items()) if wl else 0
                sr = sum(r[c] * wi for c, wi in ws.items()) if ws else 0
                rets.append((d, lr - sr))
    return pd.Series({d: v for d, v in rets}).sort_index().dropna()


def stat(s):
    if len(s) < 3:
        return "n/a"
    nav = (1 + s).cumprod()
    ann = s.mean() * 252
    vol = s.std(ddof=0) * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    # 月换手: 每周池变化平均
    return f"年化={ann:.1%} 夏普={ann/vol if vol>0 else 0:.2f} 回撤={mdd:.1%} 波动={vol:.1%} n={len(s)}"


def main():
    runner, cfg = build_runner()
    base = dict(B3)
    manual29 = ["A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM", "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL", "TS", "Y", "ZN"]
    comm23 = [s for s in manual29 if s not in FIN]
    # 4 方案因子集
    V8 = dict(B3); V8["intraday_seat_count_rank_20d"] = -1; V8["intraday_settle_close_basis_20d"] = -1
    V10 = dict(V8); V10["intraday_seat_long_short_ratio_20d"] = 1; V10["intraday_price_rank_vol_20d"] = 1
    V12 = dict(V10); V12["intraday_volume_rank_ratio_20d"] = 1; V12["intraday_settle_basis_rank_20d"] = 1
    factor_sets = {
        "B3 6因子": B3,
        "8因子": V8,
        "10因子(6簇)": V10,
        "12因子(6簇扩)": V12,
    }
    # 板块/ICIR 对照 (B3)
    cases = {
        "A0 基准(manual29)": dict(universe=manual29, sector=False, icir=0, cap=0),
        "A1 商品23(去金融)": dict(universe=comm23, sector=False, icir=0, cap=0),
        "B1 板块中性": dict(universe=manual29, sector=True, icir=0, cap=0),
        "C1 ICIR加权": dict(universe=manual29, sector=False, icir=60, cap=0),
        "D1 商品+板块+ICIR": dict(universe=comm23, sector=True, icir=60, cap=0),
        "E1 配额cap=3": dict(universe=manual29, sector=False, icir=0, cap=3),
        "E2 配额cap=4": dict(universe=manual29, sector=False, icir=0, cap=4),
        "E3 配额cap=2": dict(universe=manual29, sector=False, icir=0, cap=2),
    }
    print("===== 板块/ICIR/配额 对照 (B3 6因子) =====")
    for label, cfg_ in cases.items():
        s = backtest(runner, base, cfg_["universe"], "2025-01-01", "2026-05-15",
                     sector_neutral=cfg_["sector"], icir_window=cfg_["icir"],
                     sector_cap=cfg_["cap"])
        print(f"[{label}] {stat(s)}")
    print("===== 4 方案 × cap (0 vs 3) =====")
    for lab, F in factor_sets.items():
        s0 = backtest(runner, F, manual29, "2025-01-01", "2026-05-15", sector_cap=0)
        s3 = backtest(runner, F, manual29, "2025-01-01", "2026-05-15", sector_cap=3)
        print(f"[{lab}] cap0: {stat(s0)}")
        print(f"[{lab}] cap3: {stat(s3)}")


if __name__ == "__main__":
    main()
