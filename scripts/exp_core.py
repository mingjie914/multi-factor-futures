"""exp_core — 对照实验共享核心 (独立于 production).

模块化设计:
  - 不修改 strategies/combined.py / factors/ 等 production 代码
  - 统一调度: run_all.py 调 4 个实验, 共享本核心
  - 因子集: 默认 6 因子生产; 可选全有效因子 (27候选+6=33)
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from core.config import load_config
from pipeline.runner import PipelineRunner
from factors.engine import FactorEngine
from optimization.risk_budgeting import RiskBudgetingOptimizer
from strategies.combined import SECTOR_MAP, SECTOR_CAP

UNIV38 = ["A", "AG", "AL", "AU", "CU", "FU", "HC", "I", "IC", "IF", "IH", "J", "JM",
          "M", "MA", "NI", "P", "RB", "RM", "RU", "SA", "SN", "SR", "T", "TA", "TL",
          "TS", "Y", "ZN", "IM", "TF", "CF", "OI", "LH", "JD", "SC", "V", "UR"]
PROD6 = {
    "intraday_jump_intensity_20d": -1, "intraday_price_peak_count_20d": 1,
    "intraday_realised_skewness_20d": 1, "intraday_dtws_20d": 1,
    "intraday_drip_stone_20d": -1, "intraday_peak_ridge_ratio_20d": -1,
}
# 27 候选 (来自 docs/有效因子库.md 第二层) + 6 生产 = 33 全有效
CAND27 = [
    "intraday_zero_ret_freq_20d", "intraday_open_close_drift_20d",
    "intraday_volatility_clustering_20d", "intraday_oi_vol_corr_daily_20d",
    "intraday_oi_time_centroid_20d", "intraday_wash_trade_20d",
    "intraday_settle_position_20d", "intraday_cross_vol_20d",
    "intraday_amihud_vol_ratio_20d", "intraday_oi_skew_stability_20d",
    "intraday_depth_trend_20d", "intraday_open_close_volume_ratio_20d",
    "intraday_oi_quantile_range_20d", "intraday_settle_gap_20d",
    "intraday_amihud_trend_20d", "intraday_price_delay_20d",
    "intraday_overnight_absorption_20d", "intraday_session_symmetry_20d",
    "intraday_lowest_time_20d", "intraday_oi_peak_ridge_ratio_20d",
    "intraday_seat_long_short_seat_ratio_20d", "intraday_volume_rank_ratio_20d",
    "intraday_herding_20d", "intraday_volume_time_shape_20d",
    "intraday_extreme_freq_balance_20d", "intraday_term_vol_ratio_20d",
    "intraday_turnover_velocity_20d",
]
# 方向: 依据 docs 有效因子库 (IC>0 正向, IC<0 负向)
CAND_DIR = {
    "intraday_zero_ret_freq_20d": -1, "intraday_open_close_drift_20d": 1,
    "intraday_volatility_clustering_20d": -1, "intraday_oi_vol_corr_daily_20d": 1,
    "intraday_oi_time_centroid_20d": -1, "intraday_wash_trade_20d": 1,
    "intraday_settle_position_20d": -1, "intraday_cross_vol_20d": 1,
    "intraday_amihud_vol_ratio_20d": 1, "intraday_oi_skew_stability_20d": -1,
    "intraday_depth_trend_20d": 1, "intraday_open_close_volume_ratio_20d": -1,
    "intraday_oi_quantile_range_20d": -1, "intraday_settle_gap_20d": 1,
    "intraday_amihud_trend_20d": 1, "intraday_price_delay_20d": -1,
    "intraday_overnight_absorption_20d": 1, "intraday_session_symmetry_20d": -1,
    "intraday_lowest_time_20d": 1, "intraday_oi_peak_ridge_ratio_20d": -1,
    "intraday_seat_long_short_seat_ratio_20d": 1, "intraday_volume_rank_ratio_20d": 1,
    "intraday_herding_20d": 1, "intraday_volume_time_shape_20d": 1,
    "intraday_extreme_freq_balance_20d": -1, "intraday_term_vol_ratio_20d": 1,
    "intraday_turnover_velocity_20d": 1,
}


class ExpEnv:
    """统一实验环境: 数据/因子/日历缓存."""

    def __init__(self, factors=None):
        self.cfg = load_config('config/intraday_backtest.yaml')
        self.runner = PipelineRunner(config=self.cfg)
        self.cal = pd.DatetimeIndex(self.runner.data_manager.get_calendar(
            pd.Timestamp('2025-01-01'), pd.Timestamp('2026-07-31')))
        self.u = list(UNIV38)
        self.engine = FactorEngine(self.runner.data_manager)
        self.close = self.runner.data_manager.get('close', self.cal, self.u)
        self.daily_ret = self.close.pct_change()
        self.vol20 = self.close.pct_change().rolling(20, min_periods=10).std(ddof=0)
        # 默认 6 因子
        self.factors = factors if factors is not None else dict(PROD6)
        self._comp = None
        self.sector_of = {}
        for sec, mem in SECTOR_MAP.items():
            for m in mem:
                if m in self.u:
                    self.sector_of[m] = sec

    def compute_scores(self):
        """等权合成截面得分 (factor rank 加权)."""
        if self._comp is None:
            self._comp = self.engine.compute_factors(list(self.factors), self.cal, self.u, parallel=True)
        score = pd.DataFrame(index=self.cal, columns=self.u, dtype=float)
        for n, direction in self.factors.items():
            r = self._comp[n].rank(axis=1, pct=True)
            score = score.add(r if direction == 1 else (1 - r), fill_value=0)
        return score.div(len(self.factors))

    def capped(self, row, cap_n=SECTOR_CAP, ascending=False):
        """板块配额选池: 按得分排序取 top10, 每板块≤cap.
        ascending=False 取最高分(多头), True 取最低分(空头)."""
        order = row.sort_values(ascending=ascending).index.tolist()
        picks, counts = [], {}
        for s in order:
            sec = self.sector_of.get(s, '其他')
            if counts.get(sec, 0) >= cap_n:
                continue
            picks.append(s)
            counts[sec] = counts.get(sec, 0) + 1
            if len(picks) >= 10:
                break
        return picks

    def erc_w(self, pool, t):
        """池内 ERC 权重 (正数, 归一化1)."""
        if len(pool) < 2:
            return None
        sd = t - pd.Timedelta(days=90)
        c = pd.DatetimeIndex(self.runner.data_manager.get_calendar(sd, t))
        rs = self.daily_ret.reindex(c)[list(pool)].dropna()
        if rs.shape[0] < 10:
            return None
        cov_raw = rs.cov().values
        cov = 0.7 * cov_raw + 0.3 * np.diag(np.diag(cov_raw))
        try:
            w = RiskBudgetingOptimizer._erc_weights(cov, np.ones(len(pool)))
            return dict(zip(pool, w))
        except (RuntimeError, ValueError):
            v = rs.std(ddof=0).replace(0, np.nan).dropna()
            if v.empty:
                return None
            w = (1.0 / v).values
            return dict(zip(pool, w / w.sum()))


def stats(s):
    """统计: 年化/夏普/回撤/波动/OOS/实盘."""
    if len(s) < 3:
        return None
    ann = s.mean() * 252
    vol = s.std(ddof=0) * np.sqrt(252)
    navs = (1 + s).cumprod()
    mdd = (navs / navs.cummax() - 1).min()
    oos = s[(s.index >= pd.Timestamp('2026-03-01')) & (s.index <= pd.Timestamp('2026-05-15'))]
    live = s[s.index > pd.Timestamp('2026-05-15')]
    oos_sh = oos.mean() * 252 / (oos.std(ddof=0) * np.sqrt(252)) if len(oos) > 2 and oos.std() > 0 else 0
    live_sh = live.mean() * 252 / (live.std(ddof=0) * np.sqrt(252)) if len(live) > 2 and live.std() > 0 else 0
    return {'ann': ann, 'sharpe': ann / vol if vol > 0 else 0, 'mdd': mdd, 'vol': vol,
            'oos': oos_sh, 'live': live_sh, 'n': len(s)}


def format_stats(st):
    if st is None:
        return 'n/a'
    return (f"年化={st['ann']:.1%} 夏普={st['sharpe']:.2f} 回撤={st['mdd']:.1%} "
            f"波动={st['vol']:.1%} OOS={st['oos']:.2f} 实盘={st['live']:.2f}")
