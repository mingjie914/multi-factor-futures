"""重新生成 #351-#370 (20个) — 修正聚合模式 (逐列循环, 参考 #336).

正确模式:
  for dt in days:
      vals = {}
      for col in cols:          # 逐品种标量
          ...
          vals[col] = float(...)
      if vals: interactions[dt] = pd.Series(vals)
  daily = pd.DataFrame(interactions).T
  daily.index = pd.DatetimeIndex(daily.index)
  return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)
"""


def block(num, name, title, doc, desc, direction, body):
    """body: 函数体缩进文本, 以 8 空格缩进."""
    cls = "".join(w.capitalize() for w in name.split("_")) + "20d"
    return f'''
# {num}. {name} — {title}
# ═══════════════════════════════════════════════════════════════════════════════

@register_factor("{name}_20d", category="intraday_advanced")
class {cls}(Factor):
    """{doc}
    方向: {direction}.
    """
    name = "{name}_20d"
    category = "intraday_advanced"
    frequency = "daily"
    description = "{desc}"
    validation_horizons = (5, 10, 20)

    def dependencies(self) -> list:
        return []

    def compute(self, data, dates, universe):
{body}'''


PANEL_GUARD = '''        panel = _get_minute_panel(data, dates, universe, freq="1min")
        if not {FIELDS}.issubset(panel.keys()):
            return pd.DataFrame(np.nan, index=dates, columns=universe)
'''

# 通用尾部: interactions dict -> daily
TAIL = '''        if not interactions:
            return pd.DataFrame(np.nan, index=dates, columns=universe)
        daily = pd.DataFrame(interactions).T
        daily.index = pd.DatetimeIndex(daily.index)
        return daily.rolling(20, min_periods=5).mean().reindex(dates).shift(1).reindex(columns=universe)
'''

B = []

# ── A. 波动-流动性交互延伸 (351-355) ────────────────────────────────────────
B.append(block(351, "intraday_liq_shock_freq", "流动性冲击频率",
    "流动性冲击频率因子 (Amihud 突增>μ+σ 的分钟占比). 高频冲击=流动性脆弱, 负向.",
    "Amihud突增频率 (脆弱=负向)", "负向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "amount"}') + '''        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_a = amihud.loc[day == dt]
            if len(grp_a) < 30:
                continue
            vals = {}
            for col in grp_a.columns:
                a = grp_a[col].dropna()
                if len(a) < 30:
                    continue
                mu, sigma = a.mean(), a.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    continue
                vals[col] = float((a > mu + sigma).mean())
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(352, "intraday_amihud_slope", "Amihud 日内斜率",
    "Amihud 日内演化斜率因子 (后半段均值 - 前半段均值). 冲击恶化=流动性恶化, 负向.",
    "Amihud日内恶化斜率 (负向)", "负向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "amount"}') + '''        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_a = amihud.loc[day == dt]
            if len(grp_a) < 60:
                continue
            vals = {}
            for col in grp_a.columns:
                a = grp_a[col].dropna()
                if len(a) < 60:
                    continue
                half = len(a) // 2
                vals[col] = float(a.iloc[half:].mean() - a.iloc[:half].mean())
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(353, "intraday_vol_liq_divergence", "波动-流动性背离",
    "波动与流动性背离因子 (波动高位但Amihud低位的占比). 高波动无冲击=有承接, 正向.",
    "波动无冲击占比 (健康=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "amount"}') + '''        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 30:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 30:
                    continue
                r_c, a_c = r.loc[common], a.loc[common]
                rmed, amed = r_c.median(), a_c.median()
                if pd.isna(rmed) or pd.isna(amed):
                    continue
                vals[col] = float(((r_c > rmed) & (a_c < amed)).mean())
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(354, "intraday_vol_surge_liq_before", "波动突增前流动性",
    "波动突增前流动性水平因子 (波动突增分钟前20分钟平均Amihud). 突增前流动性差=真实冲击, 负向.",
    "突增前流动性 (冲击真实性, 负向)", "负向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "amount"}') + '''        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 60:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 60:
                    continue
                r_c, a_c = r.loc[common], a.loc[common]
                mu, sigma = r_c.mean(), r_c.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    continue
                surge_idx = np.where(r_c.values > mu + 2 * sigma)[0]
                before = []
                for idx in surge_idx:
                    lo = max(0, idx - 20)
                    win = a_c.iloc[lo:idx].dropna()
                    if len(win) > 5:
                        before.append(win.mean())
                vals[col] = float(np.mean(before)) if before else np.nan
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(355, "intraday_liq_resilience", "流动性恢复速度",
    "流动性恢复速度因子 (冲击后Amihud回落半周期, 取负). 恢复快=市场韧性强, 正向.",
    "冲击后流动性恢复速度 (韧性=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "amount"}') + '''        close, amount = panel["close"], panel["amount"]
        ret_abs = close.pct_change().abs()
        amihud = ret_abs / (amount + 1e-12)
        day = ret_abs.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret_abs.loc[day == dt]
            grp_a = amihud.loc[day == dt]
            if len(grp_r) < 60:
                continue
            vals = {}
            for col in grp_r.columns:
                r = grp_r[col].dropna()
                a = grp_a[col].dropna()
                common = r.index.intersection(a.index)
                if len(common) < 60:
                    continue
                r_c, a_c = r.loc[common], a.loc[common]
                mu, sigma = r_c.mean(), r_c.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    continue
                amed = a_c.median()
                surge_idx = np.where(r_c.values > mu + 2 * sigma)[0]
                half_lives = []
                for idx in surge_idx:
                    if idx + 1 >= len(a_c):
                        continue
                    tail = a_c.iloc[idx + 1:]
                    below = np.where(tail.values < amed)[0]
                    if len(below) > 0:
                        half_lives.append(below[0] + 1)
                vals[col] = -float(np.mean(half_lives)) if half_lives else np.nan
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

# ── B. 跨品种截面相对延伸 (356-360) ────────────────────────────────────────
B.append(block(356, "intraday_cross_momentum_slope", "截面动量斜率",
    "截面动量演化因子 (日内前半-后半截面排名的相关). 截面动量稳定=趋势延续, 正向.",
    "截面动量稳定性 (延续=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close"}') + '''        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            half = len(grp) // 2
            ret_h1 = grp.iloc[half].div(grp.iloc[0].replace(0, np.nan)).dropna()
            ret_h2 = grp.iloc[-1].div(grp.iloc[half].replace(0, np.nan)).dropna()
            common = ret_h1.index.intersection(ret_h2.index)
            if len(common) < 10:
                continue
            corr = ret_h1[common].corr(ret_h2[common])
            if not pd.isna(corr):
                interactions[dt] = pd.Series({"x": float(corr)})
''' + TAIL.replace("pd.DataFrame(interactions).T", "pd.DataFrame(interactions).T.dropna(axis=1)").replace(".reindex(columns=universe)", ".reindex(columns=universe)")))

B.append(block(357, "intraday_cross_divergence", "截面发散度",
    "截面收益发散度因子 (日内截面收益的标准差). 发散加剧=信息不对称增加, 负向.",
    "截面发散度 (分歧=负向)", "负向",
    PANEL_GUARD.replace("{FIELDS}", '{"close"}') + '''        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            ret_cs = grp.iloc[-1].div(grp.iloc[0].replace(0, np.nan)).dropna()
            sd = ret_cs.std(ddof=0)
            if not pd.isna(sd):
                interactions[dt] = pd.Series({"x": float(sd)})
''' + TAIL.replace("pd.DataFrame(interactions).T", "pd.DataFrame(interactions).T.dropna(axis=1)").replace(".reindex(columns=universe)", ".reindex(columns=universe)")))

B.append(block(358, "intraday_cross_concentration", "截面收益集中度",
    "截面收益集中度因子 (前3强品种收益占全部正收益比例). 集中=龙头驱动, 正向.",
    "截面收益集中度 (龙头驱动=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close"}') + '''        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            ret_cs = grp.iloc[-1].div(grp.iloc[0].replace(0, np.nan)).dropna()
            pos = ret_cs[ret_cs > 0]
            if len(pos) < 5 or pos.sum() <= 0:
                continue
            top3 = pos.nlargest(3).sum()
            interactions[dt] = pd.Series({"x": float(top3 / pos.sum())})
''' + TAIL.replace("pd.DataFrame(interactions).T", "pd.DataFrame(interactions).T.dropna(axis=1)").replace(".reindex(columns=universe)", ".reindex(columns=universe)")))

B.append(block(359, "intraday_cross_rank_stability", "截面排名稳定性",
    "截面排名稳定性因子 (日内排名与前日排名的一致性). 排名稳定=趋势持续, 正向.",
    "截面排名稳定性 (趋势=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close"}') + '''        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        prev_rank = None
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            ret_cs = grp.iloc[-1].div(grp.iloc[0].replace(0, np.nan)).dropna()
            rank = ret_cs.rank(pct=True)
            if prev_rank is not None:
                common = rank.index.intersection(prev_rank.index)
                if len(common) >= 10:
                    corr = rank[common].corr(prev_rank[common])
                    if not pd.isna(corr):
                        interactions[dt] = pd.Series({"x": float(corr)})
            prev_rank = rank
''' + TAIL.replace("pd.DataFrame(interactions).T", "pd.DataFrame(interactions).T.dropna(axis=1)").replace(".reindex(columns=universe)", ".reindex(columns=universe)")))

B.append(block(360, "intraday_cross_leader_follow", "截面龙头跟随",
    "截面龙头跟随因子 (前日龙头今日仍居前的比例). 龙头延续=资金抱团, 正向.",
    "截面龙头延续 (抱团=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close"}') + '''        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        prev_leader = None
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            ret_cs = grp.iloc[-1].div(grp.iloc[0].replace(0, np.nan)).dropna()
            leader = set(ret_cs.nlargest(3).index)
            if prev_leader is not None:
                overlap = len(leader & prev_leader) / max(len(prev_leader), 1)
                interactions[dt] = pd.Series({"x": float(overlap)})
            prev_leader = leader
''' + TAIL.replace("pd.DataFrame(interactions).T", "pd.DataFrame(interactions).T.dropna(axis=1)").replace(".reindex(columns=universe)", ".reindex(columns=universe)")))

# ── C. 订单簿失衡代理延伸 (361-365) ────────────────────────────────────────
B.append(block(361, "intraday_buy_aggression", "主动买入强度",
    "主动买入强度因子 (收盘>开盘分钟计入主动买入, 其成交量占比). 买入攻击性=看涨, 正向.",
    "主动买入量占比 (攻击性=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "open", "volume"}') + '''        close, open_, volume = panel["close"], panel["open"], panel["volume"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            grp_v = volume.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns or col not in grp_v.columns:
                    continue
                c, o, v = grp_c[col].dropna(), grp_o[col].dropna(), grp_v[col].dropna()
                common = c.index.intersection(o.index).intersection(v.index)
                if len(common) < 30:
                    continue
                c_c, o_c, v_c = c.loc[common], o.loc[common], v.loc[common]
                buy_vol = v_c[c_c > o_c].sum()
                tot = v_c.sum()
                if tot <= 0 or pd.isna(tot):
                    continue
                vals[col] = float(buy_vol / tot)
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(362, "intraday_buy_sustain", "主动买入持续性",
    "主动买入持续性因子 (连续主动买入分钟的最长run). 持续买入=资金坚定, 正向.",
    "主动买入最长run (坚定=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "open", "volume"}') + '''        import itertools
        close, open_, volume = panel["close"], panel["open"], panel["volume"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns:
                    continue
                c, o = grp_c[col].dropna(), grp_o[col].dropna()
                common = c.index.intersection(o.index)
                if len(common) < 30:
                    continue
                buy = (c.loc[common] > o.loc[common]).astype(int).values
                runs = [len(list(g)) for k, g in itertools.groupby(buy) if k == 1]
                vals[col] = float(max(runs)) if runs else 0.0
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(363, "intraday_buy_sell_imbalance_slope", "买卖失衡斜率",
    "主动买卖失衡日内演化因子 (后半段失衡 - 前半段失衡). 失衡转买=尾盘吸筹, 正向.",
    "买卖失衡日内斜率 (尾盘吸筹=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "open", "volume"}') + '''        close, open_, volume = panel["close"], panel["open"], panel["volume"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            grp_v = volume.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns or col not in grp_v.columns:
                    continue
                c, o, v = grp_c[col].dropna(), grp_o[col].dropna(), grp_v[col].dropna()
                common = c.index.intersection(o.index).intersection(v.index)
                if len(common) < 60:
                    continue
                c_c, o_c, v_c = c.loc[common], o.loc[common], v.loc[common]
                buy = (c_c > o_c).astype(float)
                half = len(buy) // 2
                imb_h1 = (buy.iloc[:half] * v_c.iloc[:half]).sum() / v_c.iloc[:half].sum() if v_c.iloc[:half].sum() > 0 else np.nan
                imb_h2 = (buy.iloc[half:] * v_c.iloc[half:]).sum() / v_c.iloc[half:].sum() if v_c.iloc[half:].sum() > 0 else np.nan
                if pd.isna(imb_h1) or pd.isna(imb_h2):
                    continue
                vals[col] = float(imb_h2 - imb_h1)
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(364, "intraday_big_buy_share", "大单买入占比",
    "大单买入占比因子 (主动买入且量>μ+2σ的分钟量占比). 大单买入=机构参与, 正向.",
    "大单主动买入占比 (机构=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "open", "volume"}') + '''        close, open_, volume = panel["close"], panel["open"], panel["volume"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            grp_v = volume.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns or col not in grp_v.columns:
                    continue
                c, o, v = grp_c[col].dropna(), grp_o[col].dropna(), grp_v[col].dropna()
                common = c.index.intersection(o.index).intersection(v.index)
                if len(common) < 30:
                    continue
                c_c, o_c, v_c = c.loc[common], o.loc[common], v.loc[common]
                mu, sigma = v_c.mean(), v_c.std(ddof=0)
                if sigma == 0 or pd.isna(sigma):
                    continue
                big_buy = v_c[(c_c > o_c) & (v_c > mu + 2 * sigma)].sum()
                tot = v_c.sum()
                if tot <= 0 or pd.isna(tot):
                    continue
                vals[col] = float(big_buy / tot)
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(365, "intraday_order_flow_asymmetry", "订单流不对称",
    "订单流不对称因子 (主动买量-主动卖量)/总成交量. 净主动买入=看涨共识, 正向.",
    "订单流净失衡 (共识=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "open", "volume"}') + '''        close, open_, volume = panel["close"], panel["open"], panel["volume"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            grp_v = volume.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns or col not in grp_v.columns:
                    continue
                c, o, v = grp_c[col].dropna(), grp_o[col].dropna(), grp_v[col].dropna()
                common = c.index.intersection(o.index).intersection(v.index)
                if len(common) < 30:
                    continue
                c_c, o_c, v_c = c.loc[common], o.loc[common], v.loc[common]
                buy = (c_c > o_c).astype(float)
                buy_vol = (buy * v_c).sum()
                sell_vol = v_c.sum() - buy_vol
                tot = v_c.sum()
                if tot <= 0 or pd.isna(tot):
                    continue
                vals[col] = float((buy_vol - sell_vol) / tot)
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

# ── D. 多时点交互/组合形态 (366-370) ────────────────────────────────────────
B.append(block(366, "intraday_open_close_drift", "开盘-收盘漂移",
    "开盘收盘漂移因子 (日内收益 vs 开盘跳空方向一致性). 跳空被日内确认=共识, 正向.",
    "跳空日内确认 (共识=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "open"}') + '''        close, open_ = panel["close"], panel["open"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_c = close.loc[day == dt]
            grp_o = open_.loc[day == dt]
            vals = {}
            for col in grp_c.columns:
                if col not in grp_o.columns:
                    continue
                c, o = grp_c[col].dropna(), grp_o[col].dropna()
                common = c.index.intersection(o.index)
                if len(common) < 30:
                    continue
                c_c, o_c = c.loc[common], o.loc[common]
                intraday = c_c.iloc[-1] - o_c.iloc[0]
                gap = o_c.iloc[0] - c_c.iloc[0]
                vals[col] = float(np.sign(intraday) == np.sign(gap))
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(367, "intraday_session_shape", "交易时段形态",
    "交易时段形态因子 (早/中/晚三段收益同向度). 三段同向=趋势全天延续, 正向.",
    "三段收益同向度 (趋势=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close"}') + '''        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 90:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 90:
                    continue
                third = len(c) // 3
                seg1 = c.iloc[third] / c.iloc[0] - 1
                seg2 = c.iloc[2 * third] / c.iloc[third] - 1
                seg3 = c.iloc[-1] / c.iloc[2 * third] - 1
                signs = np.sign([seg1, seg2, seg3])
                vals[col] = float((signs == signs[0]).mean())
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(368, "intraday_vol_volume_sync", "量价同步度",
    "量价同步度因子 (放量分钟与涨跌方向的一致性). 放量同向=资金驱动, 正向.",
    "放量方向一致性 (资金驱动=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close", "volume"}') + '''        close, volume = panel["close"], panel["volume"]
        ret = close.pct_change()
        day = ret.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp_r = ret.loc[day == dt]
            grp_v = volume.loc[day == dt]
            vals = {}
            for col in grp_r.columns:
                if col not in grp_v.columns:
                    continue
                r, v = grp_r[col].dropna(), grp_v[col].dropna()
                common = r.index.intersection(v.index)
                if len(common) < 30:
                    continue
                r_c, v_c = r.loc[common], v.loc[common]
                vmed = v_c.median()
                if vmed <= 0 or pd.isna(vmed):
                    continue
                high_vol = v_c > vmed
                if high_vol.sum() == 0:
                    continue
                rsign = np.sign(r_c)
                vals[col] = float(np.mean(rsign[high_vol] == np.sign(r_c.median())) if r_c.median() != 0 else np.mean(rsign[high_vol] == 1))
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(369, "intraday_momentum_vol_adjusted", "波动调整动量",
    "波动调整动量因子 (日内动量 / 日内波动率). 单位风险动量高=效率高, 正向.",
    "单位风险动量 (效率=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close"}') + '''        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 60:
                    continue
                ret_day = c.iloc[-1] / c.iloc[0] - 1
                vol = c.pct_change().std(ddof=0)
                if vol == 0 or pd.isna(vol):
                    continue
                vals[col] = float(ret_day / vol)
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))

B.append(block(370, "intraday_path_smoothness", "路径平滑度",
    "路径平滑度因子 (日内收益二阶差分平方和的倒数). 平滑=有序推进, 正向.",
    "路径平滑度 (有序=正向)", "正向",
    PANEL_GUARD.replace("{FIELDS}", '{"close"}') + '''        close = panel["close"]
        day = close.index.normalize()
        interactions: dict = {}
        for dt in sorted(set(day)):
            grp = close.loc[day == dt]
            if len(grp) < 60:
                continue
            vals = {}
            for col in grp.columns:
                c = grp[col].dropna()
                if len(c) < 60:
                    continue
                d2 = c.pct_change().diff()
                rough = d2.pow(2).sum()
                if rough == 0 or pd.isna(rough):
                    continue
                vals[col] = float(1.0 / rough)
            if vals:
                interactions[dt] = pd.Series(vals)
''' + TAIL))


if __name__ == "__main__":
    path = "factors/library/intraday.py"
    s = open(path, encoding="utf-8").read()
    s = s.rstrip() + "\n" + "\n".join(B)
    open(path, "w", encoding="utf-8").write(s)
    print(f"已追加 {len(B)} 个因子 (#351-#370)")
