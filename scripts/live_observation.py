"""组合观测工具 (通用): 任意因子集 × 任意日期区间 的等权打分+池内rp组合.

用途:
  - 实盘观测: 在研究终点(data_research_end)之后的区间运行固定因子集, 观测"实盘成绩"
  - 一般回测: 任意因子集在任意区间的组合表现

设计原则:
  - 普适: 因子集/方向/区间/权重全部参数化, 不为任何特定因子或组合特例化
  - 零侵入: 不修改 research/backtest/combined.py
  - 可重跑: 随时更换因子集或区间重新运行

用法:
  python scripts/live_observation.py \
    --factors "intraday_jump_intensity_20d:-1,intraday_settle_close_basis_20d:-1" \
    --start 2026-05-16 --end 2026-07-31
  # 或 --factors-file factors.yaml (yaml: {name: direction})
  # 不传 --start 时默认 research_end+1 (研究终点后第一日)

输出 (--out 目录):
  holdings.csv     每日目标持仓 (品种×权重)
  metrics.json     年化/夏普/回撤/波动/换手/月收益
  nav.png          净值曲线
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import load_config
from factors.engine import FactorEngine


def parse_factors(spec: str) -> dict:
    """解析 'name:dir,name2:dir2' -> {name: direction}."""
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, d = item.rsplit(":", 1)
            out[name.strip()] = int(d.strip())
        else:
            out[item] = 1
    return out


def backtest_series(engine, runner, cal, universe, factors: dict, top_n: int = 10,
                    weight_scheme: str = "rp") -> tuple[pd.Series, dict]:
    """等权打分 + Top/Bottom 选池 + 池内加权 (rp=波动率倒数 / equal)."""
    names = list(factors)
    computed = engine.compute_factors(names, cal, universe, parallel=True)
    score = pd.DataFrame(index=cal, columns=universe, dtype=float)
    for n, d in factors.items():
        r = computed[n].rank(axis=1, pct=True)
        score = score.add(r if d == 1 else (1 - r), fill_value=0)
    score = score.div(len(factors))
    close = runner.data_manager.get("close", cal, universe)
    fwd = close.pct_change().shift(-1)
    vol20 = close.pct_change().rolling(20, min_periods=10).std(ddof=0)
    rebal = score.resample("W-FRI").last()

    holdings_rows = []
    rets = []
    prev_long, prev_short = set(), set()
    for t in rebal.index:
        row = rebal.loc[t].dropna()
        if len(row) < 2 * top_n:
            continue
        rk = row.rank(ascending=False)
        top = rk[rk <= top_n].index
        bot = rk[rk > len(rk) - top_n].index
        vt = vol20.loc[t] if t in vol20.index else vol20.asof(t)

        def pool_w(pool):
            if weight_scheme == "equal":
                return {c: 1.0 / len(pool) for c in pool} if pool else None
            v = vt[pool].replace(0, np.nan).dropna()
            if v.empty:
                return None
            w = 1.0 / v
            return (w / w.sum()).to_dict()

        wl, ws = pool_w(top), pool_w(bot)
        w = pd.Series(0.0, index=universe)
        if wl:
            for k, v in wl.items():
                w[k] += v
        if ws:
            for k, v in ws.items():
                w[k] -= v
        # 记录持仓 (每周最后一个交易日)
        holdings_rows.append({"date": t.strftime("%Y-%m-%d"),
                              **{c: round(float(w[c]), 5) for c in w.index if abs(w[c]) > 1e-6}})
        wd = cal[(cal >= t) & (cal < t + pd.Timedelta(days=7))]
        for d in wd:
            if d in fwd.index:
                r = fwd.loc[d]
                lr = sum(r[c] * wi for c, wi in wl.items()) if wl else 0.0
                sr = sum(r[c] * wi for c, wi in ws.items()) if ws else 0.0
                rets.append((d, float(lr - sr)))
    s = pd.Series({d: v for d, v in rets}).sort_index().dropna()
    return s, {"holdings": holdings_rows}


def metrics(ret: pd.Series) -> dict:
    if len(ret) < 5:
        return {"n_days": int(len(ret))}
    ann = ret.mean() * 252
    vol = ret.std(ddof=0) * np.sqrt(252)
    nav = (1 + ret).cumprod()
    dd = (nav / nav.cummax() - 1).min()
    # 月度收益
    monthly = ret.groupby(ret.index.to_period("M")).sum()
    return {
        "n_days": int(len(ret)),
        "annual_return": round(float(ann), 6),
        "sharpe": round(float(ann / vol), 4) if vol > 0 else None,
        "volatility": round(float(vol), 6),
        "max_drawdown": round(float(dd), 6),
        "total_return": round(float(nav.iloc[-1] - 1), 6),
        "monthly_returns": {str(k): round(float(v), 6) for k, v in monthly.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="组合观测工具 (任意因子集×任意区间)")
    parser.add_argument("--config", default="config/intraday_backtest.yaml")
    parser.add_argument("--factors", default=None,
                        help="因子集 'name:dir,name2:dir2' (dir=1正/-1负); 缺省用 combined.FACTORS")
    parser.add_argument("--factors-file", default=None, help="yaml 因子文件 {name: dir}")
    parser.add_argument("--start", default=None, help="起始日 (默认 research_end 次日)")
    parser.add_argument("--end", default=None, help="结束日 (默认数据最新)")
    parser.add_argument("--weight", default="rp", choices=["rp", "equal"])
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--out", default="runs/observation", help="输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config)
    from pipeline.runner import PipelineRunner
    runner = PipelineRunner(config=cfg)
    engine = FactorEngine(runner.data_manager)
    universe = list(cfg.universe)

    if args.factors_file:
        import yaml
        factors = {k: int(v) for k, v in yaml.safe_load(open(args.factors_file, encoding="utf-8")).items()}
    elif args.factors:
        factors = parse_factors(args.factors)
    else:
        from strategies.combined import FACTORS
        factors = dict(FACTORS)

    # 默认区间: research_end 次日 ~ 数据最新
    research_end = pd.Timestamp(getattr(cfg.date_range, "end", "2026-05-15"))
    start = pd.Timestamp(args.start) if args.start else (research_end + pd.Timedelta(days=1))
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp("2026-12-31")
    cal = pd.DatetimeIndex(runner.data_manager.get_calendar(start, end))
    if len(cal) == 0:
        print(f"区间 {start.date()}~{end.date()} 无数据 (研究终点后数据未更新)")
        return

    s, info = backtest_series(engine, runner, cal, universe, factors,
                              top_n=args.topn, weight_scheme=args.weight)
    m = metrics(s)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if info["holdings"]:
        pd.DataFrame(info["holdings"]).to_csv(out / "holdings.csv", index=False)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"factors": factors, "start": str(start.date()), "end": str(cal.max().date()),
                   **m}, f, ensure_ascii=False, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 5))
    nav = (1 + s).cumprod()
    ax.plot(nav.index, nav.values, color="#2ecc71", linewidth=1.5)
    ax.set_title(f"观测净值 ({len(factors)}因子, {start.date()}~{cal.max().date()})")
    ax.set_ylabel("净值")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "nav.png", dpi=150)

    print(f"区间: {start.date()} ~ {cal.max().date()} ({len(cal)}交易日, {len(factors)}因子)")
    print(f"年化={m.get('annual_return', 0):.2%} 夏普={m.get('sharpe', 0):.2f} "
          f"回撤={m.get('max_drawdown', 0):.2%} 波动={m.get('volatility', 0):.2%} "
          f"总收益={m.get('total_return', 0):.2%} 天数={m.get('n_days', 0)}")
    print(f"持仓明细: {out/'holdings.csv'} | 指标: {out/'metrics.json'} | 净值图: {out/'nav.png'}")


if __name__ == "__main__":
    main()
