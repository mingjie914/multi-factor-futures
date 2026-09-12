"""Explicit frozen-cohort attribution and deletion diagnostics; no promotion."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

from backtest.research_ledger import build_close_marked_ledger
from backtest.metrics import compute_turnover_metrics
from optimization.costs import static_cost_stress, evaluate_cost_sensitivity
from core.config import load_config, load_strategy_library
from core.sectors import SECTOR_MAP
from research.historical_portfolio_search import (
    CausalEligibilityEnvironment, PortfolioEvaluator, performance_metrics, robust_summary,
)
from research.portfolio_experiment_support import FactorPanelRunner as Runner
from workflows.experiments.historical_portfolio_search import _search_source_snapshot
from workflows.factor_selection import _cluster, _configured_cost_model, _exposure_correlation, _production_recipe


NAMES = ("板块量持", "席位价形", "精简板块", "波动跟随", "结构融合")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def signature(members):
    return json.dumps(sorted((n, int(d)) for n, d in members.items()))


def deletion_jobs(peers, clusters):
    """Every single-factor and within-cohort cluster deletion, shared by signature."""
    jobs = {}
    for peer in peers:
        members = peer["directions"]
        groups = [("factor", n, [n]) for n in sorted(members)]
        groups += [("cluster", str(c), [n for n in members if clusters[n] == c])
                   for c in sorted({clusters[n] for n in members})]
        for kind, label, removed in groups:
            reduced = {n: d for n, d in sorted(members.items()) if n not in removed}
            key = signature(reduced)
            row = jobs.setdefault(key, {"directions": reduced, "uses": []})
            row["uses"].append({"strategy": peer["name"], "kind": kind,
                                "removed": removed, "label": label})
    return jobs


def attributed_period(daily, contributions, weights, sector_of, fine_sector_of):
    """Additive NAV-point attribution, with fees separate and each interval rebased."""
    idx = daily.index
    c, w = contributions.loc[idx], weights.loc[idx]
    net = daily["net_return"].to_numpy(dtype=float)
    costs = daily[["trade_cost", "holding_cost"]].to_numpy(dtype=float)
    values = c.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(net).all() or not np.isfinite(costs).all():
        raise ValueError("nonfinite attribution inputs")
    np.testing.assert_allclose(values.sum(axis=1) - costs.sum(axis=1), net, rtol=0, atol=1e-12)
    if (net <= -1).any():
        raise ValueError("nonpositive NAV path")
    nav = np.cumprod(1 + net)
    previous = np.r_[1.0, nav[:-1]]
    linked = values * previous[:, None]
    fees = -(costs * previous[:, None]).sum(axis=0)
    total = float(nav[-1] - 1)
    np.testing.assert_allclose(linked.sum() + fees.sum(), total, rtol=0, atol=1e-10)
    gross = values.sum(axis=1)
    variance = float(np.var(gross, ddof=1)) if len(gross) > 1 else 0.0
    # Signed covariance shares sum to one for nonconstant gross returns; not net risk attribution.
    risk = ((values-values.mean(axis=0)).T @ (gross-gross.mean()) / (len(gross)-1) / variance
            if len(gross) > 1 and variance > 0 else np.full(values.shape[1], np.nan))
    peak = np.maximum.accumulate(np.r_[1.0, nav])
    trough = int(np.argmin(np.r_[1.0, nav] / peak - 1))
    peak_pos = int(np.argmax(np.r_[1.0, nav][:trough+1]))
    dd_linked = linked[peak_pos:trough].sum(axis=0) / peak[trough]
    rows = []
    for i, n in enumerate(c.columns):
        rows.append({"instrument": n, "sector": sector_of[n], "fine_sector": fine_sector_of[n],
            "contribution": float(linked[:, i].sum()),
            "positive_contribution": float(np.maximum(linked[:, i], 0).sum()),
            "negative_contribution": float(np.minimum(linked[:, i], 0).sum()),
            "long_contribution": float(linked[:, i][w[n].to_numpy() > 0].sum()),
            "short_contribution": float(linked[:, i][w[n].to_numpy() < 0].sum()),
            "mean_abs_exposure": float(w[n].abs().mean()), "max_abs_exposure": float(w[n].abs().max()),
            "active_days": int(w[n].abs().gt(1e-12).sum()),
            "gross_variance_share": float(risk[i]), "max_drawdown_gross_contribution": float(dd_linked[i])})
    sectors = {}
    for field in ("sector", "fine_sector"):
        sectors[field] = {label: float(sum(r["contribution"] for r in rows if r[field] == label))
                          for label in sorted({r[field] for r in rows})}
    return {"total_return": total, "trade_cost_contribution": float(fees[0]),
        "holding_cost_contribution": float(fees[1]), "assets": rows, **sectors,
        "long_contribution": sum(r["long_contribution"] for r in rows),
        "short_contribution": sum(r["short_contribution"] for r in rows),
        "reconciliation_error": float(linked.sum()+fees.sum()-total)}


def period_metrics(daily, start, cutoff):
    periods = {"selection": daily.loc[:cutoff].iloc[1:], "observation": daily.loc[daily.index > cutoff],
               "full": daily.iloc[1:]}
    result = {}
    for label, frame in periods.items():
        net = frame["net_return"]
        # Static fee stress holds the original realized exposures fixed, as in the old review.
        stressed = static_cost_stress(frame, trade_multiplier=2.0, holding_multiplier=2.0)
        result[label] = {"base": performance_metrics(net), "static_cost_2x": performance_metrics(stressed)}
    segments = [(pd.Timestamp(start), pd.Timestamp("2019-12-31")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
        (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        (pd.Timestamp("2025-01-01"), cutoff)]
    result["robustness"] = robust_summary(daily.loc[:cutoff], segments, initial_anchor=True)
    return result


def _report_peers(contract, catalog):
    """Select current non-archived definitions without changing frozen evidence."""
    saved = {p["name"]: p for p in contract["peers"]}
    subsets = {f.id: f for f in catalog.factor_sets}
    peers = []
    for entry in catalog.strategies:
        if entry.status == "archived":
            continue
        if entry.source != "effective_library":
            raise ValueError("current report requires effective-library strategies")
        peer = saved.get(entry.name)
        subset = subsets[entry.factor_set_id]
        if (peer is None or set(subset.factors) != set(peer["directions"])
                or subset.selection_context.get("directions") != peer["directions"]):
            raise ValueError(f"frozen report members/directions differ: {entry.id}")
        peers.append(peer)
    if not peers:
        raise ValueError("current report has no active strategies")
    return peers


def report(output, *, source_dir=None, catalog_path=None, factor_types=None):
    """Read completed artifacts only; do not select or mutate the cohort."""
    from run_portfolio_workflow import _write_comparison_plot
    from core.registry import get as registry_get
    import factors.library  # noqa: F401
    output = Path(output)
    source_dir = Path(source_dir) if source_dir else output
    if catalog_path and output.resolve() == source_dir.resolve():
        raise ValueError("filtered report must preserve its source evidence directory")
    contract = read_json(source_dir / "contract.json")
    peers, source = contract["peers"], contract["source"]
    if catalog_path:
        peers = _report_peers(contract, load_strategy_library(str(catalog_path)))
    peer_names = {p["name"] for p in peers}
    baseline = {n: r for n, r in read_json(source_dir / "baseline_summary.json").items() if n in peer_names}
    deletion_baseline = (read_json(Path(contract["deletion_reference"]) / "baseline_summary.json")
                         if contract.get("deletion_reference") else baseline)
    deletions = {k: {**r, "uses": [u for u in r["uses"] if u["strategy"] in peer_names]}
                 for k, r in read_json(source_dir / "deletion_summary.json").items()
                 if any(u["strategy"] in peer_names for u in r["uses"])}
    cluster_data = read_json(source_dir / "clusters.json")
    clusters = cluster_data["clusters"]
    library = {r["factor"]: r for r in read_json("factor_library/library.json")["factors"]}
    view_contract = output / "report_contract.json"
    factor_types = factor_types or (read_json(view_contract).get("factor_types", {}) if view_contract.is_file() else {})
    from research.governance import factor_family
    types = {n: factor_types.get(n, factor_family(n)) for p in peers for n in p["directions"]}
    if catalog_path and not set(types).issubset(library):
        raise ValueError("current report members must belong to effective library")
    output.mkdir(parents=True, exist_ok=True)
    if source_dir.resolve() != output.resolve():
        dump_json(output / "report_contract.json", {"source_dir": str(source_dir.resolve()),
            "catalog_path": str(Path(catalog_path).resolve()) if catalog_path else None,
            "peers": peers, "factor_types": types, "source": source,
            "source_sha256": {n: hashlib.sha256((source_dir / n).read_bytes()).hexdigest()
                for n in ("contract.json", "baseline_summary.json", "deletion_summary.json", "clusters.json", "results.duckdb", "accounting_review.json")
                if (source_dir / n).is_file()},
            "library_sha256": hashlib.sha256(Path("factor_library/library.json").read_bytes()).hexdigest(),
            "catalog_sha256": hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest() if catalog_path else None})
    repaired = set() if catalog_path else set(contract.get("recomputed_factors", []))
    exact_count = sum(bool(r["exact_nav_match"]) for r in baseline.values())
    lines = [f"# {len(peers)}组备选策略归因与因子结构报告", "",
        f"基准回测：{source['performance_start']}至{source['end']}。选择截止{source['selection_end']}，随后仅观察。",
        "不修改成员、方向、生产默认、有效库、准入阈值，不新增实盘批准。",
        (f"当前{len(peers)}组曲线均从已核验原生账本重绘，不改变日收益。贡献按各区间净值链式累加，费用独立列示。" if catalog_path else
         f"原生账本{exact_count}条净值与原基准逐点精确一致；其余为修复后固定成员对照。贡献按各区间净值链式累加，费用独立列示。"),
        f"本报告共{len(types)}个不同成员；K簇沿用原{len(clusters)}成员联合诊断的编号，不重新聚类或编号。因子类型是经济含义标签，簇是样本相关性分组，均不等于品种板块。簇不是独立利润源。",
        f"完整账本、逐品种逐年归因与剔除证据：[源证据目录]({source_dir.resolve().as_posix()})。本报告读取已完成账本生成展示，不在报告阶段重新选组或修改数据。", "",
        "## 同口径表现与静态双倍费用", "",
        "| 组合 | 选择期年化 | 夏普 | 最差历史段夏普 | 双倍费用夏普 | 观察期累计 | 双倍费用观察期累计 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    total_seconds = 0.0
    if repaired:
        previous = read_json(Path(contract["repair_reference"]) / "baseline_summary.json")
        intro = ["## 因果修复前后：固定成员，不重新选组", "",
            "修复前的新五组含跨日前视，以下原值仅用于量化修复影响，不代表可实现收益。旧8组不含修复因子，完整日账本及逐品种明细必须精确一致。",
            "#485/#491改为当日跨品种带截距回归；保留原分钟规则、20日聚合与滞后，#491仍输出负标准差。未改变库、方向或组合配置。", "",
            "| 组合 | 原选择期年化 | 修复后年化 | 原夏普 | 修复后夏普 | 原观察期累计 | 修复后观察期累计 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for peer in peers:
            if not repaired.intersection(peer["directions"]):
                continue
            a, b = previous[peer["name"]]["metrics"], baseline[peer["name"]]["metrics"]
            intro.append(f"| {peer['name']} | {a['selection']['base']['annual_return']:.2%} | {b['selection']['base']['annual_return']:.2%} | {a['selection']['base']['sharpe']:.3f} | {b['selection']['base']['sharpe']:.3f} | {a['observation']['base']['total_return']:.2%} | {b['observation']['base']['total_return']:.2%} |")
        lines[2:2] = intro + ["", "修复后的固定组合结果不等于重新选组成功；全库资格及选择路径需独立核验。未经全库因果审计，不宣称所有因子均无前视。", ""]
    for peer in peers:
        row = baseline[peer["name"]]; m = row["metrics"]
        a, s, b = m["selection"]["base"], m["selection"]["static_cost_2x"], m["observation"]
        lines.append(f"| {peer['name']} | {a['annual_return']:.2%} | {a['sharpe']:.2f} | "
            f"{m['robustness']['worst_sharpe']:.2f} | {s['sharpe']:.2f} | {b['base']['total_return']:.2%} | "
            f"{b['static_cost_2x']['total_return']:.2%} |")
        total_seconds += row["seconds"]
    lines += ["", "双倍费用固定原始实际仓位路径，只额外扣一份原账本交易费与持有费。不是重新优化仓位的成本情景，更不等于真实滑点/容量检验。", "",
              "年化换手（倍/年）＝252×日均完整成交权重之和。权重以归一化净值基准计量，不需要实际账户金额；这是研究净值账本，不是券商资金/保证金账户。每次换手用目标权重减去收益及费用结算后漂移的持仓权重，并按具体合约累计绝对差；买卖、开平及换月两条腿均计入，不除以2。",
              "200代表约200倍/年，日均约0.794个归一化基准，不是200%。总名义敞口2倍时，这约相当于日均成交总持仓名义金额的39.7%，不是每天更换39.7%的品种。全部更换一个2倍敞口组合，平旧2＋开新2，完整换手为4；已经包含杠杆，不再额外乘2。",
              "换手没有统一合理区间，应结合净收益、手续费、价差、冲击与容量判断。当前配置对完整成交名义量收取2bp，并按总敞口另计年化10.5bp移仓预留；200倍年换手的交易费用约4%/年，2倍敞口的移仓预留约0.21%/年（粗略线性估计，非精确复利归因）。该统一费率是假设，不等于已核实每个品种的实际成交成本。", "",
              "## 逐组归因、成员与边际效应", "",
              "贡献单位为区间净值百分点；毛贡献可大于净收益，因为费用独立扣除。下表同时给出选择期和观察期，不把短期表现用于改成员。",
              "源证据目录的baseline_summary.json保存逐品种逐年、多空、平均/峰值敞口和毛收益方差份额；results.duckdb保存逐日原生明细。", ""]
    structure_lines = ["## 组合因子结构总览", "",
        "| 组合 | 成员数 | 正/负方向 | 覆盖簇数 | 近似有效维数 | 类型构成 |", "|---|---:|---:|---:|---:|---|"]
    factor_corr = pd.DataFrame(cluster_data["correlation"], index=cluster_data["names"], columns=cluster_data["names"])
    for peer in peers:
        members = peer["directions"]
        eigenvalues = np.maximum(np.linalg.eigvalsh(factor_corr.loc[list(members), list(members)].to_numpy()), 0)
        dim = float(eigenvalues.sum()**2 / np.square(eigenvalues).sum())
        counts = Counter(types[n] for n in members)
        positives = sum(d > 0 for d in members.values())
        structure_lines.append(f"| {peer['name']} | {len(members)} | {positives}/{len(members)-positives} | {len({clusters[n] for n in members})} | {dim:.2f} | " + "、".join(f"{k}×{v}" for k, v in sorted(counts.items())) + " |")
    structure_lines += ["", "近似有效维数采用因子相关矩阵特征值参与率，负特征值截零；不是独立信号数，也不是越高越好。类型计数不代表收益贡献。best_period仅为准入证据标签，不决定调仓或持有期。", ""]
    lines[2:2] = structure_lines
    for peer in peers:
        name = peer["name"]; row = baseline[name]; attr = row["attribution"]
        deletion_base = deletion_baseline[peer["reference_name"] if contract.get("deletion_reference") else name]
        lines += [f"### {name}", "", "| 分组 | 选择期毛贡献 | 观察期毛贡献 |", "|---|---:|---:|"]
        for sector in attr["selection"]["sector"]:
            lines.append(f"| {sector} | {100*attr['selection']['sector'][sector]:.2f} | {100*attr['observation']['sector'][sector]:.2f} |")
        for label, key in (("多头合计", "long_contribution"), ("空头合计", "short_contribution"),
                           ("交易费用", "trade_cost_contribution"), ("持有费用", "holding_cost_contribution"), ("净收益", "total_return")):
            lines.append(f"| {label} | {100*attr['selection'][key]:.2f} | {100*attr['observation'][key]:.2f} |")
        lines += ["", "| 品种 | 主分组/细分映射 | 选择期毛贡献 | 观察期毛贡献 | 平均绝对敞口 |", "|---|---|---:|---:|---:|"]
        live = {r["instrument"]: r for r in attr["observation"]["assets"]}
        for r in sorted(attr["selection"]["assets"], key=lambda r: r["contribution"], reverse=True):
            lines.append(f"| {r['instrument']} | {r['sector']}/{r['fine_sector']} | {100*r['contribution']:.2f} | "
                         f"{100*live[r['instrument']]['contribution']:.2f} | {r['mean_abs_exposure']:.2%} |")
        lines += ["", "| 因子 | 类型 | 诊断簇 | 方向 | 原始bar | 准入预测期 | 资格快照 | 说明 |", "|---|---|---|---:|---|---|---|---|"]
        for n, direction in peer["directions"].items():
            cls = registry_get("factor", n); evidence = library.get(n, {})
            status = "修复后待再认证（原资格不沿用）" if n in repaired else evidence.get("status", "库外历史观察")
            lines.append(f"| {n} | {types[n]} | K{clusters[n]} | {direction:+d} | {getattr(cls, 'input_bar_frequency', '1min')} | "
                f"{evidence.get('best_period', '未确认')} | {status} | {getattr(cls, 'description', '').replace('|', '/')} |")
        effects = []
        for key, result in deletions.items():
            for use in result["uses"]:
                if use["strategy"] != name:
                    continue
                if result["status"] == "evaluated":
                    metric = result["metrics"]["selection"]["base"]
                    effects.append((metric["sharpe"]-deletion_base["metrics"]["selection"]["base"]["sharpe"], use,
                                    metric["annual_return"]-deletion_base["metrics"]["selection"]["base"]["annual_return"]))
                else:
                    lines.append(f"\n剔除{use['removed']}运行失败：{result['error']}；未回退参数。")
        effects.sort(key=lambda v: v[0])
        highlighted = effects[:3] + [e for e in effects[-3:] if e not in effects[:3]]
        lines += ["", "剔除后的变化：负值意味着该成员/簇在该搭配中有帮助，正值意味着删除后该指标改善；属于反事实边际效应，不能相加作为利润份额。仅展示选择期改善/损害最大的各3项，全部试验含观察期记录均保留。", "",
                  "| 剔除类型 | 剔除成员 | 夏普变化 | 年化变化 |", "|---|---|---:|---:|"]
        if contract.get("deletion_reference"):
            lines[-3:-3] = ["本节沿用冻结的历史剔除证据，变化相对同一旧试验基准计算，仅展示截至2026-05-15的选择期指标；本次延长的是完整组合回测，没有重跑剔除搜索，不与新版基准交叉相减。", ""]
        for delta, use, annual in highlighted:
            lines.append(f"| {use['kind']} | {', '.join(use['removed'])} | {delta:+.3f} | {annual:+.2%} |")
        lines.append("")
    total_seconds += sum(r["seconds"] for r in deletions.values())
    failures = sum(r["status"] != "evaluated" for r in deletions.values())
    reused = sum(bool(r.get("reused_from")) for r in deletions.values())
    lines += ["## 验收与限制", "", f"原试验{exact_count}条净值基准精确复现；本报告引用{len(deletions)}个去重剔除试验，失败{failures}个，其中{reused}项按相同方向/成员签名复用未变证据。源试验记录累计计时{total_seconds/3600:.2f}小时（不含共享准备和报告；不是本次重绘耗时）。",
        "每个区间品种贡献+独立费用项与净收益对账；有效仓位及资产收益恒等式由原生ResearchReturnLedger.validate验证。",
        "毛收益方差份额是品种贡献与总毛收益的协方差占比，可负；不是净收益风险份额或独立风险。最大回撤阶段毛贡献不包含独立费用。",
        "剔除试验是固定规则诊断，不按结果自动重新选组。该历史区间参与过研究，不能声称独立样本外；观察期亦已被查看。",
        "资格按当前库内各批次证据记录，不把局部复核当作全库重新认证。实盘滑点、成交延迟、容量仍未模拟。",
        "结果是观察策略的归因与稳健性证据，不自动加入目录、更不自动修改preferred。", ""]
    if (source_dir / "performance.json").is_file():
        profile = read_json(source_dir / "performance.json")
        lines += ["## 最新回测性能", "", f"本次墙钟{profile['wall_seconds']:.1f}秒、CPU {profile['cpu_seconds']:.1f}秒，共享因子面板准备{profile['panel_seconds']:.1f}秒；峰值工作集{profile['peak_working_set_mib']:.1f}MiB。各因子热点保存在源目录performance.json。", ""]
    plot_rows = []
    with duckdb.connect(str(source_dir / "results.duckdb"), read_only=True) as db:
        native = db.execute("SELECT date,strategy,nav,net_return,trade_cost,holding_cost,executed_traded_notional FROM daily ORDER BY date,strategy").df()
        nav = native.pivot(index="date", columns="strategy", values="nav").reindex(columns=[p["name"] for p in peers])
        nav = nav / nav.iloc[0]
        returns = native.pivot(index="date", columns="strategy", values="net_return").reindex(columns=nav.columns)
        cost_audits = {}
        lines += ["## 通用成本承受力诊断", "",
            "方法static_turnover_rate_v1：固定原账本归一化仓位与成交路径，仅替换交易费率，持有费用保持原值；不改变默认费用或策略。"
            "不是成交仿真，也不等于重算费用后的持仓漂移。盈亏平衡费率按区间复利净收益为零求解，包含保留的持有费，不是资金容量。",
            "| 组合 | 区间 | 完整年换手(倍) | 盈亏平衡费率(bp) | 2bp净年化 | 4bp净年化 | 8bp净年化 |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for peer in peers:
            frame = native.loc[native.strategy.eq(peer["name"])].set_index("date").iloc[1:]
            cost_audits[peer["name"]] = {}
            for key, label, interval in (("selection", "截止前", frame.loc[:source["selection_end"]]),
                                         ("observation", "观察期", frame.loc[frame.index > pd.Timestamp(source["selection_end"])]),
                                         ("full", "全期", frame)):
                if interval.empty:
                    continue
                audit = evaluate_cost_sensitivity(interval)
                cost_audits[peer["name"]][key] = audit
                be = audit["breakeven_trade_cost_rate"]
                be_label = f"{be * 10000:.2f}" if be is not None else audit["breakeven_status"]
                chosen = [next(s for s in audit["scenarios"] if s["trade_cost_rate"] == rate) for rate in (.0002, .0004, .0008)]
                values = [f"{s['metrics']['annual_return']:.2%}" if s["metrics"] else "净值耗尽" for s in chosen]
                lines.append(f"| {peer['name']} | {label} | {audit['annualized_turnover']:.2f} | {be_label} | " + " | ".join(values) + " |")
        lines += ["", "全部1/2/3/4/6/8bp情景的收益、夏普和回撤保存在cost_diagnostics.json。短观察段年化仅作描述，不能据此重新择优或声明独立样本外通过。", ""]
        dump_json(output / "cost_diagnostics.json", cost_audits)
        corr = returns.loc[:source["selection_end"]].iloc[1:].corr()
        dimension = float(np.trace(corr.to_numpy())**2 / np.square(corr.to_numpy()).sum())
        dump_json(output / "strategy_correlation.json", {"names": list(corr.columns), "selection_correlation": corr.to_numpy().tolist(), "participation_dimension": dimension})
        lines += ["## 收益相关性与近似有效维数", "", f"选择截止前{len(peers)}组日净收益相关矩阵的参与率维数为{dimension:.3f}；是二阶收益分散诊断，不是独立因子数或未来盈利保证。完整矩阵见strategy_correlation.json。", ""]
        lines += ["## 年度收益与滚动稳定性", "", "年度收益各年独立累计；首日为净值锚点。滚动252日正收益占比统计全回测期，不与仅截止前的占比混用。2026包含已查看的模拟实盘观察段，不用于重新选组。", "",
                  "| 组合 | 滚动252日正收益占比 | " + " | ".join(str(y) for y in sorted(returns.index.year.unique())) + " |",
                  "|---|---:|" + "---:|" * len(returns.index.year.unique())]
        for peer in peers:
            daily = returns[peer["name"]].iloc[1:]
            rolling = daily.add(1).rolling(252).apply(np.prod, raw=True).dropna().sub(1)
            annual = daily.add(1).groupby(daily.index.year).prod().sub(1)
            lines.append(f"| {peer['name']} | {rolling.gt(0).mean():.1%} | " + " | ".join(f"{v:.2%}" for v in annual) + " |")
        for peer in peers:
            metrics = baseline[peer["name"]]["metrics"]["full"]["base"]
            executed = db.execute("SELECT executed_traded_notional FROM daily WHERE strategy=? AND date>? ORDER BY date",
                                  [peer["name"], source["performance_start"]]).df().iloc[:, 0]
            turnover = compute_turnover_metrics(executed)["annualized_turnover"]
            plot_rows.append({"strategy": peer["name"], **metrics, "volatility": metrics["annual_volatility"], "annualized_turnover": turnover})
    if repaired:
        before_nav = pd.read_parquet(Path(contract["reference"]) / "comparison_nav.parquet")
        comparison, comparison_rows = {}, []
        for peer in peers:
            if not repaired.intersection(peer["directions"]):
                continue
            for suffix, frame, data in (("修复前（含前视）", before_nav[peer["reference_name"]], previous), ("修复后", nav[peer["name"]], baseline)):
                label = peer["name"] + "·" + suffix
                comparison[label] = frame
                metrics = data[peer["name"]]["metrics"]["full"]["base"]
                comparison_rows.append({"strategy": label, **metrics, "volatility": metrics["annual_volatility"]})
        _write_comparison_plot(output, pd.DataFrame(comparison), comparison_rows, cutoff=pd.Timestamp(source["selection_end"]),
            title="新5组因果修复前后：固定成员与全部交易参数\n修复前含前视，仅作影响对照；截止后仅模拟实盘观察")
        (output / "nav_comparison.png").replace(output / "repair_nav_comparison.png")
    nav.to_parquet(output / "comparison_nav.parquet")
    lines += ["", "## 图表", "", "![净值与全期指标](nav_comparison.png)", "", "![板块贡献热力图](sector_contributions.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_comparison_plot(output, nav, plot_rows, cutoff=pd.Timestamp(source["selection_end"]),
        title=f"{len(peers)}组备选策略净值对比" + "：相同配方、相同数据\n2026-05-15之后仅模拟实盘观察；默认配置未变")
    plot_sector_contributions(output, baseline, peers)


def plot_sector_contributions(output, baseline, peers):
    """Render saved contributions only, without rewriting the reviewed report."""
    from monitoring.weekly_report import _mk_png
    from monitoring.plot_style import signed_heatmap_cmap

    plt = _mk_png()
    sectors = list(baseline[peers[0]["name"]]["attribution"]["selection"]["sector"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, period, title in zip(axes, ("selection", "observation"), ("选择期毛贡献（净值百分点）", "模拟实盘毛贡献（净值百分点）")):
        values = np.array([[baseline[p["name"]]["attribution"][period]["sector"][s]*100 for s in sectors] for p in peers])
        bound = max(float(np.max(np.abs(values))), 1e-9)
        im = ax.imshow(values, cmap=signed_heatmap_cmap(), vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(len(sectors)), sectors, rotation=30, ha="right")
        ax.set_yticks(range(len(peers)), [p["name"] for p in peers]); ax.set_title(title, fontsize=11)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="净值百分点（各图独立色标）")
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(output / "sector_contributions.png", dpi=160); plt.close(fig)


def run(reference: Path, output: Path, *, repair_reference: Path | None = None):
    started = time.perf_counter()
    old = read_json(reference / "pool_contract.json")
    repair = read_json(output / "repair_provenance.json") if repair_reference else None
    repaired = set(repair["recomputed_factors"]) if repair else set()
    engineering = read_json(repair["engineering_parity"]) if repair else None
    previous_contract = read_json(repair_reference / "contract.json") if repair else None
    if repair:
        if (Path(repair["baseline_audit"]).resolve() != repair_reference.resolve()
                or previous_contract["source"] != old
                or repair["before_factor_source_sha256"] != old["factor_source_sha256"]
                or repair["after_factor_source_sha256"] != Runner._source_tree_fingerprint()
                or not repair["unchanged_module_ast_exact"] or not repair["old_tree_reconstructed_exact"]
                or not engineering["all_13_exact"]):
            raise ValueError("repair provenance does not match frozen baseline and current code")
    # Neither modify nor silently bypass the completed search's frozen contract.
    for path, expected in old["input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            relative = Path(path).resolve().relative_to(Path.cwd()).as_posix()
            if (not repair or engineering["before_code_sha256"].get(relative) != expected
                    or engineering["after_code_sha256"].get(relative) != hashlib.sha256(Path(path).read_bytes()).hexdigest()):
                raise ValueError(f"reference input changed: {path}")
    if not repair and Runner._source_tree_fingerprint() != old["factor_source_sha256"]:
        raise ValueError("factor computation source changed")
    snapshot = _search_source_snapshot(old["panel_start"], old["end"])
    if snapshot["source_fingerprint"] != old["source_fingerprint"]:
        raise ValueError("market source changed")
    comparisons = read_json(reference / "comparison_metrics.json")
    if len(comparisons) != 13:
        raise ValueError("expected the frozen thirteen-strategy cohort")
    peers = [{"name": r["strategy"] if i < 8 else NAMES[i-8],
              "reference_name": r["strategy"], "directions": dict(sorted(r["directions"].items()))}
             for i, r in enumerate(comparisons)]
    config = load_config("config/default.yaml")
    recipe = _production_recipe(config)
    if json.loads(json.dumps(recipe.to_dict())) != old["recipe"]:
        raise ValueError("recipe changed")
    source_run = Path(old["reference"])
    caches = [reference / "factor_panel_checkpoint", source_run / "factor_panel_checkpoint",
              source_run / "additional_factor_panel_checkpoint"]
    artifacts = [reference / n for n in ("pool_contract.json", "comparison_metrics.json", "comparison_nav.parquet")]
    for cache in caches:
        manifest = read_json(cache / "manifest.json")
        if set(manifest["completed"]) != set(manifest["contract"]["factors"]):
            raise ValueError("incomplete reference cache; no recomputation during audit")
        artifacts += [cache / "manifest.json"] + [cache / f for f in manifest["completed"].values()]
    if repair:
        for path, expected in previous_contract["artifact_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"original artifact changed: {path}")
        artifacts += [repair_reference / n for n in ("contract.json", "baseline_summary.json", "deletion_summary.json", "results.duckdb")]
        artifacts += [output / "repair_provenance.json", Path(repair["engineering_parity"])]
        fixed_cache = output / "factor_panel_checkpoint"
        manifest = read_json(fixed_cache / "manifest.json")
        if set(manifest["completed"]) != repaired:
            raise ValueError("repaired factor panels are incomplete")
        artifacts += [fixed_cache / "manifest.json"] + [fixed_cache / f for f in manifest["completed"].values()]
    contract = {"reference": str(reference.resolve()), "peers": peers, "source": old,
        "artifact_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "clustering": "joint 42-member diagnostic pool; complete linkage abs rank corr .50; cutoff only",
        "experiments": "all single-factor and diagnostic-cluster deletions; no re-selection or promotion",
        "stress": "static twice original daily ledger fees, original exposures unchanged"}
    if repair:
        contract.update(repair_reference=str(repair_reference.resolve()), recomputed_factors=sorted(repaired),
                        current_factor_source_sha256=Runner._source_tree_fingerprint(),
                        current_input_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in old["input_sha256"]})
    output.mkdir(parents=True, exist_ok=True)
    path = output / "contract.json"
    if path.exists() and read_json(path) != contract:
        raise ValueError("audit contract changed")
    dump_json(path, contract)
    with duckdb.connect(str(output / "results.duckdb")) as db:
        db.execute("CREATE TABLE IF NOT EXISTS baselines (name VARCHAR PRIMARY KEY, payload VARCHAR)")
        db.execute("CREATE TABLE IF NOT EXISTS deletions (signature VARCHAR PRIMARY KEY, payload VARCHAR)")
        first_cache = fixed_cache if repair else caches[0]
        first = read_json(first_cache / "manifest.json")["contract"]["factors"]
        runner = Runner(first, start=old["panel_start"], end=old["end"],
                        factor_directions=old["directions"], ic_horizon=1, checkpoint_dir=first_cache)
        if runner.computed_factor_count:
            raise AssertionError("unexpected factor recomputation")
        for cache in (caches if repair else caches[1:]):
            if repair:
                # Explicit, hash-audited reuse of unchanged definitions; never relabel an old checkpoint.
                manifest = read_json(cache / "manifest.json")
                current_contract = runner._checkpoint_contract(manifest["contract"]["factors"])
                expected_contract = {**current_contract, "source_tree_sha256": old["factor_source_sha256"]}
                if manifest["contract"] != expected_contract:
                    raise ValueError("unchanged panel data or axes changed")
                needed = {n for p in peers for n in p["directions"]} - repaired
                raw = {n: pd.read_parquet(cache / f) for n, f in manifest["completed"].items() if n in needed}
                for n, frame in raw.items():
                    if not frame.index.equals(runner.cal) or list(frame.columns) != list(runner.u):
                        raise ValueError(f"unchanged panel axes changed: {n}")
            else:
                raw, _ = runner._load_factor_checkpoint(cache, read_json(cache / "manifest.json")["contract"]["factors"])
            runner.raw_ranks.update({n: f.rank(axis=1, pct=True) for n, f in raw.items()})
        runner.get_contract_schedule()
        runner.env = CausalEligibilityEnvironment(runner.cal, runner.daily_ret, runner.env.sector_of)
        gc.collect()
        cutoff = pd.Timestamp(old["selection_end"])
        names = sorted({n for peer in peers for n in peer["directions"]})
        ranks = {n: runner.raw_ranks[n].loc[old["performance_start"]:cutoff] for n in names}
        corr = _exposure_correlation(ranks, names)
        clusters = _cluster(corr, names)
        dump_json(output / "clusters.json", {"clusters": clusters, "names": names, "correlation": corr.to_numpy().tolist()})
        jobs = deletion_jobs(peers, clusters)
        dump_json(output / "jobs.json", jobs)
        nav_reference = pd.read_parquet(reference / "comparison_nav.parquet")
        for peer in peers:
            if db.execute("SELECT count(*) FROM baselines WHERE name=?", [peer["name"]]).fetchone()[0]:
                continue
            wall, cpu = time.perf_counter(), time.process_time()
            view = runner.for_factors(list(peer["directions"]), factor_directions=peer["directions"])
            evaluator = PortfolioEvaluator(view, start=old["performance_start"], end=old["end"],
                cost_model=_configured_cost_model(config), ic_window=int(config.production_portfolio.ic_window),
                risk_lookback_calendar_days=int(config.production_portfolio.risk_lookback_calendar_days))
            weights = evaluator.weights(list(peer["directions"]), recipe)
            result = build_close_marked_ledger(weights, view.daily_ret.reindex(index=weights.index, columns=weights.columns),
                **evaluator.cost_model.ledger_parameters(), contract_schedule=view.get_contract_schedule(),
                decision_tradable=view.close_tradable, initial_nav=1000.0)
            result.validate()
            daily = result.daily.copy()
            daily["nav"] = daily["nav_after"]
            affected = bool(repaired.intersection(peer["directions"]))
            exact_nav = np.array_equal(daily["nav"].to_numpy()/daily["nav"].iloc[0], nav_reference[peer["reference_name"]].to_numpy())
            if not affected and not exact_nav:
                raise AssertionError(f"unchanged strategy NAV changed: {peer['name']}")
            if repair and not affected:
                with duckdb.connect(str(repair_reference / "results.duckdb"), read_only=True) as previous:
                    expected = previous.execute("SELECT * FROM daily WHERE strategy=? ORDER BY date", [peer["name"]]).df().set_index("date")
                    for col in result.daily:
                        np.testing.assert_array_equal(result.daily[col].to_numpy(), expected[col].to_numpy())
                    assets = previous.execute("SELECT * FROM asset_daily WHERE strategy=?", [peer["name"]]).df()
                    for field, actual in (("effective_weight", result.effective_weights), ("asset_return", result.asset_returns), ("contribution", result.contributions)):
                        expected = assets.pivot(index="date", columns="instrument", values=field).reindex(index=actual.index, columns=actual.columns)
                        np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
            metrics = period_metrics(daily, old["performance_start"], cutoff)
            attribution = {}
            periods = {"selection": daily.loc[:cutoff].iloc[1:], "observation": daily.loc[daily.index > cutoff], "full": daily.iloc[1:]}
            periods.update({str(y): f for y, f in daily.iloc[1:].groupby(daily.index[1:].year)})
            for label, frame in periods.items():
                attribution[label] = attributed_period(frame, result.contributions, result.effective_weights,
                                                       view.env.sector_of, SECTOR_MAP)
            frame = daily.reset_index(names="date").assign(strategy=peer["name"])
            detail = pd.DataFrame({"date": np.repeat(daily.index, len(view.u)), "strategy": peer["name"],
                "instrument": np.tile(view.u, len(daily)), "effective_weight": result.effective_weights.to_numpy().ravel(),
                "asset_return": result.asset_returns.to_numpy().ravel(), "contribution": result.contributions.to_numpy().ravel()})
            payload = {"metrics": metrics, "attribution": attribution, "seconds": time.perf_counter()-wall,
                       "cpu_seconds": time.process_time()-cpu, "exact_nav_match": exact_nav, "repaired_factor_member": affected}
            # Atomic completion includes the full ledger, not just a summary marker.
            db.execute("BEGIN")
            for table, df in (("daily", frame), ("asset_daily", detail)):
                db.register("incoming", df)
                db.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM incoming LIMIT 0")
                db.execute(f"INSERT INTO {table} SELECT * FROM incoming")
                db.unregister("incoming")
            db.execute("INSERT INTO baselines VALUES (?,?)", [peer["name"], json.dumps(payload)])
            db.execute("COMMIT")
            evaluator.clear_transient_caches()
            print(f"baseline {peer['name']} attribution passed; exact previous NAV={exact_nav}", flush=True)
            dump_json(output / "progress.json", {"phase": "baselines", "completed": db.execute("SELECT count(*) FROM baselines").fetchone()[0], "total": 13})
        dump_json(output / "baseline_summary.json", {n: json.loads(p) for n, p in db.execute("SELECT * FROM baselines").fetchall()})
        previous_deletions = read_json(repair_reference / "deletion_summary.json") if repair else {}
        for key, job in jobs.items():
            if db.execute("SELECT count(*) FROM deletions WHERE signature=?", [key]).fetchone()[0]:
                continue
            wall, cpu = time.perf_counter(), time.process_time()
            members = job["directions"]
            if repair and not repaired.intersection(members) and key in previous_deletions:
                previous = previous_deletions[key]
                if previous["directions"] != members:
                    raise ValueError("deletion signature does not match members and directions")
                payload = {k: v for k, v in previous.items() if k not in ("directions", "uses", "seconds", "cpu_seconds")}
                payload.update(seconds=0.0, cpu_seconds=0.0, reused_from=str(repair_reference))
                db.execute("INSERT INTO deletions VALUES (?,?)", [key, json.dumps(payload)])
                continue
            view = runner.for_factors(list(members), factor_directions=members)
            evaluator = PortfolioEvaluator(view, start=old["performance_start"], end=old["end"],
                cost_model=_configured_cost_model(config), ic_window=int(config.production_portfolio.ic_window),
                risk_lookback_calendar_days=int(config.production_portfolio.risk_lookback_calendar_days))
            try:
                daily = evaluator.ledger(list(members), recipe)
                payload = {"status": "evaluated", "metrics": period_metrics(daily, old["performance_start"], cutoff)}
            except (RuntimeError, ValueError) as exc:
                payload = {"status": "rejected_runtime", "error": str(exc)}
            payload.update(seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
            db.execute("INSERT INTO deletions VALUES (?,?)", [key, json.dumps(payload)])
            evaluator.clear_transient_caches()
            count = db.execute("SELECT count(*) FROM deletions").fetchone()[0]
            dump_json(output / "progress.json", {"phase": "deletions", "completed": count, "total": len(jobs)})
            print(f"deletion {count}/{len(jobs)} {payload['status']}", flush=True)
        dump_json(output / "deletion_summary.json", {k: {**jobs[k], **json.loads(p)} for k, p in db.execute("SELECT * FROM deletions").fetchall()})
        dump_json(output / "progress.json", {"phase": "computed", "baselines": 13, "deletions": len(jobs),
            "seconds_this_invocation": time.perf_counter()-started, "report_review_required": True})
    report(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--catalog", type=Path, help="Report only: select current non-archived strategies from frozen reference evidence")
    parser.add_argument("--repair-reference", type=Path, help="Explicit fixed-cohort repair comparison with audited unchanged panels; never used by default")
    args = parser.parse_args()
    if args.report_only:
        report(args.output, source_dir=args.reference if args.catalog else None, catalog_path=args.catalog)
    else:
        if args.catalog:
            parser.error("--catalog requires --report-only")
        run(args.reference, args.output, repair_reference=args.repair_reference)
