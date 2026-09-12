"""Explicit frozen-cohort attribution and deletion diagnostics; no promotion."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

from backtest.research_ledger import build_close_marked_ledger
from core.config import load_config
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
        stressed = net - frame["trade_cost"] - frame["holding_cost"]
        result[label] = {"base": performance_metrics(net), "static_cost_2x": performance_metrics(stressed)}
    segments = [(pd.Timestamp(start), pd.Timestamp("2019-12-31")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
        (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        (pd.Timestamp("2025-01-01"), cutoff)]
    result["robustness"] = robust_summary(daily.loc[:cutoff], segments, initial_anchor=True)
    return result


def report(output):
    """Read completed artifacts only; do not select or mutate the cohort."""
    from run_portfolio_workflow import _write_comparison_plot
    from core.registry import get as registry_get
    import factors.library  # noqa: F401
    contract = read_json(output / "contract.json")
    peers, source = contract["peers"], contract["source"]
    baseline = read_json(output / "baseline_summary.json")
    deletions = read_json(output / "deletion_summary.json")
    clusters = read_json(output / "clusters.json")["clusters"]
    library = {r["factor"]: r for r in read_json("factor_library/library.json")["factors"]}
    repaired = set(contract.get("recomputed_factors", []))
    exact_count = sum(bool(r["exact_nav_match"]) for r in baseline.values())
    lines = ["# 13组固定策略归因与剔除验证报告", "",
        f"基准回测：{source['performance_start']}至{source['end']}。选择截止{source['selection_end']}，随后仅观察。",
        "不修改成员、方向、生产默认、有效库、准入阈值，不新增实盘批准。",
        f"原生账本{exact_count}条净值与原基准逐点精确一致；其余为修复后固定成员对照。贡献按各区间净值链式累加，费用独立列示。",
        "联合42成员诊断K簇不覆盖原搜索簇，也不改变选组过程。簇不是独立利润源。", "",
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
              "## 逐组归因、成员与边际效应", "",
              "贡献单位为区间净值百分点；毛贡献可大于净收益，因为费用独立扣除。下表同时给出选择期和观察期，不把短期表现用于改成员。",
              "逐品种逐年、多空、平均/峰值敞口和毛收益方差份额保存在baseline_summary.json；results.duckdb保存逐日原生明细。", ""]
    for peer in peers:
        name = peer["name"]; row = baseline[name]; attr = row["attribution"]
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
        lines += ["", "| 因子 | 诊断簇 | 方向 | 原始bar | 准入预测期 | 资格快照 | 说明 |", "|---|---|---:|---|---|---|---|"]
        for n, direction in peer["directions"].items():
            cls = registry_get("factor", n); evidence = library.get(n, {})
            status = "修复后待再认证（原资格不沿用）" if n in repaired else evidence.get("status", "库外历史观察")
            lines.append(f"| {n} | K{clusters[n]} | {direction:+d} | {getattr(cls, 'input_bar_frequency', '1min')} | "
                f"{evidence.get('best_period', '未确认')} | {status} | {getattr(cls, 'description', '').replace('|', '/')} |")
        effects = []
        for key, result in deletions.items():
            for use in result["uses"]:
                if use["strategy"] != name:
                    continue
                if result["status"] == "evaluated":
                    metric = result["metrics"]["selection"]["base"]
                    effects.append((metric["sharpe"]-row["metrics"]["selection"]["base"]["sharpe"], use,
                                    metric["annual_return"]-row["metrics"]["selection"]["base"]["annual_return"]))
                else:
                    lines.append(f"\n剔除{use['removed']}运行失败：{result['error']}；未回退参数。")
        effects.sort(key=lambda v: v[0])
        highlighted = effects[:3] + [e for e in effects[-3:] if e not in effects[:3]]
        lines += ["", "剔除后的变化：负值意味着该成员/簇在该搭配中有帮助，正值意味着删除后该指标改善；属于反事实边际效应，不能相加作为利润份额。仅展示选择期改善/损害最大的各3项，全部试验含观察期记录均保留。", "",
                  "| 剔除类型 | 剔除成员 | 夏普变化 | 年化变化 |", "|---|---|---:|---:|"]
        for delta, use, annual in highlighted:
            lines.append(f"| {use['kind']} | {', '.join(use['removed'])} | {delta:+.3f} | {annual:+.2%} |")
        lines.append("")
    total_seconds += sum(r["seconds"] for r in deletions.values())
    failures = sum(r["status"] != "evaluated" for r in deletions.values())
    reused = sum(bool(r.get("reused_from")) for r in deletions.values())
    lines += ["## 验收与限制", "", f"{exact_count}条基准精确复现；{len(deletions)}个去重剔除试验，失败{failures}个，其中{reused}项按相同方向/成员签名复用未变证据。累计实际计时{total_seconds/3600:.2f}小时（不含共享准备和报告）。",
        "每个区间品种贡献+独立费用项与净收益对账；有效仓位及资产收益恒等式由原生ResearchReturnLedger.validate验证。",
        "毛收益方差份额是品种贡献与总毛收益的协方差占比，可负；不是净收益风险份额或独立风险。最大回撤阶段毛贡献不包含独立费用。",
        "剔除试验是固定规则诊断，不按结果自动重新选组。该历史区间参与过研究，不能声称独立样本外；观察期亦已被查看。",
        "成员在连续合约修补后的正式准入再认证尚未完成；不缩小FDR假设族，也不改阈值。实盘滑点、成交延迟、容量仍未模拟。",
        "结果是观察策略的归因与稳健性证据，不自动加入目录、更不自动修改preferred。", ""]
    plot_rows = []
    with duckdb.connect(str(output / "results.duckdb"), read_only=True) as db:
        native = db.execute("SELECT date,strategy,nav,net_return FROM daily ORDER BY date,strategy").df()
        nav = native.pivot(index="date", columns="strategy", values="nav").reindex(columns=[p["name"] for p in peers])
        nav = nav / nav.iloc[0]
        returns = native.pivot(index="date", columns="strategy", values="net_return").reindex(columns=nav.columns)
        corr = returns.loc[:source["selection_end"]].iloc[1:].corr()
        dimension = float(np.trace(corr.to_numpy())**2 / np.square(corr.to_numpy()).sum())
        dump_json(output / "strategy_correlation.json", {"names": list(corr.columns), "selection_correlation": corr.to_numpy().tolist(), "participation_dimension": dimension})
        lines += ["## 收益相关性与近似有效维数", "", f"选择截止前13组日净收益相关矩阵的参与率维数为{dimension:.3f}；是二阶收益分散诊断，不是独立因子数或未来盈利保证。完整矩阵见strategy_correlation.json。", ""]
        for peer in peers:
            metrics = baseline[peer["name"]]["metrics"]["full"]["base"]
            turnover = db.execute("SELECT avg(executed_traded_notional)*252 FROM daily WHERE strategy=? AND date>?",
                                  [peer["name"], source["performance_start"]]).fetchone()[0]
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
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_comparison_plot(output, nav, plot_rows, cutoff=pd.Timestamp(source["selection_end"]),
        title="13组固定策略归因验证" + ("：新5组残差因子修复后" if repaired else "：相同配方、相同数据") + "\n2026-05-15之后仅模拟实盘观察；生产配置未变")
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
    parser.add_argument("--repair-reference", type=Path, help="Explicit fixed-cohort repair comparison with audited unchanged panels; never used by default")
    args = parser.parse_args()
    if args.report_only:
        report(args.output)
    else:
        run(args.reference, args.output, repair_reference=args.repair_reference)
