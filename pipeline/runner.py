"""PipelineRunner — 端到端多因子研究与目标权重编排器."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

from core.types import DateIndex, Universe
from core.config import load_config, FrameworkConfig
from core.period import PeriodContext
from core.registry import create, list_registered
from data.manager import DataManager
from factors.engine import FactorEngine
from factors.processor import (
    FactorProcessor,
    build_processing_context,
    build_processing_steps,
)

logger = logging.getLogger(__name__)


def _require_trade_calendar(calendar, start, end) -> pd.DatetimeIndex:
    """Return a real exchange calendar or fail before research/backtesting."""
    dates = pd.DatetimeIndex(calendar)
    if dates.empty:
        raise RuntimeError(f"交易日历为空: {start} ~ {end}")
    return dates


class PipelineRunner:
    """端到端多因子 pipeline 编排器.

    从配置文件构建完整的数据→因子→检验→预测→风险→优化→回测流程.
    已配置组件必须成功构造；正式链路不替换模型或静默跳过约束.

    Usage:
        runner = PipelineRunner("config/default.yaml")
        result = runner.run_full_pipeline()
        print(result.summary())
    """

    def __init__(self, config_path: str = None, config: FrameworkConfig = None,
                 *, frequency: str = "daily"):
        """初始化 PipelineRunner.

        CR-024: 支持直接传入 config 对象, 允许 CLI 覆盖在 Runner 创建前生效.
        Args:
            config_path: 配置文件路径 (与 config 二选一)
            config: 已加载的 FrameworkConfig 对象 (优先于 config_path)
            frequency: 本编排器当前只支持 ``daily``。非日度因子研究使用
                       ``workflows.research`` 的 FrequencyDataProvider 路径。
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config: FrameworkConfig = load_config(config_path)
        else:
            raise ValueError("必须提供 config_path 或 config 参数")
        self.period_ctx = PeriodContext.from_string(frequency)
        if not self.period_ctx.is_daily:
            raise ValueError(
                "PipelineRunner is daily-only; use the frequency-aware research "
                "workflow for intraday bars"
            )
        self._setup_logging()
        self.research_artifacts = None
        self._adaptivity_artifact_df = None
        self._adaptivity_sector_df = None
        self._load_research_artifacts()

        logger.info("正在初始化数据层...")
        self._build_data_layer()

        logger.info("正在初始化因子层...")
        self._build_factor_layer()

        logger.info("正在初始化回测相关组件...")
        self._build_testing_layer()
        self._build_alpha_layer()
        self._build_risk_layer()
        self._build_optimization_layer()
        self._build_backtest_layer()

        registered = {k: list(v.keys()) for k, v in list_registered().items()}
        logger.info(f"初始化完成. 已注册: {registered}")

    # ------------------------------------------------------------------
    # Component construction
    # ------------------------------------------------------------------

    def _setup_logging(self):
        from core.logger import setup_logger
        setup_logger("multi_factor", logging.INFO)

    def _load_research_artifacts(self):
        """Load and validate one immutable point-in-time artifact bundle."""
        artifact_cfg = getattr(self.config, "research_artifacts", None)
        if artifact_cfg is None or not getattr(artifact_cfg, "enabled", False):
            logger.info("研究产物 bundle 未启用；板块筛选与相关性聚类不加载")
            return
        raw_path = str(getattr(artifact_cfg, "path", "") or "").strip()
        if not raw_path:
            raise ValueError("research_artifacts.enabled=true 但 path 为空")

        from research.artifacts import ResearchArtifactBundle, canonical_config_hash

        project_root = Path(__file__).resolve().parents[1]
        bundle_path = Path(raw_path)
        if not bundle_path.is_absolute():
            bundle_path = project_root / bundle_path
        expected_hash = None
        if bool(getattr(artifact_cfg, "strict_config_hash", True)):
            expected_hash = canonical_config_hash(self.config)
        bundle = ResearchArtifactBundle.load(
            bundle_path,
            decision_date=self.config.date_range.start,
            expected_config_hash=expected_hash,
        )
        from core.sectors import taxonomy_sha256
        from research.validation import validation_policy_sha256

        metadata = dict(bundle.manifest.get("metadata", {}) or {})
        expected_governance = {
            "validation_policy_sha256": validation_policy_sha256(
                self.config.validation_policy
            ),
            "taxonomy_sha256": taxonomy_sha256(),
        }
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected_governance.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "research artifact governance hash mismatch; full P0 replay "
                f"required: {mismatches}"
            )

        self.research_artifacts = bundle
        if bundle.has("factor_adaptivity_summary"):
            self._adaptivity_artifact_df = bundle.read_csv(
                "factor_adaptivity_summary", encoding="utf-8-sig"
            )
        if bundle.has("factor_sector_selection"):
            self._adaptivity_sector_df = bundle.read_csv(
                "factor_sector_selection", encoding="utf-8-sig"
            )
        logger.info(
            "研究产物 bundle 已冻结加载: id=%s as_of=%s",
            bundle.artifact_id,
            bundle.as_of_date.date(),
        )

    def _build_data_layer(self):
        self.data_manager = DataManager.from_config(self.config)
        self.cache = self.data_manager.cache
        logger.info(f"数据层: source={self.config.data.source}")

    def _build_factor_layer(self):
        self.factor_engine = FactorEngine(self.data_manager)

        from factors import library as _factor_library  # noqa: F401
        from processing import fillna, neutralize, standardize, winsorize  # noqa: F401

        proc_configs = []
        for s in self.config.processing:
            d = self._to_dict(s) if not isinstance(s, dict) else s
            proc_configs.append(d)
        self.processing_steps = build_processing_steps(proc_configs)
        self.processor = FactorProcessor(self.processing_steps)
        logger.info(f"因子层: {len(self.config.factors)} factors, "
                    f"{len(self.processing_steps)} processing steps")

    def _build_testing_layer(self):
        from testing.ic_test import ICTest
        from testing.layered import LayeredBacktest
        from testing.regression import RegressionTest

        tc = self.config.testing
        ic_cfg = tc.ic if isinstance(tc.ic, dict) else {}
        layered_cfg = tc.layered if isinstance(tc.layered, dict) else {}
        reg_cfg = tc.regression if isinstance(tc.regression, dict) else {}

        # CR-026: 持有期 (前向收益天数), 用于分层年化和回归 HAC 滞后阶数
        forward_period = ic_cfg.get("forward_period", 20)

        self.factor_tests = {
            "ic": ICTest(
                methods=ic_cfg.get("methods", ["pearson", "spearman"]),
                decay_periods=ic_cfg.get("decay_periods", [1, 5, 10, 20]),
                forward_period=forward_period,
            ),
            # CR-026: 传入 holding_period 用于非重叠调仓频率年化
            "layered": LayeredBacktest(
                n_groups=layered_cfg.get("n_groups", 5),
                holding_period=forward_period,
            ),
            # CR-027: 传入 forward_period 用于 Newey-West HAC 滞后阶数
            "regression": RegressionTest(
                weighted=reg_cfg.get("weighted", True),
                forward_period=forward_period,
            ),
        }

    def _build_alpha_layer(self):
        alpha_cfg = self.config.alpha
        params = self._to_dict(alpha_cfg.params) if hasattr(alpha_cfg, 'params') else {}
        from alpha import family, ols  # noqa: F401

        self.alpha_model = create("return_model", alpha_cfg.type, **params)
        logger.info(f"收益模型: {alpha_cfg.type}")

    def _build_risk_layer(self):
        risk_cfg = self.config.risk
        from risk import barra_futures  # noqa: F401

        self.risk_model = create(
            "risk_model", risk_cfg.type,
            style_factors=risk_cfg.style_factors,
            estimation_window=risk_cfg.estimation_window,
            covariance_estimator=risk_cfg.covariance_estimator,
        )
        logger.info(f"风险模型: {risk_cfg.type}")

    def _build_optimization_layer(self):
        opt_cfg = self.config.optimization
        from optimization import (  # noqa: F401
            constraints,
            hierarchical_asset_risk_parity,
            mean_variance,
            risk_budgeting,
        )

        if opt_cfg.type == "mean_variance":
            self.optimizer = create(
                "optimizer", opt_cfg.type,
                risk_aversion=opt_cfg.risk_aversion,
                cost_penalty=opt_cfg.cost_penalty,
            )
        elif opt_cfg.type == "risk_budgeting":
            self.optimizer = create(
                "optimizer", opt_cfg.type,
                cost_penalty=opt_cfg.cost_penalty,
            )
        elif opt_cfg.type == "hierarchical_asset_risk_parity":
            params = self._to_dict(
                opt_cfg.hierarchical_asset_risk_parity
            )
            self.optimizer = create(
                "optimizer", opt_cfg.type, **params
            )
        else:
            raise ValueError(f"unsupported optimizer type: {opt_cfg.type!r}")

        optimizer_role = getattr(
            self.optimizer, "allocation_role", "general_optimizer"
        )
        deployment_status = getattr(
            self.optimizer, "deployment_status", "research_only"
        )
        logger.info(
            "组合优化器: %s | 使用场景: %s | 状态: %s",
            opt_cfg.type,
            optimizer_role,
            deployment_status,
        )
        if deployment_status == "research_only":
            logger.warning(
                "优化器 %s 仅用于研究对照，不属于正式组合路径",
                opt_cfg.type,
            )

        self.constraints = []
        for c in opt_cfg.constraints:
            ctype = c.get("type", "")
            if not ctype:
                raise ValueError("constraint type must not be empty")
            params = {k: v for k, v in c.items() if k != "type"}
            self.constraints.append(create("constraint", ctype, **params))

        self.dynamic_risk_controller = None
        dynamic_cfg = getattr(opt_cfg, "dynamic_risk_limits", None)
        if dynamic_cfg is not None and bool(getattr(dynamic_cfg, "enabled", False)):
            from optimization.risk_limits import VolatilityRiskCapController

            self.dynamic_risk_controller = VolatilityRiskCapController.from_config(
                dynamic_cfg
            )

        costs_cfg = self.config.costs
        costs_params = {k: v for k, v in self._to_dict(costs_cfg).items()
                        if k != "type"}
        costs_params.setdefault("periods_per_year", getattr(
            self, "period_ctx", PeriodContext()
        ).bars_per_year)
        try:
            self.cost_model = create("cost_model", costs_cfg.type, **costs_params)
        except Exception as e:
            raise RuntimeError(
                f"成本模型创建失败，拒绝在无成本模型下继续: {e}"
            ) from e

        logger.info(f"优化器: {opt_cfg.type}, {len(self.constraints)} 约束")

    def _new_asset_selector(self):
        cfg = getattr(self.config, "asset_selection", None)
        if cfg is None or not cfg.enabled:
            return None
        from optimization.asset_selection import SectorForecastSelector

        return SectorForecastSelector(
            mode=cfg.mode,
            top_n_per_side=cfg.top_n_per_side,
            exit_buffer=cfg.exit_buffer,
            min_abs_forecast=cfg.min_abs_forecast,
        )

    def _build_backtest_layer(self):
        bt_cfg = self.config.backtest
        from backtest.engine import Backtester
        self.backtester = Backtester(
            rebalance_freq=bt_cfg.rebalance_freq,
            cost_model=self.cost_model,
            market_name=self.config.market,
        )
        self.backtester.asset_selector = self._new_asset_selector()
        self.backtester.dynamic_risk_controller = self.dynamic_risk_controller
        self.backtester.universe_selection_config = (
            self.config.universe_selection
        )

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    @staticmethod
    def _rebalance_dates_from_calendar(
        calendar: pd.DatetimeIndex, frequency: str
    ) -> pd.DatetimeIndex:
        """Select actual last trading dates instead of calendar week/month ends."""
        dates = pd.DatetimeIndex(calendar).drop_duplicates().sort_values()
        if len(dates) == 0:
            return dates
        frequency = str(frequency).lower()
        if frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError(f"unsupported rebalance frequency: {frequency!r}")
        if frequency == "daily":
            return dates
        period_frequency = "W-FRI" if frequency == "weekly" else "M"
        grouped = pd.Series(dates, index=dates).groupby(
            dates.to_period(period_frequency)
        ).max()
        return pd.DatetimeIndex(grouped.to_numpy()).sort_values()

    def _build_dynamic_universe_schedule(
        self, universe: Universe, dates: DateIndex
    ) -> Dict:
        """构建动态 universe_schedule: 品种从上市日期开始参与计算.

        规则: 若回测起始日期早于品种上市日期, 则该品种从上市日期开始参与.
        每个调仓日的 universe 只包含上市日期 <= 该日期的品种.

        CR-004修复: schedule 的 key 必须是有效的交易日 (与回测循环中 snapped
        后的 rebalance_date 对齐), 避免在 engine.py 中查找失败而静默回退到
        完整 universe. 这里将每个调仓日映射到"不晚于该调仓日的最近交易日",
        与 _prepare_backtest_data 的 snap 逻辑保持一致.

        Args:
            universe: 配置的全部品种池
            dates: 所有调仓日期 (可能是周日/月末, 非交易日)

        Returns:
            dict: {有效交易日: pd.Index(已上市品种)}
        """
        listing_dates = self.data_manager.get_listing_dates(universe)

        if listing_dates is None or listing_dates.empty:
            raise RuntimeError("无法获取品种上市日期，拒绝构造动态品种池")

        listing_aligned = listing_dates.reindex(universe)
        missing_listing = listing_aligned[listing_aligned.isna()].index.tolist()
        if missing_listing:
            raise RuntimeError(
                "以下品种缺少可验证的上市日期: " + ", ".join(map(str, missing_listing))
            )
        listing_vals = pd.to_datetime(listing_aligned).to_numpy(
            dtype="datetime64[ns]"
        )

        # 按上市日期排序，用 searchsorted 实现 O(n log n) 计算
        sort_idx = np.argsort(listing_vals)
        sorted_listing = listing_vals[sort_idx]
        sorted_universe_labels = universe[sort_idx]

        requested_dates = pd.DatetimeIndex(dates)
        if requested_dates.tz is not None:
            requested_dates = requested_dates.tz_localize(None)
        calendar_start = requested_dates.min() - pd.Timedelta(days=10)
        calendar_end = requested_dates.max() + pd.Timedelta(days=10)
        calendar = _require_trade_calendar(
            self.data_manager.get_calendar(calendar_start, calendar_end),
            calendar_start,
            calendar_end,
        )
        if calendar.tz is not None:
            calendar = calendar.tz_localize(None)

        # Map all dates against one calendar read instead of reading it once
        # per rebalance date.
        universe_schedule: Dict[pd.Timestamp, pd.Index] = {}
        for d in requested_dates:
            position = int(calendar.searchsorted(d, side="right")) - 1
            if position < 0:
                position = int(calendar.searchsorted(d, side="left"))
            if position >= len(calendar):
                raise RuntimeError(f"无法将 {d.date()} 映射到有效交易日")
            trade_d = pd.Timestamp(calendar[position])
            # 已上市品种: 上市日 <= trade_d
            cutoff_pos = int(np.searchsorted(
                sorted_listing, np.datetime64(trade_d), side='right'
            ))
            eligible = pd.Index(sorted_universe_labels[:cutoff_pos])
            # 同一 trade_d 上多个原始调仓日合并时, 取并集避免覆盖丢失品种
            if trade_d in universe_schedule:
                universe_schedule[trade_d] = universe_schedule[trade_d].union(eligible)
            else:
                universe_schedule[trade_d] = eligible

        # 日志: 显示起始日过滤情况
        start_ts = requested_dates[0]
        not_yet_listed = [
            r for r in universe
            if pd.Timestamp(listing_aligned[r]) > start_ts
        ]
        if not_yet_listed:
            details = [
                f"{r}({pd.Timestamp(listing_aligned[r]).date()})"
                for r in not_yet_listed
            ]
            logger.info(
                f"动态上市过滤: {len(not_yet_listed)} 个品种在回测起始日 "
                f"{start_ts.date()} 后上市: {', '.join(details)}"
            )
        else:
            logger.info(
                f"全部 {len(universe)} 个品种在回测起始日 {start_ts.date()} 前已上市"
            )

        return universe_schedule

    def run_factor_research(self, dates: DateIndex = None,
                            universe: Universe = None) -> Dict[str, Any]:
        """仅跑因子检验."""
        dr = self.config.date_range

        # 优先使用传入的 universe, 其次使用 config 中的 universe
        if universe is None or len(universe) == 0:
            config_universe = getattr(self.config, 'universe', None)
            if config_universe and len(config_universe) > 0:
                universe = pd.Index(config_universe)
                logger.info(f"使用 config 中的 universe: {len(universe)} 个品种")
            else:
                ref_date = pd.Timestamp(dr.start) if dr else pd.Timestamp.now()
                universe = self.data_manager.get_universe(ref_date)

        if len(universe) == 0:
            raise ValueError("因子研究 universe 为空")

        logger.info(f"步骤1/5: 获取交易日历 {dr.start} ~ {dr.end}")
        if dates is None:
            full_dates = _require_trade_calendar(
                self.data_manager.get_calendar(dr.start, dr.end), dr.start, dr.end
            )
        else:
            full_dates = pd.DatetimeIndex(dates)
            if (
                full_dates.empty
                or full_dates.has_duplicates
                or not full_dates.is_monotonic_increasing
            ):
                raise ValueError("factor-research dates must be non-empty, unique and sorted")
            calendar = _require_trade_calendar(
                self.data_manager.get_calendar(full_dates[0], full_dates[-1]),
                full_dates[0],
                full_dates[-1],
            )
            invalid = full_dates[~full_dates.isin(calendar)]
            if len(invalid):
                raise ValueError(
                    "factor-research dates contain non-trading days: "
                    + ", ".join(str(pd.Timestamp(value).date()) for value in invalid[:5])
                )
        logger.info(f"  交易日历: {len(full_dates)} 天")

        research_dates = self._rebalance_dates_from_calendar(full_dates, "monthly")
        if len(research_dates) < 10:
            logger.info(f"月度日期只有 {len(research_dates)} 个, 改用每20个交易日取一个")
            research_dates = full_dates[::20]

        logger.info(f"步骤2/5: 获取 close 数据并过滤无数据品种")
        close = self.data_manager.get("close", full_dates, universe)
        if not close.empty:
            valid_tickers = close.columns[close.notna().any()].tolist()
            if len(valid_tickers) < len(universe):
                logger.info(f"过滤无数据品种: {len(universe)} → {len(valid_tickers)}")
                universe = pd.Index(valid_tickers)

        # IC 检验: 全交易日截面 + N 日持有期收益
        # 分层/回归: 月度截面 + 同样 N 日持有期收益
        tc = self.config.testing
        ic_cfg = tc.ic if isinstance(tc.ic, dict) else {}
        forward_period = ic_cfg.get("forward_period", 20)

        logger.info(
            f"步骤3/5: 计算因子矩阵 — {len(self.config.factors)} 个因子 × "
            f"{len(universe)} 品种 × {len(full_dates)} 天"
        )
        ctx = build_processing_context(
            self.data_manager,
            full_dates,
            universe,
            self.config.universe_selection,
        )
        factor_names = self.config.factors
        raw = self.factor_engine.compute_factors(factor_names, full_dates, universe, parallel=False)
        logger.info(f"  因子矩阵计算完成: {len(raw)} 个因子")

        logger.info(f"步骤4/5: 因子处理 + 前向收益计算")
        processed = self.processor.process_batch(raw, ctx)
        fwd_ret = self.data_manager.get_forward_returns(full_dates, universe, period=forward_period)
        if ctx.eligibility is not None:
            fwd_ret = fwd_ret.where(ctx.eligibility)
        logger.info(f"  处理完成: {len(processed)} 个因子, 前向收益 shape={fwd_ret.shape}")

        logger.info(f"步骤5/5: IC/分层/回归检验 (持有期={forward_period}日)")
        results = {}
        for idx, (name, matrix) in enumerate(processed.items()):
            if (idx + 1) % 50 == 0:
                logger.info(f"  检验进度: {idx+1}/{len(processed)}")
            fname_results = {}
            for test_name, test in self.factor_tests.items():
                try:
                    if test_name == "ic":
                        fname_results[test_name] = test.run(matrix, fwd_ret, {})
                    else:
                        matrix_subset = matrix.loc[research_dates]
                        fwd_ret_subset = fwd_ret.loc[research_dates]
                        fname_results[test_name] = test.run(matrix_subset, fwd_ret_subset, {})
                except Exception as e:
                    raise RuntimeError(
                        f"因子 {name!r} 的 {test_name!r} 检验失败: {e}"
                    ) from e
            results[name] = fname_results

        # 保存因子筛选结果到 JSON
        try:
            import json, os
            report_dir = self.config.backtest.report_dir
            os.makedirs(report_dir, exist_ok=True)
            screening = {}
            for name, tests in results.items():
                screening[name] = {}
                for test_name, result in tests.items():
                    screening[name][test_name] = result.summary() if hasattr(result, 'summary') else str(result)
            screening_path = os.path.join(report_dir, "factor_screening.json")
            with open(screening_path, "w", encoding="utf-8") as f:
                json.dump(screening, f, ensure_ascii=False, indent=2)
            logger.info(
                "因子筛选结果已保存: %s (%s 个因子)",
                screening_path,
                len(results),
            )
        except Exception as e:
            raise RuntimeError(f"保存因子筛选结果失败: {e}") from e

        logger.info(f"因子研究完成: {len(results)} 个因子")
        return results

    def run_full_pipeline(self, dates: DateIndex = None,
                          universe: Universe = None):
        """端到端全流程."""
        dr = self.config.date_range
        if dates is None:
            calendar = _require_trade_calendar(
                self.data_manager.get_calendar(dr.start, dr.end), dr.start, dr.end
            )
            dates = self._rebalance_dates_from_calendar(
                calendar, self.config.backtest.rebalance_freq
            )
        else:
            dates = pd.DatetimeIndex(dates)
            if dates.has_duplicates or not dates.is_monotonic_increasing:
                raise ValueError("backtest rebalance dates must be unique and sorted")
        if len(dates) == 0:
            raise ValueError("backtest rebalance dates are empty")

        # 优先使用 config 中配置的 universe, 否则从数据源获取
        if universe is None or len(universe) == 0:
            config_universe = getattr(self.config, 'universe', None)
            if config_universe and len(config_universe) > 0:
                universe = pd.Index(config_universe)
            else:
                universe = self.data_manager.get_universe(dates[0]) if len(dates) > 0 else pd.Index([])

        # 构建动态 universe_schedule: 品种从上市日期开始参与计算
        # 规则: 若回测起始日期早于品种上市日期, 则该品种从上市日期开始参与
        universe_schedule = self._build_dynamic_universe_schedule(
            universe, dates
        )
        factor_names = self.config.factors

        if self.backtester is None:
            raise RuntimeError("回测引擎未初始化 (检查依赖)")

        # 传递训练参数给回测引擎
        bt_cfg = self.config.backtest
        if hasattr(bt_cfg, 'training_window'):
            self.backtester.training_window = bt_cfg.training_window
        if hasattr(bt_cfg, 'retrain_freq'):
            self.backtester.retrain_freq = bt_cfg.retrain_freq
        if hasattr(bt_cfg, 'holding_period'):
            self.backtester.holding_period = bt_cfg.holding_period

        return self.backtester.run(
            data_manager=self.data_manager,
            factor_engine=self.factor_engine,
            processor=self.processor,
            factor_names=factor_names,
            universe_schedule=universe_schedule,
            rebalance_dates=dates,
            alpha_model=self.alpha_model,
            risk_model=self.risk_model,
            optimizer=self.optimizer,
            constraints=self.constraints,
        )

    # ------------------------------------------------------------------
    # 多频率子组合叠加
    # ------------------------------------------------------------------

    def run_multi_portfolio(self) -> "MultiPortfolioResult":
        """多频率子组合叠加回测 (支持元优化器动态权重分配).

        两阶段流程:
        1. 各子组合用各自宽松约束独立跑完整回测, 得到日收益率序列
        2. 元优化器用 walk-forward 方式定期重新分配资本权重
        3. 叠加净值 = 累积(Σ w_i(t) × r_i(t))

        约束层级:
        - 子组合内部 (sub_constraints): 宽松, 各子组合满仓运行
        - 元优化器 (整体约束): 通过资本权重分配控制整体风险

        Returns:
            MultiPortfolioResult: 包含各子组合结果和叠加后的组合结果.
        """
        from backtest.engine import MultiPortfolioResult

        sub_configs = self.config.sub_portfolios
        if not sub_configs:
            raise ValueError("sub_portfolios 配置为空, 请先在 config 中配置子组合")
        names = [str(config.name).strip() for config in sub_configs]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("sub-portfolio names must be non-empty and unique")

        n_sub = len(sub_configs)
        total_capital = 1.0
        dr = self.config.date_range
        universe = pd.Index(self.config.universe)
        if universe.empty or universe.has_duplicates:
            raise ValueError("multi-portfolio universe must be non-empty and unique")
        sub_results: List[dict] = []
        sub_returns: Dict[str, pd.Series] = {}
        all_dates = _require_trade_calendar(
            self.data_manager.get_calendar(dr.start, dr.end), dr.start, dr.end
        )

        # 因子合成器: 聚类内等权合成, 降低多重共线性
        # 从 factor_correlation.json 加载聚类映射, 按子组合过滤
        factor_synthesizer = self._build_factor_synthesizer()

        for idx, sp_cfg in enumerate(sub_configs):
            # 读取周期单位 (新增字段, 默认 "daily", 向后兼容)
            sp_frequency = getattr(sp_cfg, "frequency", "daily")
            if PeriodContext.from_string(sp_frequency).is_daily is False:
                raise ValueError(
                    f"sub-portfolio {sp_cfg.name!r} requests {sp_frequency!r}, but "
                    "PipelineRunner multi-portfolio accounting is daily-only"
                )
            logger.info(
                f"\n{'='*60}\n"
                f"子组合 {idx+1}/{n_sub}: {sp_cfg.name}\n"
                f"  因子数: {len(sp_cfg.factors)}, 调仓频率: {sp_cfg.rebalance_freq}, "
                f"持有期: {sp_cfg.holding_period}周期 (频率: {sp_frequency})\n"
                f"{'='*60}"
            )

            # 生成该子组合的调仓日期
            rebalance_dates = self._rebalance_dates_from_calendar(
                all_dates, sp_cfg.rebalance_freq
            )

            # 构建动态 universe_schedule
            universe_schedule = self._build_dynamic_universe_schedule(
                universe, rebalance_dates
            )

            # 构建子组合内部约束 (宽松)
            sub_constraints = self._build_sub_constraints(sp_cfg)

            # 为该子组合创建独立的 backtester 实例
            from backtest.engine import Backtester
            sub_backtester = Backtester(
                rebalance_freq=sp_cfg.rebalance_freq,
                cost_model=self.cost_model,
                market_name=self.config.market,
            )
            sub_backtester.training_window = sp_cfg.training_window
            sub_backtester.retrain_freq = sp_cfg.retrain_freq
            sub_backtester.holding_period = sp_cfg.holding_period
            sub_backtester.forecast_averaging_vintages = max(
                int(getattr(sp_cfg, "forecast_averaging_vintages", 1)), 1
            )
            sub_backtester.asset_selector = self._new_asset_selector()
            sub_backtester.dynamic_risk_controller = self.dynamic_risk_controller
            sub_backtester.universe_selection_config = (
                self.config.universe_selection
            )
            # 注入因子合成器 (按子组合因子过滤, 只合成该子组合内的因子)
            if factor_synthesizer is not None:
                sub_backtester.factor_synthesizer = (
                    factor_synthesizer.for_factors(sp_cfg.factors)
                    if hasattr(factor_synthesizer, 'for_factors')
                    else factor_synthesizer
                )

            # 注入 IC 监控器 (滚动 IC 跟踪 + 自动失效剔除)
            # decay_tolerance=1: 修复 +len(recent) bug 后, 用最激进剔除恢复正则化效果
            from alpha.ic_monitor import ICMonitor
            sub_backtester.ic_monitor = ICMonitor(
                window=60, min_ic=0.02, decay_tolerance=1, reactivation_ic=0.03
            )

            # 为子组合创建独立的 alpha_model（可带分板块因子集）。
            alpha_cfg = self.config.alpha
            alpha_params = (self._to_dict(alpha_cfg.params)
                            if hasattr(alpha_cfg, 'params') else {})
            sub_alpha, _sector_map = self._build_sub_alpha_model(
                sp_cfg, alpha_cfg, alpha_params,
            )
            selection_cfg = getattr(self.config, "asset_selection", None)
            if (
                _sector_map
                and selection_cfg is not None
                and bool(getattr(selection_cfg, "restrict_to_valid_sectors", False))
            ):
                universe_schedule = self._restrict_universe_schedule_to_sectors(
                    universe_schedule, _sector_map.keys()
                )
                if not any(len(instruments) for instruments in universe_schedule.values()):
                    raise RuntimeError(
                        f"子组合 {sp_cfg.name!r} 的确认板块没有可交易品种"
                    )

            # 运行该子组合的回测
            result = sub_backtester.run(
                data_manager=self.data_manager,
                factor_engine=self.factor_engine,
                processor=self.processor,
                factor_names=sp_cfg.factors,
                universe_schedule=universe_schedule,
                rebalance_dates=rebalance_dates,
                alpha_model=sub_alpha,
                risk_model=self.risk_model,
                optimizer=self.optimizer,
                constraints=sub_constraints,
            )

            logger.info(f"子组合 '{sp_cfg.name}' 完成: {result.summary()}")
            sub_results.append({
                "config": sp_cfg,
                "result": result,
            })
            # 提取日收益率用于元优化器
            sub_returns[sp_cfg.name] = result.nav.pct_change(fill_method=None).dropna()

        # === 元优化器: walk-forward 动态资本权重分配 ===
        meta_cfg = self.config.meta_optimizer
        if meta_cfg.enabled and n_sub > 1:
            combined_nav, weight_history = self._meta_optimize_weights(
                sub_returns, sub_results, sub_configs, meta_cfg, total_capital
            )
        else:
            # 不启用元优化器: 用固定 capital_weight
            weights = np.asarray(
                [sp.capital_weight for sp in sub_configs], dtype=float
            )
            if (
                not np.isfinite(weights).all()
                or np.any(weights < 0.0)
                or float(weights.sum()) <= 0.0
            ):
                raise ValueError(
                    "fixed sub-portfolio capital weights must be finite, "
                    "non-negative and have positive mass"
                )
            weights = weights / weights.sum()
            combined_nav, weight_history = self._fixed_weight_combine(
                sub_returns, sub_results, weights, meta_cfg, total_capital
            )

        # 构建叠加结果
        combined = MultiPortfolioResult.combine_with_dynamic_weights(
            sub_results,
            combined_nav,
            weight_history,
            total_capital,
            turnover_history=getattr(self, "_meta_turnover_history", None),
            cost_history=getattr(self, "_meta_cost_history", None),
            underlying_weights_history=getattr(
                self, "_meta_underlying_weights_history", None
            ),
            failure_ledger=getattr(self, "_meta_failure_ledger", None),
            exposure_diagnostics=getattr(
                self, "_meta_exposure_diagnostics", None
            ),
        )
        combined.combined_result.metrics["total_trade_cost"] = float(
            getattr(self, "_meta_trade_cost_history", pd.Series(dtype=float)).sum()
        )
        combined.combined_result.metrics["total_holding_cost"] = float(
            getattr(self, "_meta_holding_cost_history", pd.Series(dtype=float)).sum()
        )
        logger.info(f"\n组合叠加完成: {combined.combined_result.summary()}")
        return combined

    def recombine_multi_portfolio(
        self,
        base_result,
        *,
        method: str = None,
        fixed_weights: Optional[np.ndarray] = None,
    ):
        """Recombine completed sleeves without rerunning factors or optimizers."""
        import copy
        from backtest.engine import MultiPortfolioResult

        sub_configs = self.config.sub_portfolios
        raw = [
            {"config": config, "result": base_result.sub_results[config.name]}
            for config in sub_configs
        ]
        sub_returns = {
            config.name: base_result.sub_results[
                config.name
            ].nav.pct_change(fill_method=None).dropna()
            for config in sub_configs
        }
        meta_cfg = copy.deepcopy(self.config.meta_optimizer)

        if fixed_weights is not None:
            weights = np.asarray(fixed_weights, dtype=float)
            if weights.shape != (len(sub_configs),) or not np.isfinite(weights).all():
                raise ValueError("fixed_weights must be one finite value per sleeve")
            if weights.sum() <= 0:
                raise ValueError("fixed_weights must have positive total capital")
            weights = weights / weights.sum()
            nav, history = self._fixed_weight_combine(
                sub_returns, raw, weights, meta_cfg, 1.0
            )
        else:
            meta_cfg.method = method or meta_cfg.method
            nav, history = self._meta_optimize_weights(
                sub_returns, raw, sub_configs, meta_cfg, 1.0
            )

        result = MultiPortfolioResult.combine_with_dynamic_weights(
            raw,
            nav,
            history,
            1.0,
            turnover_history=getattr(self, "_meta_turnover_history", None),
            cost_history=getattr(self, "_meta_cost_history", None),
            underlying_weights_history=getattr(
                self, "_meta_underlying_weights_history", None
            ),
            failure_ledger=getattr(self, "_meta_failure_ledger", None),
            exposure_diagnostics=getattr(
                self, "_meta_exposure_diagnostics", None
            ),
        )
        result.combined_result.metrics["total_trade_cost"] = float(
            getattr(self, "_meta_trade_cost_history", pd.Series(dtype=float)).sum()
        )
        result.combined_result.metrics["total_holding_cost"] = float(
            getattr(self, "_meta_holding_cost_history", pd.Series(dtype=float)).sum()
        )
        return result

    def _build_sector_factor_map(self, sp_cfg) -> dict:
        """基于适配性研究为子组合生成分板块因子映射.

        从已校验的 ResearchArtifactBundle 加载适配性数据,
        为每个板块筛选该子组合中对该板块有效的因子 (valid_sectors 包含该板块).
        若文件不存在或无数据, 返回空字典 (回退到全部因子, 向后兼容).

        Args:
            sp_cfg: 子组合配置 (含 factors 列表)

        Returns:
            {sector: [factor_name, ...]} 或 {} (无适配性数据时)
        """
        if self._adaptivity_artifact_df is None:
            return {}
        df = self._adaptivity_artifact_df
        if df.empty or "factor" not in df.columns:
            return {}

        sp_factors = set(sp_cfg.factors)
        # 8 板块 (与 SectorGroupedOLSModel._SECTOR_MAP 一致)
        sectors = [
            "ferrous", "nonferrous", "precious", "energy", "agri",
            "stock_index", "bond", "other",
        ]

        sector_map: Dict[str, list] = {}
        sector_period_df = self._adaptivity_sector_df
        for sec in sectors:
            sec_factors = []
            for _, row in df.iterrows():
                fname = row.get("factor", "")
                if not isinstance(fname, str) or fname not in sp_factors:
                    continue
                valid_secs = str(row.get("valid_sectors", ""))
                if sec not in valid_secs.split("|"):
                    continue
                if sector_period_df is not None and not sector_period_df.empty:
                    matches = sector_period_df[
                        (sector_period_df["factor"].astype(str) == fname)
                        & (sector_period_df["sector"].astype(str) == sec)
                    ]
                    routed = [
                        period
                        for _, match in matches.iterrows()
                        for period in self._sector_row_horizons(match)
                    ]
                    if matches.empty or int(sp_cfg.holding_period) not in routed:
                        continue
                sec_factors.append(fname)
            if sec_factors:
                sector_map[sec] = sec_factors

        return sector_map

    def _build_factor_weight_caps(self, sp_cfg) -> dict:
        """Load observation-channel alpha-contribution caps from the bundle."""
        if self._adaptivity_artifact_df is None:
            return {}
        frame = self._adaptivity_artifact_df
        if frame.empty or not {"factor", "weight_cap"}.issubset(frame.columns):
            return {}
        allowed = set(sp_cfg.factors)
        caps = {}
        for _, row in frame.iterrows():
            name = str(row.get("factor", ""))
            if name not in allowed:
                continue
            cap = float(pd.to_numeric(row.get("weight_cap", 1.0), errors="coerce"))
            if np.isfinite(cap) and 0.0 < cap <= 1.0:
                caps[name] = cap
        return caps

    def _sector_row_horizons(self, row) -> list[int]:
        from core.period import approved_horizon_ensemble

        ensemble = getattr(self.config, "horizon_ensemble", None)
        use_valid = bool(
            ensemble is not None and ensemble.enabled and ensemble.use_valid_periods
        )
        return approved_horizon_ensemble(
            row.get("best_period"),
            row.get("valid_periods", "") if use_valid else "",
            enabled=use_valid,
            neighbor_count=(ensemble.neighbor_count if ensemble is not None else 0),
            max_log_distance=(ensemble.max_log_distance if ensemble is not None else 0.0),
        )

    @staticmethod
    def _restrict_universe_schedule_to_sectors(
        universe_schedule: dict, allowed_sectors
    ) -> dict:
        from core.sectors import sector_for

        allowed = {str(sector) for sector in allowed_sectors}
        return {
            date: pd.Index([
                instrument for instrument in instruments
                if sector_for(instrument) in allowed
            ])
            for date, instruments in universe_schedule.items()
        }

    def _period_targets_sleeve(self, period: float, sleeve_name: str) -> bool:
        """Mirror walk-forward horizon routing for sector-specific alpha maps."""
        ranked = sorted(
            self.config.sub_portfolios,
            key=lambda sub: (
                abs(np.log(max(float(period), 1.0) /
                           max(float(sub.holding_period), 1.0))),
                float(sub.holding_period),
                sub.name,
            ),
        )
        ensemble = getattr(self.config, "horizon_ensemble", None)
        targets = ranked[:1]
        if ensemble is not None and ensemble.enabled:
            limit = 1 + max(int(ensemble.neighbor_count), 0)
            targets = [
                sub for sub in ranked[:limit]
                if abs(np.log(max(float(period), 1.0) /
                              max(float(sub.holding_period), 1.0)))
                <= float(ensemble.max_log_distance)
            ] or ranked[:1]
        return sleeve_name in {sub.name for sub in targets}

    def _build_sub_alpha_model(self, sp_cfg, alpha_cfg, alpha_params):
        """为子组合创建独立的 alpha_model（可选分板块因子集）.

        若适配性数据可用且 alpha 类型为 sector_grouped_ols,
        创建带 sector_factor_map 的独立模型实例; 否则回退到全局模型.

        Args:
            sp_cfg: 子组合配置
            alpha_cfg: alpha 配置对象
            alpha_params: alpha 参数字典

        Returns:
            (alpha_model, sector_factor_map) 元组
        """
        sector_factor_map = self._build_sector_factor_map(sp_cfg)
        factor_weight_caps = self._build_factor_weight_caps(sp_cfg)

        if (
            sector_factor_map
            and alpha_cfg.type in {"sector_grouped_ols", "sector_grouped_ridge"}
        ):
            from alpha import ols  # noqa: F401
            sub_params = dict(alpha_params)
            sub_params["sector_factor_map"] = sector_factor_map
            sub_params["factor_weight_caps"] = factor_weight_caps
            sub_alpha = create("return_model", alpha_cfg.type, **sub_params)
            total_factors = sum(len(v) for v in sector_factor_map.values())
            logger.info(
                f"  分板块因子集已加载 "
                f"({len(sector_factor_map)}/{len(sp_cfg.factors)} 板块, "
                f"共 {total_factors} 个板块-因子组合)"
            )
            return sub_alpha, sector_factor_map

        return self.alpha_model, sector_factor_map

    def _build_factor_synthesizer(self):
        """构建因子合成器 (从已校验的研究 bundle 加载聚类映射).

        若 bundle 不含 factor_correlation, 返回 None (不合成).
        若配置中 factor_synthesis.enabled=false, 也返回 None.

        Returns:
            FactorSynthesizer 实例或 None.
        """
        # 检查配置开关 (默认启用)
        synth_cfg = getattr(self.config, 'factor_synthesis', None)
        if synth_cfg is None:
            # 配置中未定义, 默认启用
            enabled = True
        elif isinstance(synth_cfg, dict):
            enabled = synth_cfg.get('enabled', True)
        else:
            enabled = getattr(synth_cfg, 'enabled', True)
        if not enabled:
            logger.info("因子合成已在配置中禁用")
            return None

        from factors.synthesizer import (
            FactorSynthesizer,
            build_cluster_map_from_json,
        )

        if self.research_artifacts is None or not self.research_artifacts.has(
            "factor_correlation"
        ):
            raise RuntimeError(
                "因子合成已启用，但研究 bundle 不含 factor_correlation"
            )
        corr_path = str(self.research_artifacts.path_for("factor_correlation"))

        cluster_map, _standalone, flip_signs = build_cluster_map_from_json(
            corr_path, min_cluster_size=2
        )
        if not cluster_map:
            logger.info("无可用聚类 (所有因子独立), 不执行因子合成")
            return None

        # 日内确认配置从 config.factor_synthesis.confirm_map 读取。
        # pydantic 模型: confirm_map 是 List[ConfirmMapEntry]
        confirm_map = {}
        if isinstance(synth_cfg, dict):
            for item in synth_cfg.get('confirm_map', []):
                target = item.get('target')
                confirm = item.get('confirm')
                weight = float(item.get('weight', 0.3))
                if target and confirm:
                    confirm_map[target] = (confirm, weight)
        elif hasattr(synth_cfg, 'confirm_map'):
            for item in (synth_cfg.confirm_map or []):
                target = getattr(item, 'target', None)
                confirm = getattr(item, 'confirm', None)
                weight = float(getattr(item, 'weight', 0.3))
                if target and confirm:
                    confirm_map[target] = (confirm, weight)

        return FactorSynthesizer(
            cluster_map, flip_signs=flip_signs, confirm_map=confirm_map
        )

    def _build_sub_constraints(self, sp_cfg) -> list:
        """构建子组合内部约束.

        优先使用 sp_cfg.sub_constraints; 为空则使用默认约束:
        - net_exposure [-0.5, 0.5] (净敞口控制, 允许净多/净空/中性)
        - leverage 2.0 (总杠杆上限)
        - dynamic_risk_limits (单品种波动预算和100%名义硬上限, 优化后执行)
        - sector_exposure 0.30 (板块集中度)
        """
        from core.registry import create

        if sp_cfg.sub_constraints:
            constraint_configs = sp_cfg.sub_constraints
        else:
            # 默认约束: 兼顾收益与风险分散
            constraint_configs = [
                {"type": "net_exposure", "lower": -0.5, "upper": 0.5},
                {"type": "leverage", "limit": 2.0},
                {"type": "sector_exposure", "limit": 0.30},
            ]

        constraints = []
        for c in constraint_configs:
            ctype = c.get("type", "")
            if not ctype:
                raise ValueError("子组合约束 type 不能为空")
            params = {k: v for k, v in c.items() if k != "type"}
            constraints.append(create("constraint", ctype, **params))
        return constraints

    def _meta_optimize_weights(
        self,
        sub_returns: Dict[str, pd.Series],
        sub_results: List[dict],
        sub_configs,
        meta_cfg,
        total_capital: float,
    ) -> tuple:
        """元优化器: walk-forward 动态资本权重分配.

        每 reweight_freq 个交易日, 用过去 estimation_window 天的收益
        重新优化各子组合的资本权重.

        优化: 一次性构建 returns_df, 内层循环用 numpy 数组替代 pandas .loc,
        预分配输出数组避免 list.append 开销.

        Returns:
            (combined_nav, weight_history): 叠加净值 + 权重历史 DataFrame
        """
        from optimization.meta_optimizer import MetaOptimizer

        meta_opt = MetaOptimizer(
            method=meta_cfg.method,
            min_weight=meta_cfg.min_weight,
            max_weight=meta_cfg.max_weight,
            target_volatility=meta_cfg.target_volatility,
            estimation_window=meta_cfg.estimation_window,
            covariance_shrinkage=meta_cfg.covariance_shrinkage,
        )

        # 一次性构建完整 DataFrame (避免元优化器每次重复 pd.DataFrame(dict))
        returns_df = pd.DataFrame(sub_returns).sort_index()
        names = [sp.name for sp in sub_configs]
        valid_mask = returns_df[names].notna().any(axis=1)
        returns_df = returns_df.loc[valid_mask]
        all_dates = returns_df.index
        n_days = len(all_dates)
        n_sub = len(sub_configs)

        # 预分配输出数组，避免逐日 list.append。
        desired_weight_arr = np.empty((n_days, n_sub), dtype=np.float64)
        # 初始等权
        current_w = np.ones(n_sub) / n_sub
        capital_scale = 1.0
        last_reweight = -999

        for i in range(n_days):
            # 每 reweight_freq 天重新优化权重
            if i - last_reweight >= meta_cfg.reweight_freq and i >= 20:
                current_w = meta_opt.optimize(
                    returns_df, current_weights=current_w, date=all_dates[i]
                )
                capital_scale = meta_opt.last_capital_scale
                last_reweight = i

            desired_weight_arr[i] = current_w * capital_scale

        combined_nav, weight_df = self._combine_sleeve_path(
            returns_df,
            desired_weight_arr,
            sub_results,
            meta_cfg,
            total_capital,
        )

        logger.info(
            f"元优化器完成: 方法={meta_cfg.method}, "
            f"重权重频率={meta_cfg.reweight_freq}日, "
            f"平均权重={dict(zip(names, weight_df.mean().round(3)))}"
        )
        return combined_nav, weight_df

    def _fixed_weight_combine(
        self,
        sub_returns: Dict[str, pd.Series],
        sub_results: List[dict],
        weights: np.ndarray,
        meta_cfg,
        total_capital: float,
    ) -> tuple:
        """固定资本目标叠加，并在底层暴露违规时执行最小调整.

        处理子组合起始日/结束日不同导致的 NaN:
        - 取并集索引后, 未开始的子组合收益为 NaN
        - 用 0 填充 NaN (未交易 = 0 收益), 然后丢弃全 NaN 的日期
        - 避免 NaN 传播到 combined_nav 导致年化收益计算失败
        """
        returns_df = pd.DataFrame(sub_returns)
        names = list(sub_returns.keys())
        # 丢弃全 NaN 的日期 (所有子组合都未交易)
        valid_mask = returns_df[names].notna().any(axis=1)
        returns_df = returns_df.loc[valid_mask]
        # NaN → 0 (未交易的子组合贡献 0 收益, 不影响已交易子组合的加权)
        desired_weight_arr = np.tile(
            np.asarray(weights, dtype=float), (len(returns_df), 1)
        )
        return self._combine_sleeve_path(
            returns_df,
            desired_weight_arr,
            sub_results,
            meta_cfg,
            total_capital,
        )

    def _build_underlying_covariance_path(self, dates, instruments, config):
        """Precompute point-in-time annual covariances with one market-data read."""
        dates = pd.DatetimeIndex(dates)
        instruments = pd.Index(instruments)
        if len(dates) == 0 or len(instruments) == 0:
            raise ValueError("动态风险协方差需要非空日期和品种")
        window = max(int(getattr(config, "risk_window", 60)), 20)
        shrinkage = float(np.clip(
            getattr(config, "covariance_shrinkage", 0.30), 0.0, 1.0
        ))
        start = dates[0] - pd.Timedelta(days=window * 2)
        history_dates = _require_trade_calendar(
            self.data_manager.get_calendar(start, dates[-1]), start, dates[-1]
        ).sort_values().unique()
        close = self.data_manager.get("close", history_dates, instruments)
        returns, _ = self.data_manager.prepare_close_data(close.reindex(
            index=history_dates, columns=instruments
        ))

        # The strict close-gap policy has already rejected unexplained internal
        # gaps. Remaining NaNs are only pre-listing or first-observation anchors.
        values = np.nan_to_num(
            returns.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        cumulative = np.vstack([
            np.zeros((1, values.shape[1]), dtype=float),
            np.cumsum(values, axis=0),
        ])
        outer = np.einsum("ti,tj->tij", values, values, optimize=True)
        cumulative_outer = np.concatenate([
            np.zeros((1, values.shape[1], values.shape[1]), dtype=float),
            np.cumsum(outer, axis=0),
        ], axis=0)
        history_index = returns.index
        result = []
        for date in dates:
            end = int(history_index.searchsorted(pd.Timestamp(date), side="left"))
            begin = max(0, end - window)
            count = end - begin
            if count < 20:
                result.append(None)
                continue
            total = cumulative[end] - cumulative[begin]
            cross = cumulative_outer[end] - cumulative_outer[begin]
            covariance = (cross - np.outer(total, total) / count) / (count - 1)
            covariance = (covariance + covariance.T) / 2.0
            diagonal = np.maximum(np.diag(covariance), 1e-10)
            covariance = (
                (1.0 - shrinkage) * covariance
                + shrinkage * np.diag(diagonal)
            )
            result.append(covariance * 252.0)
        return result

    def _combine_sleeve_path(
        self,
        returns_df: pd.DataFrame,
        desired_weight_arr: np.ndarray,
        sub_results: List[dict],
        meta_cfg,
        total_capital: float,
        *,
        initial_failures: List[dict] = None,
    ) -> tuple:
        """Build a net aggregate path from audited sleeve returns and positions.

        Each sleeve keeps its own execution and holding costs.  The aggregate
        layer charges only costs caused by changing the sleeve allocation.
        """
        from optimization.meta_optimizer import UnderlyingExposureController
        from core.sectors import SECTOR_MAP

        names = list(returns_df.columns)
        dates = pd.DatetimeIndex(returns_df.index)
        if dates.empty:
            raise ValueError("cannot combine sub-portfolios without return bars")
        start_dates = []
        for item in sub_results:
            nav = getattr(item["result"], "nav", None)
            if nav is None or nav.dropna().empty:
                raise ValueError(
                    f"sub-portfolio {item['config'].name!r} has no NAV anchor"
                )
            nav = nav.dropna().sort_index()
            if nav.index.has_duplicates:
                raise ValueError(
                    f"sub-portfolio {item['config'].name!r} has duplicate NAV dates"
                )
            start_dates.append(pd.Timestamp(nav.index[0]))
        anchor_date = min(start_dates)
        if anchor_date >= dates[0]:
            raise ValueError("sub-portfolio NAV anchor must precede combined returns")
        raw_returns = returns_df[names].replace([np.inf, -np.inf], np.nan)
        for name in names:
            first_valid = raw_returns[name].first_valid_index()
            if first_valid is None:
                raise ValueError(f"sub-portfolio {name!r} has no return observations")
            active = raw_returns.loc[first_valid:, name]
            if active.isna().any():
                first_gap = active.index[active.isna()][0]
                raise ValueError(
                    f"sub-portfolio {name!r} has an internal return gap at "
                    f"{pd.Timestamp(first_gap).date()}"
                )
        # A sleeve contributes zero only before its own first valid return.
        returns_arr = raw_returns.fillna(0.0).to_numpy(dtype=float)
        desired = np.asarray(desired_weight_arr, dtype=float)
        if desired.shape != returns_arr.shape:
            raise ValueError("desired sleeve weights do not align with sleeve returns")

        (
            exposure_cube,
            instruments,
            embedded_trade_costs,
            embedded_holding_costs,
            embedded_turnovers,
        ) = self._build_effective_exposure_cube(sub_results, dates, names)
        constraint_specs = list(getattr(meta_cfg, "underlying_constraints", []) or [])
        if not constraint_specs:
            configured = list(self.config.optimization.constraints)
            constraint_specs = [
                spec for spec in configured
                if str(spec.get("type", "")) in UnderlyingExposureController._SUPPORTED
            ]
            omitted = sorted({
                str(spec.get("type", "")) for spec in configured
                if str(spec.get("type", "")) not in UnderlyingExposureController._SUPPORTED
            })
            if omitted:
                logger.info(
                    "叠加层仅继承可映射约束，以下子组合约束不在此层重复应用: %s",
                    ", ".join(omitted),
                )
        enforce = bool(getattr(meta_cfg, "enforce_underlying_constraints", True))
        controller = UnderlyingExposureController(
            constraint_specs if enforce else [],
            min_weight=float(getattr(meta_cfg, "min_weight", 0.0)),
            max_weight=float(getattr(meta_cfg, "max_weight", 1.0)),
            sector_map=SECTOR_MAP,
        )
        dynamic_controller = None
        dynamic_covariances = None
        dynamic_cfg = getattr(meta_cfg, "underlying_dynamic_risk_limits", None)
        if dynamic_cfg is not None and bool(getattr(dynamic_cfg, "enabled", False)):
            from optimization.risk_limits import VolatilityRiskCapController

            dynamic_controller = VolatilityRiskCapController.from_config(
                dynamic_cfg, sector_map=SECTOR_MAP
            )
            atr_start = dates[0] - pd.Timedelta(
                days=max(int(getattr(dynamic_cfg, "atr_window", 20)) * 2, 40)
            )
            atr_dates = self.data_manager.get_calendar(atr_start, dates[-1])
            atr_dates = _require_trade_calendar(
                atr_dates, atr_start, dates[-1]
            )
            dynamic_controller.prepare_data(
                self.data_manager, atr_dates, instruments
            )
            dynamic_covariances = self._build_underlying_covariance_path(
                dates, instruments, dynamic_cfg
            )
            missing_covariance = [
                str(pd.Timestamp(date).date())
                for date, covariance in zip(dates, dynamic_covariances)
                if covariance is None
            ]
            if missing_covariance:
                raise RuntimeError(
                    "动态风险协方差历史不足: " + ", ".join(missing_covariance[:5])
                )

        n_days, n_sub = returns_arr.shape
        n_assets = len(instruments)
        applied_weights = np.zeros((n_days, n_sub), dtype=float)
        aggregate_weights = np.zeros((n_days, n_assets), dtype=float)
        combined_returns = np.zeros(n_days, dtype=float)
        turnover = np.zeros(n_days, dtype=float)
        meta_turnover = np.zeros(n_days, dtype=float)
        costs = np.zeros(n_days, dtype=float)
        trade_costs = np.zeros(n_days, dtype=float)
        holding_costs = np.zeros(n_days, dtype=float)
        failures = list(initial_failures or [])
        diagnostics: List[dict] = []

        previous_sleeve_weights: Optional[np.ndarray] = None

        for i, date in enumerate(dates):
            matrix = exposure_cube[i]
            requested = desired[i]
            try:
                applied, diag = controller.apply(requested, matrix, instruments)
            except Exception as exc:
                raise RuntimeError(
                    f"底层暴露投影失败 @ {pd.Timestamp(date).date()}: {exc}"
                ) from exc

            if (
                dynamic_controller is not None
                and dynamic_covariances is not None
                and dynamic_covariances[i] is not None
            ):
                scale, dynamic_diag = dynamic_controller.scale_for_aggregate(
                    matrix @ applied,
                    dynamic_covariances[i],
                    instruments,
                    covariance_is_psd=True,
                    annual_volatility=dynamic_controller.annual_volatility_asof(
                        date, instruments
                    ),
                )
                if scale < 1.0:
                    applied = applied * scale
                diag.update(dynamic_diag)
                diag["constraint_adjusted"] = bool(
                    diag.get("constraint_adjusted", False) or scale < 1.0
                )
                diag["applied_capital_scale"] = float(applied.sum())
                rechecked = controller.diagnostics(matrix @ applied, instruments)
                if not rechecked["feasible"]:
                    raise RuntimeError(
                        "dynamic risk reduction violated aggregate constraints: "
                        f"{rechecked['violations']}"
                    )
                for key in (
                    "feasible", "violations", "net_exposure", "gross_exposure",
                    "max_abs_position", "sector_exposure",
                ):
                    diag[key] = rechecked[key]

            aggregate = matrix @ applied
            applied_weights[i] = applied
            aggregate_weights[i] = aggregate
            allocation_only_previous = aggregate
            if previous_sleeve_weights is not None:
                allocation_only_previous = matrix @ previous_sleeve_weights
                meta_turnover[i] = float(
                    np.abs(aggregate - allocation_only_previous).sum()
                )

            embedded_trade_cost = float(applied @ embedded_trade_costs[i])
            embedded_holding_cost = float(applied @ embedded_holding_costs[i])
            embedded_cost = embedded_trade_cost + embedded_holding_cost
            embedded_turnover = float(applied @ embedded_turnovers[i])
            turnover[i] = embedded_turnover + meta_turnover[i]
            net_sleeve_return = float(applied @ returns_arr[i])
            meta_trade_cost = 0.0
            if meta_turnover[i] > 1e-12 and self.cost_model is not None:
                try:
                    meta_trade_cost = float(
                        self.cost_model.estimate_cost(
                            pd.Series(aggregate, index=instruments, dtype=float),
                            pd.Series(
                                allocation_only_previous,
                                index=instruments,
                                dtype=float,
                            ),
                            pd.Timestamp(date),
                        )
                    )
                except Exception as exc:
                    failures.append({
                        "stage": "meta_reallocation_cost_estimation",
                        "date": str(pd.Timestamp(date).date()),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "fallback": "abort_invalid_result",
                    })
                    raise RuntimeError(
                        f"meta reallocation cost failed at {date}: {exc}"
                    ) from exc
                if not np.isfinite(meta_trade_cost) or meta_trade_cost < 0:
                    raise RuntimeError(
                        f"meta reallocation cost is invalid at {date}: {meta_trade_cost}"
                    )
            combined_returns[i] = net_sleeve_return - meta_trade_cost
            costs[i] = embedded_cost + meta_trade_cost
            trade_costs[i] = embedded_trade_cost + meta_trade_cost
            holding_costs[i] = embedded_holding_cost

            diagnostics.append({
                "date": str(pd.Timestamp(date).date()),
                **diag,
                "meta_reallocation_turnover": float(meta_turnover[i]),
                "aggregate_turnover": float(turnover[i]),
                "transaction_cost": float(costs[i]),
                "embedded_sleeve_cost": embedded_cost,
                "meta_reallocation_cost": meta_trade_cost,
            })
            previous_sleeve_weights = applied

        combined_ret_series = pd.Series(combined_returns, index=dates)
        combined_nav = pd.concat([
            pd.Series([float(total_capital)], index=[anchor_date]),
            float(total_capital) * (1.0 + combined_ret_series).cumprod(),
        ])
        weight_df = pd.DataFrame(applied_weights, index=dates, columns=names)
        underlying_df = pd.DataFrame(
            aggregate_weights, index=dates, columns=instruments
        )

        self._meta_turnover_history = pd.Series(turnover, index=dates, name="turnover")
        self._meta_reallocation_turnover_history = pd.Series(
            meta_turnover, index=dates, name="meta_reallocation_turnover"
        )
        self._meta_cost_history = pd.Series(costs, index=dates, name="transaction_cost")
        self._meta_trade_cost_history = pd.Series(
            trade_costs, index=dates, name="trade_cost"
        )
        self._meta_holding_cost_history = pd.Series(
            holding_costs, index=dates, name="holding_cost"
        )
        self._meta_underlying_weights_history = underlying_df
        self._meta_failure_ledger = failures
        self._meta_exposure_diagnostics = diagnostics
        return combined_nav, weight_df

    @staticmethod
    def _build_effective_exposure_cube(
        sub_results: List[dict],
        dates: pd.DatetimeIndex,
        names: List[str],
    ) -> tuple:
        """Build sleeve exposures, costs, and executed turnover from audited ledgers."""
        results_by_name = {
            item["config"].name: item["result"] for item in sub_results
        }
        def recorded_exposure(result) -> pd.DataFrame:
            ledger = getattr(result, "research_ledger", None)
            effective = getattr(ledger, "effective_weights", None)
            if effective is None or effective.empty:
                raise ValueError(
                    "sub-portfolio result has no audited effective-weight ledger"
                )
            return effective

        instruments = sorted({
            str(column)
            for name in names
            for column in recorded_exposure(results_by_name[name]).columns
        })
        n_days, n_assets, n_sub = len(dates), len(instruments), len(names)
        cube = np.zeros((n_days, n_assets, n_sub), dtype=float)
        embedded_trade_costs = np.zeros((n_days, n_sub), dtype=float)
        embedded_holding_costs = np.zeros((n_days, n_sub), dtype=float)
        embedded_turnovers = np.zeros((n_days, n_sub), dtype=float)

        for sleeve_index, name in enumerate(names):
            result = results_by_name[name]
            ledger = getattr(result, "research_ledger", None)
            ledger_weights = getattr(ledger, "effective_weights", None)
            if ledger_weights is None or ledger_weights.empty:
                raise ValueError(
                    f"sub-portfolio {name!r} has no audited effective-weight ledger"
                )
            effective = ledger_weights.copy()
            effective.index = pd.DatetimeIndex(effective.index)
            if effective.index.has_duplicates or effective.columns.has_duplicates:
                raise ValueError(
                    f"sub-portfolio {name!r} ledger has duplicate axes"
                )
            effective = effective.sort_index()
            if not np.isfinite(effective.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"sub-portfolio {name!r} ledger contains NaN/Inf exposures"
                )
            first_exposure = effective.index.min()
            required_dates = dates[dates >= first_exposure]
            missing_dates = required_dates.difference(effective.index)
            if len(missing_dates):
                raise ValueError(
                    f"sub-portfolio {name!r} ledger has an internal date gap at "
                    f"{pd.Timestamp(missing_dates[0]).date()}"
                )
            effective = effective.reindex(
                index=dates, columns=instruments
            ).fillna(0.0)
            cube[:, :, sleeve_index] = effective.to_numpy(dtype=float)

            daily = getattr(ledger, "daily", None)
            required_columns = {
                "trade_cost",
                "holding_cost",
                "executed_traded_notional",
            }
            if daily is None or not required_columns.issubset(daily.columns):
                raise ValueError(
                    f"sub-portfolio {name!r} ledger has no audited cost/turnover fields"
                )
            daily = daily.copy()
            daily.index = pd.DatetimeIndex(daily.index)
            if daily.index.has_duplicates:
                raise ValueError(
                    f"sub-portfolio {name!r} ledger daily rows have duplicate dates"
                )
            daily = daily.sort_index()
            missing_daily_dates = required_dates.difference(daily.index)
            if len(missing_daily_dates):
                raise ValueError(
                    f"sub-portfolio {name!r} ledger daily rows have an internal gap at "
                    f"{pd.Timestamp(missing_daily_dates[0]).date()}"
                )
            fields = daily.loc[:, sorted(required_columns)].apply(
                pd.to_numeric, errors="coerce"
            )
            if (
                not np.isfinite(fields.to_numpy(dtype=float)).all()
                or bool((fields < 0.0).any().any())
            ):
                raise ValueError(
                    f"sub-portfolio {name!r} ledger costs/turnover are invalid"
                )
            fields = fields.reindex(dates).fillna(0.0)
            embedded_trade_costs[:, sleeve_index] = fields["trade_cost"]
            embedded_holding_costs[:, sleeve_index] = fields["holding_cost"]
            embedded_turnovers[:, sleeve_index] = fields[
                "executed_traded_notional"
            ]
        return (
            cube,
            instruments,
            embedded_trade_costs,
            embedded_holding_costs,
            embedded_turnovers,
        )

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(obj: Any) -> dict:
        """pydantic v2 兼容转换."""
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return dict(obj)
