"""Pydantic configuration models for the multi-factor framework.

Supports:
- YAML and JSON config files (auto-detected by extension)
- Environment variable overrides for local runtime paths
- ${VAR} and ${VAR:default} inline env var expansion
- Nested validation via pydantic

Current research mode: published local Parquet configured in local.yaml or env vars.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel

try:
    from pydantic import ConfigDict
except ImportError:  # Pydantic 1.x compatibility
    ConfigDict = None

_PYDANTIC_V2 = hasattr(BaseModel, "model_validate")


class StrictConfigModel(BaseModel):
    """Reject misspelled or obsolete configuration keys by default."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:
            extra = "forbid"


# ${VAR} 或 ${VAR:default} 模式 (CR-012: 环境变量引用)
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _expand_env_vars(value: Any) -> Any:
    """递归展开配置中的 ${VAR} 和 ${VAR:default} 引用.

    CR-012: 凭据不再明文存储在 YAML, 改为环境变量引用.
    若环境变量未设置且无默认值, 返回空字符串 (生产环境应设置所有必需变量).
    """
    if isinstance(value, str):
        def _replacer(m: re.Match) -> str:
            var_name = m.group(1)
            default = m.group(2)  # None if no :default part
            env_val = os.environ.get(var_name)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            return ""  # 未设置且无默认值 → 空字符串
        return _ENV_VAR_PATTERN.sub(_replacer, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Sub-config models
# ---------------------------------------------------------------------------

class ParquetConfig(StrictConfigModel):
    """Local partitioned-Parquet futures data configuration."""

    root_path: str = ""
    datasets: Dict[str, str] = {
        "daily": "futureshistoryprices1d",
        "1min": "futureshistoryprices1m",
        "5min": "futureshistoryprices5m",
        "15min": "futureshistoryprices15m",
    }
    seat_dataset: str = "futuresseatdata"
    dominant_lag_days: int = 1
    schedule_buffer_days: int = 45
    # Latest economically comparable contract epoch by root.  This is for
    # exchange-defined relaunches/specification breaks, not ordinary listings.
    root_active_from: Dict[str, str] = {}
    eager_fields: bool = True
    panel_cache_entries: int = 1
    curve_cache_enabled: bool = True
    curve_cache_path: str = "./cache/curve_aggregates"
    selected_cache_enabled: bool = True
    selected_cache_path: str = "./cache/selected_contracts"


class DataSourceConfig(StrictConfigModel):
    """Published local-Parquet selection, quality marks, and cache settings."""
    source: str = "parquet_futures"
    cache: Dict[str, Any] = {}
    audited_nontrading_closes: Dict[str, List[str]] = {}
    parquet: Optional[ParquetConfig] = None


class DateRangeConfig(StrictConfigModel):
    """Backtest / analysis date range."""
    start: str = "2018-01-01"
    end: str = "2024-12-31"


class ProcessingStepConfig(StrictConfigModel):
    """Single processing step (winsorize / standardize / neutralize / fillna)."""
    type: str
    params: Dict[str, Any] = {}


class FactorTestConfig(StrictConfigModel):
    """Factor testing parameters."""
    ic: Dict = {}
    layered: Dict = {}
    regression: Dict = {}


class AlphaConfig(StrictConfigModel):
    """Return prediction model configuration."""
    type: str = "ols"
    params: Dict[str, Any] = {}


class RiskConfig(StrictConfigModel):
    """Risk model configuration."""
    type: str = "barra_futures"
    style_factors: List[str] = []
    estimation_window: int = 252
    covariance_estimator: str = "shrinkage"


class DynamicRiskLimitsConfig(StrictConfigModel):
    """Volatility-aware limits applied after portfolio allocation."""
    enabled: bool = False
    asset_vol_budget: float = 0.025
    sector_vol_budget: float = 0.06
    hard_asset_cap: float = 1.0
    gross_cap: float = 2.0
    net_cap: float = 0.5
    atr_window: int = 20
    risk_window: int = 60
    covariance_shrinkage: float = 0.30


class HierarchicalAssetRiskParityConfig(StrictConfigModel):
    """Parameters used only by the three-layer futures risk allocator."""

    target_volatility: float = 0.10
    max_leverage: float = 2.0
    covariance_shrinkage: float = 0.30
    periods_per_year: float = 252.0
    volatility_floor: float = 0.01
    asset_class_budgets: Dict[str, float] = {}
    commodity_sector_budgets: Dict[str, float] = {}


class OptimizationConfig(StrictConfigModel):
    """Portfolio optimizer routing and type-specific configuration.

    The formal default is the three-layer futures allocator. ``risk_aversion``
    and ``cost_penalty`` are retained only for research-only forecast-utility
    challengers such as mean-variance. The formal allocator reads only
    ``hierarchical_asset_risk_parity``, preventing parameter leakage between
    the two scenarios.
    """

    type: str = "hierarchical_asset_risk_parity"
    risk_aversion: float = 2.0
    cost_penalty: float = 0.5
    hierarchical_asset_risk_parity: HierarchicalAssetRiskParityConfig = (
        HierarchicalAssetRiskParityConfig()
    )
    constraints: List[Dict] = []
    dynamic_risk_limits: DynamicRiskLimitsConfig = DynamicRiskLimitsConfig()


class CostConfig(BaseModel):
    """Transaction cost model."""
    type: str = "simple_futures"
    # Cost models have different constructor contracts, so their parameters
    # remain model-specific extras and are passed through by PipelineRunner.
    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"


class AssetSelectionConfig(StrictConfigModel):
    """Optional sector-aware forecast gate applied before optimization."""
    enabled: bool = False
    mode: str = "hysteresis_top_n"
    top_n_per_side: int = 2
    exit_buffer: int = 1
    min_abs_forecast: float = 0.0
    restrict_to_valid_sectors: bool = False


class ProductionPortfolioConfig(StrictConfigModel):
    """Frozen production-construction parameters, expressed per sleeve."""

    factor_weight_method: str = "lw_abs"
    ic_window: int = 60
    top_n_per_side: int = 10
    sector_count_cap: int = 3
    asset_weight_method: str = "erc"
    asset_min_fraction: float = 0.005
    asset_max_fraction: float = 0.20
    asset_max_overrides: Dict[str, float] = {}
    sector_weight_caps: Dict[str, float] = {}
    gross_exposure: float = 2.0
    risk_lookback_calendar_days: int = 90
    minimum_risk_observations: int = 10
    covariance_shrinkage: float = 0.30


class UniverseSelectionConfig(StrictConfigModel):
    """Point-in-time, sector-balanced liquidity universe used before research."""

    enabled: bool = False
    mode: str = "lagged_liquidity_sector_balanced"
    lookback: int = 60
    rebalance_freq: str = "monthly"
    target_count: int = 32
    min_count: int = 28
    max_count: int = 35
    min_listing_days: int = 120
    min_data_coverage: float = 0.95
    score_weights: Dict[str, float] = {"amount": 0.7, "oi": 0.3}
    exit_buffer: int = 4
    sector_minimums: Dict[str, int] = {}
    sector_maximums: Dict[str, int] = {}


class BacktestConfig(StrictConfigModel):
    """Backtest engine configuration."""
    rebalance_freq: str = "weekly"
    benchmark: str = "csi300"
    plot: bool = True
    report_dir: str = "./reports"
    training_window: int = 750    # 日频正式训练窗口下限 ~3年
    retrain_freq: int = 10        # 重训频率 (交易日)
    holding_period: int = 5       # 持有期 (周期数, 非天数; daily频率下1周期=1交易日)


class SubPortfolioConfig(StrictConfigModel):
    """子组合配置 — 用于多频率子组合叠加.

    每个子组合有独立的因子集、调仓频率、持有期和资金占比.
    最终组合净值 = Σ(资本占比_i × 子组合净值_i).

    约束层级:
    - 子组合内部约束 (sub_constraints): 更宽松, 允许各子组合满仓运行
    - 整体组合约束 (optimization.constraints): 严格, 通过元优化器分配资本权重实现

    周期架构说明:
    - holding_period 是"周期数" (bar数), 不是天数
    - PipelineRunner 当前只接受 frequency="daily" (1周期=1交易日)
    - 非日度研究走 FrequencyDataProvider 专用工作流，避免只改年化参数却仍读取日线
    """
    name: str = ""
    factors: List[str] = []
    rebalance_freq: str = "weekly"   # daily / weekly / monthly
    holding_period: int = 5          # 持有期 (周期数, 非天数)
    retrain_freq: int = 10           # 重训频率 (交易日)
    training_window: int = 750       # 日频正式训练窗口下限
    capital_weight: float = 0.5      # 资本占比 (0~1, 所有子组合之和应为1.0)
    # 子组合内部约束 (更宽松; 为空时使用默认宽松约束)
    sub_constraints: List[dict] = []
    # 当前多子组合账本只支持 daily；保留字段用于显式拒绝误配。
    frequency: str = "daily"
    # Optional staggered forecast vintages; 1 preserves existing behavior.
    forecast_averaging_vintages: int = 1


class MetaOptimizerConfig(StrictConfigModel):
    """元优化器配置 — 优化各子组合的资本配置权重.

    基于各子组合的历史收益和协方差矩阵, 优化资本配置权重,
    目标=最大化整体夏普比率. 整体组合的风险约束在此层执行.
    """
    enabled: bool = True             # 是否启用元优化器 (False 则用固定 capital_weight)
    method: str = "shrinkage_min_variance"
    reweight_freq: int = 20          # 重新优化权重的频率 (交易日)
    min_weight: float = 0.1          # 单子组合最小权重 (避免某子组合权重为0)
    max_weight: float = 0.6          # 单子组合最大权重 (避免过度集中)
    target_volatility: float = 0.10  # 整体组合目标年化波动率 (用于缩放)
    estimation_window: int = 252     # 估计协方差矩阵的窗口 (交易日)
    covariance_shrinkage: float = 0.30
    # 在 sleeve 叠加后按真实底层品种暴露再次校验整体约束。为空时继承
    # optimization.constraints 中可映射到叠加层的约束。
    enforce_underlying_constraints: bool = True
    underlying_constraints: List[Dict] = []
    underlying_dynamic_risk_limits: DynamicRiskLimitsConfig = DynamicRiskLimitsConfig(
        asset_vol_budget=0.03,
        sector_vol_budget=0.075,
    )


class ConfirmMapEntry(StrictConfigModel):
    """日内确认映射条目."""
    target: str = ""          # 被确认的合成因子名
    confirm: str = ""         # 确认因子名
    weight: float = 0.3       # 确认强度


class FactorSynthesisConfig(StrictConfigModel):
    """因子合成配置 — 聚类合成 + 日内确认."""
    enabled: bool = True
    confirm_map: List[ConfirmMapEntry] = []


class ResearchArtifactsConfig(StrictConfigModel):
    """Point-in-time research artifact bundle used by the trading pipeline."""
    enabled: bool = False
    path: str = ""
    strict_config_hash: bool = True


class FactorGovernanceConfig(StrictConfigModel):
    """Optional training-only economic-family selection caps."""
    enabled: bool = False
    default_max_per_family: int = 20
    family_caps: Dict[str, int] = {}
    explicit_family_map: Dict[str, str] = {}


class ValidationScorecardConfig(StrictConfigModel):
    """Frozen post-discovery scorecard configuration.

    A scorecard may be reported before calibration, but it cannot become a
    promotion gate until an isolated pilot artifact and its SHA-256 digest are
    declared.  This prevents current candidates from calibrating their own
    admission rule.
    """

    enabled: bool = True
    enforced: bool = False
    calibrated: bool = False
    calibration_source: str = ""
    calibration_sha256: str = ""
    weights: Dict[str, float] = {
        "annual_direction": 0.25,
        "annual_effect": 0.25,
        "hit_rate": 0.25,
        "ir_stability": 0.25,
    }
    ir_std_max: float = 0.30
    threshold: float = 0.75


class ValidationPolicyConfig(StrictConfigModel):
    """Versioned discovery, validation and observation-channel policy."""

    version: str = "factor_validation_v2"
    discovery_method: str = "hierarchical_fdr"
    discovery_q: float = 0.10
    fwer_report_alpha: float = 0.05
    min_abs_ic: float = 0.01
    min_abs_t: float = 2.0
    deployment_hit_rate: float = 0.52
    deployment_oos_hit_rate: float = 0.50
    oos_fold_sign_ratio: float = 0.60
    annual_direction_ratio: float = 0.60
    annual_effect_ratio: float = 0.65
    minimum_calendar_years: int = 5
    intraday_minimum_calendar_years: int = 1
    minimum_year_observations: int = 20
    minimum_train_bars_by_frequency: Dict[str, int] = {
        "daily": 750,
        "1min": 14400,
        "5min": 2880,
        "15min": 960,
        "30min": 480,
        "hourly": 240,
    }
    minimum_test_bars_by_frequency: Dict[str, int] = {
        "daily": 250,
        "1min": 4800,
        "5min": 960,
        "15min": 320,
        "30min": 160,
        "hourly": 80,
    }
    minimum_train_days_by_frequency: Dict[str, int] = {
        "daily": 750, "daily_intraday": 63,
        "1min": 60, "5min": 60, "15min": 60, "30min": 60, "hourly": 60,
    }
    minimum_test_days_by_frequency: Dict[str, int] = {
        "daily": 250, "daily_intraday": 42,
        "1min": 20, "5min": 20, "15min": 20, "30min": 20, "hourly": 20,
    }
    minimum_train_test_ratio: float = 3.0
    n_return_groups: int = 3
    wf_train_bars_by_frequency: Dict[str, int] = {
        "daily": 500, "daily_intraday": 126, "1min": 6000,
    }
    wf_test_bars_by_frequency: Dict[str, int] = {
        "daily": 125, "daily_intraday": 42, "1min": 2000,
    }
    wf_step_bars_by_frequency: Dict[str, int] = {
        "daily": 125, "daily_intraday": 42, "1min": 2000,
    }
    warmup_days_by_frequency: Dict[str, int] = {
        "daily": 252, "daily_intraday": 90, "1min": 60,
    }
    monthly_turnover_reference: float = 0.50
    cost_safety_margin: float = 1.50
    single_instrument_min_trading_days: int = 750
    single_instrument_bootstrap_samples: int = 399
    observation_weight_cap: float = 0.50
    require_predeclared_direction_for_promotion: bool = True
    expected_directions: Dict[str, int] = {}
    dual_track_families: List[str] = ["carry", "trend", "macro_trend"]
    family_horizons: Dict[str, List[int]] = {}
    scorecard: ValidationScorecardConfig = ValidationScorecardConfig()


class HorizonEnsembleConfig(StrictConfigModel):
    """Optional neighbouring-horizon assignment for walk-forward experiments."""
    enabled: bool = False
    neighbor_count: int = 1
    max_log_distance: float = 0.80
    use_valid_periods: bool = False
    retrain_freq_by_horizon: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class FrameworkConfig(StrictConfigModel):
    """Top-level framework configuration.

    Field names match the YAML keys in config/default.yaml:
        market, seed, data, date_range, factors, validated_candidates,
        processing, testing,
        alpha, risk, optimization, costs, backtest
    """
    market: str = "futures"
    seed: int = 42
    data: DataSourceConfig = DataSourceConfig()
    date_range: DateRangeConfig = DateRangeConfig()
    universe: List[str] = []
    factors: List[str] = []
    # Audited research watchlist only; runtime workflows never promote these
    # names into ``factors`` automatically.
    validated_candidates: List[str] = []
    processing: List[ProcessingStepConfig] = []
    testing: FactorTestConfig = FactorTestConfig()
    alpha: AlphaConfig = AlphaConfig()
    risk: RiskConfig = RiskConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    costs: CostConfig = CostConfig()
    universe_selection: UniverseSelectionConfig = UniverseSelectionConfig()
    asset_selection: AssetSelectionConfig = AssetSelectionConfig()
    production_portfolio: ProductionPortfolioConfig = ProductionPortfolioConfig()
    backtest: BacktestConfig = BacktestConfig()
    sub_portfolios: List[SubPortfolioConfig] = []  # 多频率子组合 (为空时走单组合回测)
    meta_optimizer: MetaOptimizerConfig = MetaOptimizerConfig()  # 子组合资本权重元优化器
    factor_synthesis: FactorSynthesisConfig = FactorSynthesisConfig()  # 因子合成 + 日内确认
    research_artifacts: ResearchArtifactsConfig = ResearchArtifactsConfig()
    factor_governance: FactorGovernanceConfig = FactorGovernanceConfig()
    validation_policy: ValidationPolicyConfig = ValidationPolicyConfig()
    horizon_ensemble: HorizonEnsembleConfig = HorizonEnsembleConfig()


# ---------------------------------------------------------------------------
# Environment variable override (research → production path)
# ---------------------------------------------------------------------------

_ENV_MAP = {
    # env_var: (config_path_tuple, converter)
    "MF_MARKET": (("market",), str),
    "MF_DATA_SOURCE": (("data", "source"), str),
    "MF_PARQUET_ROOT": (("data", "parquet", "root_path"), str),
    "MF_BT_FREQ": (("backtest", "rebalance_freq"), str),
    "MF_DATE_START": (("date_range", "start"), str),
    "MF_DATE_END": (("date_range", "end"), str),
}


def _apply_env_overrides(raw: dict) -> dict:
    """Apply documented local-runtime overrides to a raw config dict."""
    for env_key, (path, converter) in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        try:
            val = converter(val)
        except (ValueError, TypeError):
            continue
        # Walk into nested dict
        d = raw
        for key in path[:-1]:
            if key not in d or d[key] is None:
                d[key] = {}
            d = d[key]
        d[path[-1]] = val
    return raw


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> FrameworkConfig:
    """Load framework configuration from a YAML or JSON file.

    Supports documented environment variable overrides such as
    ``MF_PARQUET_ROOT``.
    See _ENV_MAP for available override keys.

    Args:
        path: Path to config file (.yaml, .yml, or .json)

    Returns:
        Validated FrameworkConfig instance.
    """
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")):
            raw = yaml.safe_load(fh)
        else:
            raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a dict at top level, got {type(raw)}")

    extends = raw.pop("extends", None)
    if extends:
        base_path = str(extends)
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)
        with open(base_path, "r", encoding="utf-8") as fh:
            if base_path.endswith((".yaml", ".yml")):
                base_raw = yaml.safe_load(fh)
            else:
                base_raw = json.load(fh)
        if not isinstance(base_raw, dict):
            raise ValueError(
                f"Extended config must contain a dict, got {type(base_raw)}"
            )
        if "extends" in base_raw:
            raise ValueError("Nested config extends is not supported")
        raw = _deep_merge(base_raw, raw)

    # CR-012: 展开 ${VAR} 和 ${VAR:default} 环境变量引用
    raw = _expand_env_vars(raw)

    # Load optional machine-local path overrides (gitignored).
    local_path = os.path.join(os.path.dirname(path), "local.yaml")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as fh:
            local_raw = yaml.safe_load(fh) or {}
        if isinstance(local_raw, dict):
            raw = _deep_merge(raw, local_raw)
            raw = _expand_env_vars(raw)  # local.yaml 中也可能有 ${VAR}

    # Apply env var overrides (production path)
    raw = _apply_env_overrides(raw)

    return FrameworkConfig(**raw)


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base (override 优先)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
