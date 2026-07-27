"""Pydantic configuration models for the multi-factor framework.

Supports:
- YAML and JSON config files (auto-detected by extension)
- Environment variable overrides for sensitive fields (research → production path)
- ${VAR} and ${VAR:default} inline env var expansion (CR-012: no plaintext secrets)
- Nested validation via pydantic

Research mode: secrets in local.yaml (gitignored) or env vars
Production mode: set MF_MYSQL_HOST, MF_MYSQL_USER, MF_MYSQL_PASSWORD env vars
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

class MySQLTableConfig(StrictConfigModel):
    """Wind → framework field mapping for a single RDS table."""
    table_name: str
    columns: Dict[str, str] = {}
    timestamp_convention: str = "start"
    bar_minutes: int = 0


class MySQLEndpointConfig(StrictConfigModel):
    """One ordered MySQL endpoint. First healthy endpoint becomes active."""
    name: str = ""
    host: str = ""
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = ""
    charset: str = "utf8mb4"


class MySQLConfig(StrictConfigModel):
    """MySQL / RDS connection configuration.

    Research: values from YAML.
    Production: override via MF_MYSQL_HOST / MF_MYSQL_USER / MF_MYSQL_PASSWORD env vars.
    """
    host: str = ""
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = ""
    charset: str = "utf8mb4"
    tables: Dict[str, MySQLTableConfig] = {}
    query_timeout: int = 30
    pool_size: int = 5
    connect_timeout: int = 5
    failure_cooldown: float = 30.0
    dominant_lag_days: int = 1
    schedule_buffer_days: int = 20
    endpoints: List[MySQLEndpointConfig] = []
    fallbacks: List[MySQLEndpointConfig] = []


class DDBConfig(StrictConfigModel):
    """DolphinDB connection configuration (CR-014: 完整配置支持).

    Credentials via ${VAR} env var expansion.
    """
    host: str = ""
    port: int = 8961
    user: str = ""
    password: str = ""
    minute_db: str = "dfs://kline_db"
    minute_table: str = "kline_futures_1min"
    eod_db: str = "dfs://wind_db"
    eod_table: str = "CCommodityFuturesEODPrices"
    data_version: str = "v1"
    dominant_lag_days: int = 1


class ParquetConfig(StrictConfigModel):
    """Local partitioned-Parquet futures data configuration."""

    root_path: str = ""
    datasets: Dict[str, str] = {
        "daily": "futureshistoryprices1d",
        "1min": "futureshistoryprices1m",
        "15min": "futureshistoryprices15m",
    }
    dominant_lag_days: int = 1
    schedule_buffer_days: int = 45
    eager_fields: bool = True
    panel_cache_entries: int = 1
    curve_cache_enabled: bool = True
    curve_cache_path: str = "./cache/curve_aggregates"
    selected_cache_enabled: bool = True
    selected_cache_path: str = "./cache/selected_contracts"


class DataSourceConfig(StrictConfigModel):
    """Data source selection + cache + MySQL params."""
    source: str = "akshare_futures"
    cache: Dict[str, Any] = {}
    mysql: Optional[MySQLConfig] = None
    ddb: Optional[DDBConfig] = None  # CR-014: DDB 配置链路完整支持
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
    # Allow arbitrary extra fields (commission_rate, slippage, margin_rate, ...)
    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"


class SignalModeConfig(StrictConfigModel):
    """Signal generation configuration."""
    mode: str = "trend_following"
    position_sizer: str = "fixed_fraction"
    position_sizer_params: Dict[str, Any] = {}
    sl_tp_rules: List[Dict] = []
    batch: Dict = {}
    output: Dict = {}


class AssetSelectionConfig(StrictConfigModel):
    """Optional sector-aware forecast gate applied before optimization."""
    enabled: bool = False
    mode: str = "hysteresis_top_n"
    top_n_per_side: int = 2
    exit_buffer: int = 1
    min_abs_forecast: float = 0.0
    restrict_to_valid_sectors: bool = False


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
    training_window: int = 504    # 训练窗口 ~2年
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
    - frequency 表示周期单位, 默认 "daily" (1周期=1交易日)
    - 当 frequency="15min" 时, holding_period=5 表示 5 个15分钟bar
    - frequency 仅用于日志显示和未来扩展, 当前回测引擎按日度运行
    """
    name: str = ""
    factors: List[str] = []
    rebalance_freq: str = "weekly"   # daily / weekly / monthly
    holding_period: int = 5          # 持有期 (周期数, 非天数)
    retrain_freq: int = 10           # 重训频率 (交易日)
    training_window: int = 504       # 训练窗口
    capital_weight: float = 0.5      # 资本占比 (0~1, 所有子组合之和应为1.0)
    # 子组合内部约束 (更宽松; 为空时使用默认宽松约束)
    sub_constraints: List[dict] = []
    # 周期单位 (新增, 可选): "daily" / "15min" / "30min" / "hourly"
    # 默认 "daily", 与现有行为完全一致
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
    # 子组合已经扣除各自成本；叠加层会加回这些成本并按聚合底层净交易
    # 重新估算一次，从而允许不同 sleeve 的同品种交易相互抵消。
    net_underlying_costs: bool = True
    underlying_dynamic_risk_limits: DynamicRiskLimitsConfig = DynamicRiskLimitsConfig(
        asset_vol_budget=0.03,
        sector_vol_budget=0.075,
    )


class ConfirmMapEntry(StrictConfigModel):
    """日内确认映射条目 (方案A)."""
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
    required: bool = False
    strict_config_hash: bool = True


class FactorGovernanceConfig(StrictConfigModel):
    """Optional training-only economic-family selection caps."""
    enabled: bool = False
    default_max_per_family: int = 20
    family_caps: Dict[str, int] = {}
    explicit_family_map: Dict[str, str] = {}


class HorizonEnsembleConfig(StrictConfigModel):
    """Optional neighbouring-horizon assignment for walk-forward experiments."""
    enabled: bool = False
    neighbor_count: int = 1
    max_log_distance: float = 0.80
    use_valid_periods: bool = False
    retrain_freq_by_horizon: Dict[str, int] = {}


class DefensiveSleeveConfig(StrictConfigModel):
    """Isolated optional trend/risk-allocation benchmark from index-method ideas."""
    enabled: bool = False
    integration_mode: str = "standalone"
    universe: List[str] = []
    lookbacks: List[int] = [20, 60, 120]
    rebalance_freq: int = 5
    volatility_window: int = 60
    top_n_per_sector: int = 1
    exit_buffer: int = 1
    allocation: str = "inverse_volatility"
    covariance_shrinkage: float = 0.30
    target_volatility: float = 0.10
    asset_cap: float = 0.20
    sector_cap: float = 0.35
    turnover_cap: float = 0.50
    annual_fee: float = 0.001


class SupertrendSleeveConfig(StrictConfigModel):
    """ATR(20, 2) rule sleeve kept in shadow mode until promotion gates pass."""
    enabled: bool = True
    integration_mode: str = "shadow"
    capital_weight: float = 0.0
    rebalance_freq: int = 5
    rebalance_on_flip: bool = False
    target_volatility: float = 0.12
    asset_vol_budget: float = 0.025
    sector_vol_budget: float = 0.06
    hard_asset_cap: float = 1.0
    gross_cap: float = 2.0
    net_cap: float = 0.5
    turnover_cap: float = 0.5


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class FrameworkConfig(StrictConfigModel):
    """Top-level framework configuration.

    Field names match the YAML keys in config/default.yaml:
        market, seed, data, date_range, factors, processing, testing,
        alpha, risk, optimization, costs, signals, backtest
    """
    market: str = "futures"
    seed: int = 42
    data: DataSourceConfig = DataSourceConfig()
    date_range: DateRangeConfig = DateRangeConfig()
    universe: List[str] = []
    factors: List[str] = []
    processing: List[ProcessingStepConfig] = []
    testing: FactorTestConfig = FactorTestConfig()
    alpha: AlphaConfig = AlphaConfig()
    risk: RiskConfig = RiskConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    costs: CostConfig = CostConfig()
    signals: SignalModeConfig = SignalModeConfig()
    universe_selection: UniverseSelectionConfig = UniverseSelectionConfig()
    asset_selection: AssetSelectionConfig = AssetSelectionConfig()
    backtest: BacktestConfig = BacktestConfig()
    sub_portfolios: List[SubPortfolioConfig] = []  # 多频率子组合 (为空时走单组合回测)
    meta_optimizer: MetaOptimizerConfig = MetaOptimizerConfig()  # 子组合资本权重元优化器
    factor_synthesis: FactorSynthesisConfig = FactorSynthesisConfig()  # 因子合成 + 日内确认
    research_artifacts: ResearchArtifactsConfig = ResearchArtifactsConfig()
    factor_governance: FactorGovernanceConfig = FactorGovernanceConfig()
    horizon_ensemble: HorizonEnsembleConfig = HorizonEnsembleConfig()
    defensive_sleeve: DefensiveSleeveConfig = DefensiveSleeveConfig()
    supertrend_sleeve: SupertrendSleeveConfig = SupertrendSleeveConfig()


# ---------------------------------------------------------------------------
# Environment variable override (research → production path)
# ---------------------------------------------------------------------------

_ENV_MAP = {
    # env_var: (config_path_tuple, converter)
    "MF_MARKET": (("market",), str),
    "MF_DATA_SOURCE": (("data", "source"), str),
    "MF_MYSQL_HOST": (("data", "mysql", "host"), str),
    "MF_MYSQL_PORT": (("data", "mysql", "port"), int),
    "MF_MYSQL_USER": (("data", "mysql", "user"), str),
    "MF_MYSQL_PASSWORD": (("data", "mysql", "password"), str),
    "MF_MYSQL_DATABASE": (("data", "mysql", "database"), str),
    "MF_DDB_HOST": (("data", "ddb", "host"), str),
    "MF_DDB_PORT": (("data", "ddb", "port"), int),
    "MF_DDB_USER": (("data", "ddb", "user"), str),
    "MF_DDB_PASSWORD": (("data", "ddb", "password"), str),
    "MF_PARQUET_ROOT": (("data", "parquet", "root_path"), str),
    "MF_BT_FREQ": (("backtest", "rebalance_freq"), str),
    "MF_DATE_START": (("date_range", "start"), str),
    "MF_DATE_END": (("date_range", "end"), str),
}


def _apply_env_overrides(raw: dict) -> dict:
    """Apply environment variable overrides to a raw config dict.

    This is the research → production bridge:
    - Research: all config in YAML (including passwords)
    - Production: set MF_MYSQL_PASSWORD etc. as env vars, YAML values get overridden

    Usage in production:
        export MF_MYSQL_PASSWORD=real_password
        python main.py --config config/default.yaml
    """
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

    Supports environment variable overrides for sensitive fields.
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

    # CR-012: 加载 local.yaml 覆盖 (gitignored, 含本地凭据)
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
