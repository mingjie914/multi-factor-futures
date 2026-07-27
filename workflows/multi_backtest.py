"""多频率子组合叠加回测脚本 — 独立运行, 无需配置 PyCharm Run 参数.

每个子组合独立因子集 + 独立调仓频率 + 独立持有期, 净值按资本占比叠加.
需在 config 中配置 sub_portfolios 字段.

Usage:
    # 使用 config 中的 sub_portfolios 配置
    python main.py multi

    # 自定义配置文件
    python main.py multi --config config/default.yaml

    # 自定义日期范围
    python main.py multi --start 2022-01-01 --end 2024-12-31

    # 跳过净值图绘制
    python main.py multi --no-plot
"""
from __future__ import annotations
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    import argparse

    try:
        from core.logger import setup_logger
        from pipeline.runner import PipelineRunner
    except ImportError as e:
        print(f"框架模块导入失败: {e}")
        print(f"   请安装依赖: python -m pip install -r requirements-minimal.txt")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="多频率子组合叠加回测")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="配置文件路径 (默认: config/default.yaml)")
    parser.add_argument(
        "--start", default=None,
        help="起始日期 (覆盖配置文件)")
    parser.add_argument(
        "--end", default=None,
        help="结束日期 (覆盖配置文件)")
    parser.add_argument(
        "--no-plot", action="store_true",
        help="跳过净值图绘制")
    parser.add_argument(
        "--cache-only", action="store_true",
        help="严格离线模式；缓存未命中时不访问数据库")
    parser.add_argument(
        "--output-dir", default=None,
        help="报告输出目录 (覆盖配置文件)")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)
    config_path = os.path.normpath(config_path)

    setup_logger("multi_factor")

    try:
        from core.config import load_config

        cfg = load_config(config_path)
        if args.start:
            cfg.date_range.start = args.start
        if args.end:
            cfg.date_range.end = args.end
        if args.cache_only:
            cfg.data.cache["only"] = True
        if args.output_dir:
            cfg.backtest.report_dir = args.output_dir
        runner = PipelineRunner(config=cfg)
    except Exception as e:
        print(f"框架初始化失败: {e}")
        print(f"   请检查配置文件: {config_path}")
        sys.exit(1)

    # 检查 sub_portfolios 配置
    if not runner.config.sub_portfolios:
        print("错误: config 中未配置 sub_portfolios")
        print("   请在 config/default.yaml 中添加 sub_portfolios 字段,")
        print("   每个子组合需指定 name/factors/rebalance_freq/holding_period/capital_weight")
        sys.exit(1)

    print("=" * 60)
    print("多频率子组合叠加回测模式")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  日期范围: {runner.config.date_range.start} ~ {runner.config.date_range.end}")
    print(f"  子组合数: {len(runner.config.sub_portfolios)}")
    for sp in runner.config.sub_portfolios:
        sp_freq = getattr(sp, "frequency", "daily")
        print(f"    [{sp.name}] {sp.rebalance_freq}/{sp.holding_period}周期 "
              f"权重={sp.capital_weight:.0%} 因子数={len(sp.factors)} "
              f"(频率: {sp_freq})")

    try:
        result = runner.run_multi_portfolio()
    except Exception as e:
        print(f"多组合回测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{result.summary()}")

    output_dir = runner.config.backtest.report_dir
    result.save(
        output_dir,
        metadata={
            "experiment": "multi_portfolio_backtest",
            "config": config_path,
            "start": runner.config.date_range.start,
            "end": runner.config.date_range.end,
            "cache_only": bool(args.cache_only),
        },
    )
    print(f"  结构化回测结果 -> {output_dir}")

    if not args.no_plot:
        try:
            # 从配置推断版本号 (用于图表标题)
            version_tag = ""
            try:
                version_tag = runner.config.version or "current"
            except AttributeError:
                version_tag = "current"
            result.plot(save_dir=runner.config.backtest.report_dir, version=version_tag)
            print(f"  子组合净值图 -> {runner.config.backtest.report_dir}/multi_portfolio_nav.png")
        except Exception:
            print("  (MATPLOTLIB 未安装: 跳过净值图。运行 python -m pip install matplotlib)")

    print("\n多组合回测完成.")


if __name__ == "__main__":
    main()
