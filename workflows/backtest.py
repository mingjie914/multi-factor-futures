"""回测脚本 — 独立运行, 无需配置 PyCharm Run 参数.

Usage:
    python main.py backtest
    python main.py backtest --config config/default.yaml
    python main.py backtest --config config/default.yaml --capital 2000000
    python main.py backtest --start 2022-01-01 --end 2024-12-31 --no-plot
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

    parser = argparse.ArgumentParser(description="多因子回测 — 端到端全流程")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="配置文件路径 (默认: config/default.yaml)")
    parser.add_argument(
        "--capital", type=float, default=None,
        help="初始资金 (覆盖配置文件)")
    parser.add_argument(
        "--factors", default=None,
        help="覆盖配置中的因子列表, 逗号分隔 (如: momentum_20d,skewness_20d)")
    parser.add_argument(
        "--start", default=None,
        help="起始日期 (覆盖配置文件)")
    parser.add_argument(
        "--end", default=None,
        help="结束日期 (覆盖配置文件)")
    parser.add_argument(
        "--freq", default=None, choices=["weekly", "monthly"],
        help="调仓频率 (覆盖配置文件)")
    parser.add_argument(
        "--no-plot", action="store_true",
        help="跳过净值图绘制")
    parser.add_argument(
        "--export-signals", action="store_true",
        help="导出信号到 CSV")
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

    # CR-024: 先加载配置, 应用 CLI 覆盖, 再创建 Runner
    from core.config import load_config
    try:
        cfg = load_config(config_path)
        # 应用 CLI 覆盖 (在 Runner 创建前, 确保 Backtester 拿到正确值)
        if args.factors:
            cfg.factors = [f.strip() for f in args.factors.split(",")]
        if args.start:
            cfg.date_range.start = args.start
        if args.end:
            cfg.date_range.end = args.end
        if args.freq:
            cfg.backtest.rebalance_freq = args.freq
        if args.capital:
            cfg.backtest.initial_capital = args.capital
        if args.cache_only:
            cfg.data.cache["only"] = True
        if args.output_dir:
            cfg.backtest.report_dir = args.output_dir
        runner = PipelineRunner(config=cfg)
    except Exception as e:
        print(f"框架初始化失败: {e}")
        print(f"   请检查配置文件: {config_path}")
        sys.exit(1)

    print("=" * 60)
    print("完整回测模式")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  因子数量: {len(runner.config.factors)}")
    print(f"  因子列表: {runner.config.factors}")
    print(f"  日期范围: {runner.config.date_range.start} ~ {runner.config.date_range.end}")
    print(f"  调仓频率: {runner.config.backtest.rebalance_freq}")
    print(f"  初始资金: {runner.config.backtest.initial_capital:,.0f}")

    try:
        result = runner.run_full_pipeline()
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n回测结果: {result.summary()}")

    if not args.no_plot:
        try:
            result.plot(save_dir=runner.config.backtest.report_dir)
            print(f"  净值图 -> {runner.config.backtest.report_dir}/backtest_nav.png")
        except Exception:
            print("  (MATPLOTLIB 未安装: 跳过净值图。运行 python -m pip install matplotlib)")

    if args.export_signals:
        runner.export_signals(result, fmt="csv")
        print(f"  信号明细 -> signals_output/final_signals.csv")

    print("\n回测完成.")


if __name__ == "__main__":
    main()
