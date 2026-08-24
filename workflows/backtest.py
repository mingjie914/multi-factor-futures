"""回测脚本 — 独立运行, 无需配置 PyCharm Run 参数.

Usage:
    python main.py backtest
    python main.py backtest --config config/default.yaml
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
        print(f"   请安装依赖: python -m pip install -r requirements.txt")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="多因子回测 — 端到端全流程")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="配置文件路径 (默认: config/default.yaml)")
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
        "--target-output", default=None,
        help="将最后一个收盘决策日的目标权重导出到显式 CSV 路径")
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

    try:
        result = runner.run_full_pipeline()
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n回测结果: {result.summary()}")

    result.save(
        runner.config.backtest.report_dir,
        metadata={
            "experiment": "single_portfolio_backtest",
            "config": config_path,
            "start": runner.config.date_range.start,
            "end": runner.config.date_range.end,
        },
    )
    print(f"  结构化回测结果 -> {runner.config.backtest.report_dir}")

    if not args.no_plot:
        try:
            result.plot(save_dir=runner.config.backtest.report_dir)
            print(f"  净值图 -> {runner.config.backtest.report_dir}/backtest_nav.png")
        except ImportError as exc:
            if "matplotlib" not in str(exc).lower():
                raise
            print("  (MATPLOTLIB 未安装: 跳过净值图。运行 python -m pip install matplotlib)")

    if args.target_output:
        exported = result.export_target_weights(args.target_output)
        print(f"  目标权重 -> {exported}")

    print("\n回测完成.")


if __name__ == "__main__":
    main()
