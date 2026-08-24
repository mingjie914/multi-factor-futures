"""Inspect or update the structured effective-factor library."""
from __future__ import annotations

import argparse
from pathlib import Path

from core.config import load_config
from research.effective_factor_library import (
    admit_validation_run,
    effective_factor_names,
    export_current_csv,
    load_library,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="有效因子库管理")
    parser.add_argument("--config", default="config/default.yaml")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check", help="校验并报告当前有效因子库")
    sub.add_parser("export", help="重新生成 current.csv")
    admit = sub.add_parser("admit", help="从已完成的标准检验run显式入库")
    admit.add_argument("--run-dir", required=True)
    admit.add_argument("--admitted-at", required=True, help="ISO日期")
    args = parser.parse_args()

    config = load_config(args.config)
    path = Path(config.factor_library.path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path = path.resolve()

    if args.action == "admit":
        payload = admit_validation_run(
            args.run_dir, path, admitted_at=args.admitted_at
        )
        print(f"有效因子库已更新: {path}")
        print(f"当前有效因子: {len(effective_factor_names(path))}")
        print(f"来源run: {payload['source_run']}")
    elif args.action == "export":
        print(f"已导出: {export_current_csv(path)}")
    else:
        payload = load_library(path)
        print(f"有效因子库: {path}")
        print(f"记录数: {len(payload['factors'])}")
        print(f"当前有效: {len(effective_factor_names(path))}")
