"""导出有效因子库文档 — 从 strategies/combined.py 读取当前有效因子, 生成 docs/有效因子库.md.

用法:
    python scripts/export_factor_library.py

在因子被加入有效因子库(更新 strategies/combined.py::FACTORS)后重新运行, 保持文档同步.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    from strategies.combined import FACTORS, MANUAL29

    # 因子在 intraday_advanced 的编号 (前 170 号, 见 docs/因子创造方法论与完整参考.md)
    FACTOR_NUMBERS = {
        "intraday_realised_skewness_20d": 4,
        "intraday_jump_intensity_20d": 8,
        "intraday_dtws_20d": 9,
        "intraday_peak_ridge_ratio_20d": 10,
        "intraday_price_peak_count_20d": 13,
        "intraday_drip_stone_20d": 15,
    }

    rows = []
    for i, (name, direction) in enumerate(FACTORS.items(), 1):
        direction_cn = "正向" if direction == 1 else "负向"
        num = FACTOR_NUMBERS.get(name, "K/V")
        rows.append(
            f"| {i} | `{name}` | {direction_cn} | {num} | "
            f"见 `factors/library/intraday.py` 类定义 | 见类 docstring | - |"
        )

    doc = f"""# 有效因子库 (Validated Factor Library)

> 维护: 2026-08-02 | 自动生成: `scripts/export_factor_library.py`
> 定义: 通过 FDR + |IC|/|t| + 分层单调性 + 入池相关性门槛的因子, 用于信号合成

## 当前有效因子 ({len(FACTORS)}个, B3 冠军方案)

**来源**: `strategies/combined.py::FACTORS` | 品种池: manual29 ({len(MANUAL29)}品种) | 调仓: 周度 | 权重: 池内 ERC

| # | 注册名 | 方向 | intraday编号 | 公式来源 | 简要说明 | 簇 |
|---|--------|------|-------------|----------|----------|-----|
{chr(10).join(rows)}

> 编号为 intraday_advanced 内序号 (见 `docs/因子创造方法论与完整参考.md`), "K/V" 表示补充批次因子.
> 公式与详细说明见 `factors/library/intraday.py` 中各因子类的 docstring.

## 因子库更新流程

1. 新因子通过研究检验 (`research --multi-period`) + 相关性门槛 (|corr|<0.5, `--corr-threshold 0.5`)
2. 手工决定加入有效库 (更新 `strategies/combined.py::FACTORS` 及上文 `FACTOR_NUMBERS` 编号)
3. 重新运行: `python scripts/export_factor_library.py`
4. 本文档自动更新
"""

    out = _PROJECT_ROOT / "docs" / "有效因子库.md"
    out.write_text(doc, encoding="utf-8")
    print(f"有效因子库已导出: {out} ({len(FACTORS)} 个因子)")
    for name, direction in FACTORS.items():
        print(f"  {'+' if direction > 0 else '-'} {name}")


if __name__ == "__main__":
    main()
