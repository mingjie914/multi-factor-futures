"""Setup script for multi-factor-framework."""
from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="multi-factor-framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0,<2",
        "pydantic>=1.9.0,<3.0.0",
        # CR-031: 补全实际导入依赖 (扫描非标准库 import)
        "pyyaml>=5.4",          # core/config.py: import yaml
        "scipy>=1.7.0,<1.14",   # testing/ic_test.py, testing/robustness.py: from scipy.stats import spearmanr
        "pyarrow>=6.0",         # data/cache.py: parquet 后端 (pandas to_parquet 引擎)
        "sqlalchemy>=1.4.0",    # data/mysql_source.py: from sqlalchemy import create_engine, text
        "pymysql>=1.0.0",       # data/mysql_source.py: mysql+pymysql:// 连接串
        "requests>=2.28.0",     # data/akshare_futures_source.py 等网络请求
        "cvxpy>=1.2.0",         # optimization/ 模块
    ],
    extras_require={
        "public-data": ["akshare>=1.10.0"],
        "ddb": ["dolphindb>=1.30.0"],
        "research": [
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
    },
    python_requires=">=3.9",
    author="multi_factor",
    description="A multi-factor quantitative trading framework",
)
