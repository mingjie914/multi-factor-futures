"""Setup script for multi-factor-framework."""
from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="multi-factor-framework",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0,<4",
        "numpy>=1.24,<3",
        "pydantic>=1.10,<3",
        "pyyaml>=6,<7",
        "scipy>=1.10,<2",
        "pyarrow>=12",
        "cvxpy>=1.3,<2",
    ],
    extras_require={
        "public-data": ["akshare>=1.10.0"],
        "ddb": ["dolphindb>=1.30.0"],
        "mysql": ["sqlalchemy>=1.4.0,<3", "pymysql>=1.0.0,<2"],
        "research": [
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "ta_cn[cn]==0.5.2",
        ],
        "dev": [
            "pytest>=8,<10",
            "sqlalchemy>=1.4.0,<3",
            "pymysql>=1.0.0,<2",
            "ta_cn[cn]==0.5.2",
        ],
    },
    python_requires=">=3.10",
    author="multi_factor",
    description="A multi-factor quantitative trading framework",
)
