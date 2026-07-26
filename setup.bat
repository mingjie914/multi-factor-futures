@echo off
REM ============================================================
REM 多因子框架 — 一键环境初始化
REM ============================================================
REM 使用你的 Python: C:\pythonvenv\Scripts\python.exe
REM ============================================================

cd /d "%~dp0"

echo ╔══════════════════════════════════════════════╗
echo ║    多因子投资框架 — 环境初始化                ║
echo ╚══════════════════════════════════════════════╝
echo.

set PY=C:\pythonvenv\Scripts\python.exe

echo [1/4] 检查 Python 版本...
%PY% --version
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 未找到 C:\pythonvenv\Scripts\python.exe
    echo    请确认 Python 虚拟环境已创建.
    pause
    exit /b 1
)

echo [2/4] 修复 NumPy 版本冲突...
REM 当前环境 numpy=2.0 与 scipy/pandas/matplotlib 不兼容,
REM 降级到 numpy 1.x
%PY% -m pip install "numpy<2" --force-reinstall
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ numpy 降级失败, 继续尝试...
)

echo [3/4] 安装核心依赖...
%PY% -m pip install -r requirements-minimal.txt
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ 部分依赖安装有警告, 继续...
)

echo [4/4] 验证框架导入...
%PY% -c "import sys; sys.path.insert(0, '.'); from core.config import load_config; load_config('config/default.yaml'); print('\n✅ 配置加载 OK')"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ 框架导入失败.
    pause
    exit /b 1
)

%PY% tests\test_import.py
if %ERRORLEVEL% NEQ 0 (
    echo ⚠ 某些可选模块未安装, 不影响基础功能.
)

echo.
echo ╔══════════════════════════════════════════════╗
echo ║   ✅ 环境就绪!                               ║
echo ║                                              ║
echo ║   用以下命令运行:                            ║
echo ║   C:\pythonvenv\Scripts\python.exe main.py  ║
echo ║                                              ║
echo ║   因子检验: --research                       ║
echo ║   完整回测: (不加参数)                       ║
echo ╚══════════════════════════════════════════════╝
echo.
pause
