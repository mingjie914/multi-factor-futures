@echo off
setlocal

cd /d "%~dp0"
set "MF_VENV=%CD%\.venv"
set "MF_PYTHON=%MF_VENV%\Scripts\python.exe"

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher ^(py.exe^) was not found.
    echo Install Python 3.10 or newer, then run setup.bat again.
    exit /b 1
)

if not exist "%MF_PYTHON%" (
    echo [1/4] Creating .venv...
    py -3 -m venv "%MF_VENV%"
    if errorlevel 1 exit /b 1
) else (
    echo [1/4] Reusing .venv...
)

echo [2/4] Checking Python version...
"%MF_PYTHON%" -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ is required'; print(sys.version)"
if errorlevel 1 exit /b 1

echo [3/4] Installing development dependencies...
"%MF_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%MF_PYTHON%" -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b 1

echo [4/4] Running smoke checks...
"%MF_PYTHON%" -X utf8 -B -c "from core.config import load_config; load_config('config/default.yaml'); load_config('config/parquet_research.yaml'); print('config OK')"
if errorlevel 1 exit /b 1
"%MF_PYTHON%" -X utf8 -B main.py mining dev-smoke --periods 160 --symbols 8 --population 12 --generations 1 --max-candidates 2
if errorlevel 1 exit /b 1

echo.
echo Environment ready.
echo Run: .venv\Scripts\python.exe main.py --help
exit /b 0
