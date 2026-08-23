@echo off
rem ============================================================
rem  Saint George Classifier - One-click environment setup (Windows)
rem  Double-click this file to install everything automatically.
rem ============================================================

echo ============================================================
echo  Saint George Classifier - One-click Setup (Windows)
echo ============================================================
echo.

rem ---- Step 1: check Python ----
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ first:
    echo         https://www.python.org/downloads/
    echo         (Remember to check "Add Python to PATH" during install)
    pause
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [ERROR] Python version is too old. Need 3.10 or newer.
    pause
    exit /b 1
)
echo [OK] Python found: 
python --version

echo.

rem ---- Step 2: create virtual environment ----
if exist venv (
    echo [OK] Virtual environment "venv" already exists, skipping creation.
) else (
    echo [1/3] Creating virtual environment "venv" ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

echo.

rem ---- Step 3: install dependencies ----
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

echo [2/3] Upgrading pip ...
python -m pip install --upgrade pip

echo [3/3] Installing PyTorch (CUDA 12.1 build, works on GPU and CPU) ...
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo.
    echo [ERROR] PyTorch install failed. Check your network and try again.
    echo         If you are in China, try:  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pause
    exit /b 1
)

echo Installing remaining dependencies ...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    pause
    exit /b 1
)

echo.

rem ---- Step 4: verify ----
echo Verifying installation ...
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo  Next steps:
echo    To activate the environment every time you open a new terminal:
echo      venv\Scripts\activate
echo.
echo    Run evaluation:
echo      python project/src/eval.py --model_path project/results/best_model.pth
echo.
echo    Start web demo:
echo      python project/src/demo.py
echo ============================================================
pause
