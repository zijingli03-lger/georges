#!/usr/bin/env bash
# ============================================================
#  Saint George Classifier - One-click environment setup (Linux / macOS)
#  Run:  bash setup.sh
# ============================================================
set -e

echo "============================================================"
echo " Saint George Classifier - One-click Setup (Linux/macOS)"
echo "============================================================"

# ---- Step 1: check python3 ----
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found. Please install Python 3.10+ first."
    exit 1
fi
echo "[OK] Python: $(python3 --version)"

# ---- Step 2: create virtual environment ----
if [ -d "venv" ]; then
    echo "[OK] Virtual environment 'venv' already exists, skipping creation."
else
    echo "[1/3] Creating virtual environment 'venv' ..."
    python3 -m venv venv
fi

source venv/bin/activate

# ---- Step 3: install dependencies ----
echo "[2/3] Upgrading pip ..."
python -m pip install --upgrade pip

echo "[3/3] Installing PyTorch (CUDA 12.1 build, works on GPU and CPU) ..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
echo "Installing remaining dependencies ..."
pip install -r requirements.txt

# ---- Step 4: verify ----
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Activate the environment next time with:  source venv/bin/activate"
echo "  Evaluate:    python project/src/eval.py --model_path project/results/best_model.pth"
echo "  Web demo:    python project/src/demo.py"
echo "============================================================"
