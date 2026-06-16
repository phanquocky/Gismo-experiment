#!/usr/bin/env bash
# Setup script for the baseline DL experiment on a Linux server.
# Run once before launching baseline_dl.py.
# Usage: bash setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo "============================================="
echo "  Baseline DL Experiment — Setup"
echo "============================================="
echo ""

# ── 1. Find or install Python 3.10+ ──────────────────────────────────────────
info "Checking Python version..."

find_python() {
    for cmd in python3.12 python3.11 python3.10 python3; do
        if command -v "$cmd" &>/dev/null; then
            local major minor
            major=$("$cmd" -c "import sys; print(sys.version_info.major)")
            minor=$("$cmd" -c "import sys; print(sys.version_info.minor)")
            if [ "$major" -ge "$PYTHON_MIN_MAJOR" ] && [ "$minor" -ge "$PYTHON_MIN_MINOR" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

if PYTHON=$(find_python); then
    PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    info "Found Python $PY_VER at $(command -v "$PYTHON")"
else
    warn "Python $PYTHON_MIN_MAJOR.$PYTHON_MIN_MINOR+ not found. Attempting to install via apt..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
        PYTHON=python3.11
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y python3.11 python3.11-venv python3.11-devel
        PYTHON=python3.11
    elif command -v yum &>/dev/null; then
        sudo yum install -y python3 python3-venv python3-devel
        PYTHON=python3
    else
        error "Cannot install Python automatically. Please install Python 3.10+ manually."
    fi
fi

# ── 2. Create virtual environment ─────────────────────────────────────────────
info "Setting up virtual environment at $VENV_DIR ..."

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists — skipping creation."
else
    "$PYTHON" -m venv "$VENV_DIR"
    info "Virtual environment created."
fi

PY="$VENV_DIR/bin/python3"
PIP="$VENV_DIR/bin/pip"

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
info "Upgrading pip..."
"$PY" -m ensurepip --upgrade --quiet 2>/dev/null || true
"$PIP" install --upgrade pip --quiet

# ── 4. Install Python dependencies ───────────────────────────────────────────
info "Installing Python packages..."
"$PIP" install numpy --quiet

# ── 5. Verify all imports work ────────────────────────────────────────────────
info "Verifying installation..."
"$PY" - <<'PYCHECK'
import sys, importlib, multiprocessing, resource, threading, json

required = ["numpy"]
for pkg in required:
    mod = importlib.import_module(pkg)
    ver = getattr(mod, "__version__", "?")
    print(f"  {pkg:<15} {ver}")

print(f"  {'python':<15} {sys.version.split()[0]}")
print(f"  {'cpu_count':<15} {multiprocessing.cpu_count()}")

# Verify resource limits are settable
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
print(f"  {'RLIMIT_AS':<15} soft={soft}  hard={hard}")

print("\nAll checks passed.")
PYCHECK

# ── 6. Create datasets directory if missing ───────────────────────────────────
DATASETS_DIR="$SCRIPT_DIR/../datasets"
if [ ! -d "$DATASETS_DIR" ]; then
    mkdir -p "$DATASETS_DIR"
    warn "datasets/ directory was missing — created at $DATASETS_DIR"
    warn "Place your .txt or .mtx graph files there before running."
else
    N_GRAPHS=$(find "$DATASETS_DIR" -maxdepth 1 \( -name "*.txt" -o -name "*.mtx" \) | wc -l)
    info "datasets/ found — $N_GRAPHS graph file(s) ready."
fi

# ── 7. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Setup complete."
echo "============================================="
echo ""
echo "  Python : $("$PY" --version)"
echo "  Venv   : $VENV_DIR"
echo ""
echo "  To run the experiment:"
echo "    $PY $SCRIPT_DIR/baseline_dl.py"
echo ""
echo "  Output files:"
echo "    $SCRIPT_DIR/Using_DL_Disjunct_records.json   (checkpoint)"
echo "    $SCRIPT_DIR/Using_DL_Disjunct_result.txt     (raw results)"
echo "    $SCRIPT_DIR/Using_DL_Disjunct_report.txt     (summary table)"
echo ""
