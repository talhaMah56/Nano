#!/bin/bash
# One-time environment setup (idempotent). Run on the LOGIN node inside tmux:
#   tmux new -s cisc662   then   bash cisc662/env/setup_env.sh 2>&1 | tee $CISC662_WORK/logs/setup_env.log
# Network-bound only (downloads); no compilation happens here.
set -euo pipefail
source /u/tmahmood1/project/nanochat/cisc662/env/env.sh

# 1) uv (standalone binary into ~/.local/bin)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

# 2) nanochat's locked deps, CUDA 12.8 torch build (aarch64 wheels exist for torch 2.9.1+cu128)
cd "$CISC662_REPO"
uv sync --frozen --inexact --extra gpu --group dev
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# 3) extra tools for the project, installed without touching uv.lock.
#    NOTE: later `uv sync` calls must use --inexact or these get removed.
uv pip install pandas ninja tabulate

python - <<'EOF'
import torch, triton, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
EOF
echo "SETUP DONE"
