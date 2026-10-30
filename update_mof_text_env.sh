#!/usr/bin/env bash
#
# Install the libraries required by custom-sold (SOLD) into an existing
# multi-object-fetch conda environment, by default "mof-text".
#
# The environment must already contain multi_object_fetch and a working
# mujoco-py, e.g. created with multi-object-fetch-with-text/create_mof_text_env.sh.
#
# Versions are pinned to a combination verified to run all four SOLD scripts
# (generate_dataset, train_autoencoder, train_sold, evaluate_sold). See
# BUGS_AND_PITFALLS.md, items 3a-3c, for why lightning, rich and setuptools
# must not be upgraded.
#
# Usage:
#   ./update_mof_text_env.sh [-n ENV_NAME] [--lfs-pull] [--smoke-test]
#
#   -n, --name NAME   Conda environment to update (default: mof-text)
#   --lfs-pull        Install git-lfs into the env and download the checkpoints
#   --smoke-test      Run all four SOLD scripts at tiny sizes in a temp dir
#                     (needs the checkpoints, so implies --lfs-pull)
#   -h, --help        Show this help

set -euo pipefail

ENV_NAME="mof-text"
LFS_PULL=false
SMOKE_TEST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name) ENV_NAME="$2"; shift 2 ;;
        --lfs-pull) LFS_PULL=true; shift ;;
        --smoke-test) SMOKE_TEST=true; LFS_PULL=true; shift ;;
        -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { echo -e "\n\033[1;34m==> $*\033[0m"; }
die()  { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

# Pinned SOLD dependencies.
PACKAGES=(
    "torch==2.9.0"
    "torchvision==0.24.0"
    "lightning==2.5.2"              # >=2.5.3 crashes OnlineProgressBar (3a)
    "lightning-bolts==0.7.0"
    "pytorch-lightning==1.9.5"      # required by lightning-bolts 0.7.0
    "torchmetrics==1.8.2"
    "rich==14.0.0"                  # >=14.1 crashes lightning 2.5.2 (3b)
    "setuptools<81"                 # >=81 drops pkg_resources (3c)
    "hydra-core==1.3.2"
    "hydra-submitit-launcher==1.2.0"
    "omegaconf==2.3.0"
    "av==12.0.0"
    "moviepy==1.0.3"
    "matplotlib==3.10.7"
    "tensorboard==2.20.0"
    "tensorboardX==2.6.4"
    "termcolor==3.2.0"
    "wandb==0.22.3"
    "pytest==8.4.2"                 # Test suite in tests/ (see tests/README.md)
    "pytest-cov==7.0.0"
    "pytest-timeout==2.4.0"
)

# Packages of the simulator stack that must not change.
CONSTRAINTS=(
    "numpy==1.24.3"
    "gym==0.17.3"
    "cython<3"
    "mujoco-py==2.1.2.14"
)

# ---------------------------------------------------------------- conda
find_conda() {
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then echo "$CONDA_EXE"; return; fi
    if command -v conda >/dev/null 2>&1; then command -v conda; return; fi
    for c in "$(dirname "$REPO_DIR")/miniconda3" /samsung/miniconda3 "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
        if [[ -x "$c/bin/conda" ]]; then echo "$c/bin/conda"; return; fi
    done
    return 1
}
# Resolve the "defaults" channel to conda-forge for every conda call in this
# script. Newer conda refuses Anaconda's own channels until their Terms of
# Service are accepted, and everything needed here is on conda-forge. This only
# affects this script, not the user's conda configuration.
export CONDA_DEFAULT_CHANNELS="${CONDA_DEFAULT_CHANNELS:-https://conda.anaconda.org/conda-forge}"
CONDA_BIN="$(find_conda)" || die "conda not found. Put it on PATH or set CONDA_EXE."
CONDA_BASE="$("$CONDA_BIN" info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

[[ -d "$ENV_PREFIX" ]] || die "Environment '$ENV_NAME' not found at $ENV_PREFIX."
[[ -d "$REPO_DIR/sold" && -d "$REPO_DIR/configs" ]] || die "Run this script from the custom-sold repository."

# conda's activation scripts are not written for `set -u`.
set +u
while [[ -n "${CONDA_PREFIX:-}" && "${CONDA_SHLVL:-0}" -gt 0 ]]; do conda deactivate; done
conda activate "$ENV_NAME"
set -u
[[ "${CONDA_PREFIX:-}" == "$ENV_PREFIX" ]] || die "Failed to activate $ENV_NAME."

log "Environment: $ENV_NAME ($ENV_PREFIX), $(python --version)"

# ------------------------------------------------------ prerequisite check
log "Checking that multi_object_fetch and mujoco-py work"
python - <<'EOF' || die "The environment lacks a working multi_object_fetch / mujoco-py. Create it with create_mof_text_env.sh first."
import warnings
warnings.filterwarnings("ignore")
import mujoco_py, multi_object_fetch
print("multi_object_fetch:", multi_object_fetch.__file__)
EOF

# ------------------------------------------------------------ pip install
CONSTRAINTS_FILE="$(mktemp --suffix=.txt)"
trap 'rm -f "$CONSTRAINTS_FILE"' EXIT
printf "%s\n" "${CONSTRAINTS[@]}" "${PACKAGES[@]}" > "$CONSTRAINTS_FILE"

log "Installing SOLD dependencies"
python -m pip install -c "$CONSTRAINTS_FILE" "${PACKAGES[@]}"

log "Checking installed packages"
python -m pip check
python - <<'EOF'
import warnings
warnings.filterwarnings("ignore")
import importlib
mods = ["torch", "torchvision", "lightning", "pl_bolts", "hydra", "omegaconf", "rich", "termcolor",
        "matplotlib", "tensorboard", "tensorboardX", "av", "moviepy", "wandb", "numpy", "gym", "mujoco_py", "pytest"]
for name in mods:
    module = importlib.import_module(name)
    print(f"  {name:<14} {getattr(module, '__version__', 'ok')}")
import torch
print("  CUDA available:", torch.cuda.is_available())
EOF

# ---------------------------------------------------------- checkpoints
if $LFS_PULL; then
    log "Installing git-lfs and downloading checkpoints"
    if ! git lfs version >/dev/null 2>&1; then
        conda install -y -n "$ENV_NAME" --override-channels -c conda-forge git-lfs
        hash -r
    fi
    git -C "$REPO_DIR" lfs install --local
    git -C "$REPO_DIR" lfs pull
    if head -c 40 "$REPO_DIR/checkpoints/autoencoder/savi/reach_red/latest.ckpt" | grep -q "git-lfs"; then
        die "Checkpoints are still LFS pointer files after git lfs pull."
    fi
    find "$REPO_DIR/checkpoints" -name "*.ckpt" -exec ls -lh {} \;
fi

# ------------------------------------------------------------ smoke test
if $SMOKE_TEST; then
    TMP_DIR="$(mktemp -d)"
    log "Running smoke test in $TMP_DIR"
    cd "$REPO_DIR/sold"
    run() {
        local name="$1"; shift
        echo "--- $name"
        if ! python "$@" > "$TMP_DIR/$name.log" 2>&1; then
            tail -20 "$TMP_DIR/$name.log"
            die "$name failed. Full log: $TMP_DIR/$name.log"
        fi
    }
    run generate_dataset generate_dataset.py experiment=smoke save_format=npz \
        num_train=4 num_val=2 num_test=2 num_workers=2 hydra.run.dir="$TMP_DIR/data/ds"
    run train_autoencoder train_autoencoder.py experiment=smoke \
        dataset.path="$TMP_DIR/data" dataset.data_dir=ds dataset.batch_size=2 dataset.num_workers=0 \
        trainer.max_epochs=1 trainer.check_val_every_n_epoch=1 \
        logger.tensorboard.save_dir="$TMP_DIR/ae" hydra.run.dir="$TMP_DIR/ae_run"
    run train_sold train_sold.py experiment=smoke model.max_steps=300 model.num_seed=100 \
        model.eval_freq=100000 model.num_eval_episodes=1 model.buffer_capacity=20000 model.batch_size=4 \
        model.autoencoder.checkpoint_path="$REPO_DIR/checkpoints/autoencoder/savi/reach_red/latest.ckpt" \
        logger.tensorboard.save_dir="$TMP_DIR/sold" hydra.run.dir="$TMP_DIR/sold_run"
    run evaluate_sold evaluate_sold.py \
        checkpoint_path="$REPO_DIR/checkpoints/sold/reach_red/latest.ckpt" eval_episodes=1 \
        hydra.run.dir="$TMP_DIR/eval_run"
    echo "--- evaluation metrics"
    cat "$TMP_DIR/eval_run/metrics.jsonl"
    rm -rf "$TMP_DIR"
    log "Smoke test passed"
fi

log "Done. To run SOLD:"
echo "    conda activate $ENV_NAME"
echo "    cd $REPO_DIR/sold"
echo "    python train_sold.py"
