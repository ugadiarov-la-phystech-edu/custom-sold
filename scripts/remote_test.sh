#!/usr/bin/env bash
#
# Run the test suite on a remote GPU machine without touching its working copy.
#
# Copies the tracked and untracked (non-ignored) files of this clone, except checkpoints/ and paper/, to a scratch
# directory on the remote machine, links the remote repository's checkpoints/ into it (the golden tests need the real
# LFS files), and runs pytest there at low priority so running trainings are not slowed down.
#
# Usage:
#   scripts/remote_test.sh [PYTEST_ARGS...]        e.g. scripts/remote_test.sh -q --cov
#
# Settings come from the environment or from scripts/remote_test.env (gitignored), for example:
#   REMOTE_HOST=user@gpu-host           # required
#   REMOTE_PORT=2222                    # default 22
#   REMOTE_KEY=/path/to/private_key     # optional, as seen from the jump host (or locally without a jump host)
#   REMOTE_JUMP=user@jump-host          # optional: run the ssh to REMOTE_HOST from this host
#   REMOTE_REPO=/path/to/custom-sold    # remote clone whose checkpoints/ is linked
#   REMOTE_DIR=/tmp/sold-test           # scratch directory, recreated on every run
#   REMOTE_CONDA_SH=/path/to/conda.sh   # sourced before activating REMOTE_CONDA_ENV
#   REMOTE_CONDA_ENV=mof-text
#   REMOTE_THREADS=4                    # OMP/MKL threads for the tests

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[[ -f "$REPO_DIR/scripts/remote_test.env" ]] && source "$REPO_DIR/scripts/remote_test.env"

: "${REMOTE_HOST:?set REMOTE_HOST (see the header of $0)}"
: "${REMOTE_REPO:?set REMOTE_REPO (see the header of $0)}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/sold-test}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-mof-text}"
REMOTE_THREADS="${REMOTE_THREADS:-4}"

# LogLevel=ERROR hides OpenSSH's post-quantum key exchange warning.
ssh_opts=(-o BatchMode=yes -o LogLevel=ERROR -o ConnectTimeout=20 -p "$REMOTE_PORT")
[[ -n "${REMOTE_KEY:-}" ]] && ssh_opts+=(-i "$REMOTE_KEY")

# Run a command on REMOTE_HOST (via REMOTE_JUMP if set), passing stdin through.
remote() {
    if [[ -n "${REMOTE_JUMP:-}" ]]; then
        local command
        printf -v command '%q ' ssh "${ssh_opts[@]}" "$REMOTE_HOST" "$1"
        ssh -o BatchMode=yes -o LogLevel=ERROR "$REMOTE_JUMP" "$command"
    else
        ssh "${ssh_opts[@]}" "$REMOTE_HOST" "$1"
    fi
}

# ---------------------------------------------------------------------------- copy the clone
cd "$REPO_DIR"
git ls-files -z --cached --others --exclude-standard | grep -z -v -E '^(checkpoints|paper)/' \
    | tar --null -czf - -T - \
    | remote "rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR && tar xzf - -C $REMOTE_DIR \
              && ln -s $REMOTE_REPO/checkpoints $REMOTE_DIR/checkpoints"

# ------------------------------------------------------------------------------- run pytest
pytest_args=""
[[ $# -gt 0 ]] && printf -v pytest_args ' %q' "$@"

# Upload the runner as a file: run with `bash -s`, its commands could read its remaining lines from stdin.
remote "cat > $REMOTE_DIR/.run_tests.sh && bash $REMOTE_DIR/.run_tests.sh < /dev/null" <<EOF
set -e
${REMOTE_CONDA_SH:+source $REMOTE_CONDA_SH}
conda activate $REMOTE_CONDA_ENV
cd $REMOTE_DIR
OMP_NUM_THREADS=$REMOTE_THREADS MKL_NUM_THREADS=$REMOTE_THREADS nice -n 10 python -m pytest -p no:cacheprovider $pytest_args
EOF
