#!/usr/bin/env bash
#
# All checks: lint, formatting, types, and tests with coverage. Run in the mof-text env; extra arguments go to pytest.
#
#   scripts/check.sh            # Everything
#   scripts/check.sh -x -q      # Stop the tests at the first failure, quietly

set -euxo pipefail  # -x echoes each check.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ruff check .
ruff format --check .
mypy
python -m pytest --cov "$@"
