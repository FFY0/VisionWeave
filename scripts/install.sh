#!/usr/bin/env bash
# Run inside an activated Python 3.12 virtual environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python - <<'PY'
import platform
import sys
if sys.version_info[:2] != (3, 12) or sys.prefix == sys.base_prefix:
    raise SystemExit("Activate a Python 3.12 virtual environment first.")
if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("This dependency profile targets Linux x86_64.")
PY
python -m pip install --no-compile -r "$ROOT/requirements-lock.txt"
# Runtime patches target this exact revision. Avoid its CUDA 13 dependency defaults.
SGLANG_BUILD_RUST_EXTS=none python -m pip install --no-compile --no-deps --no-build-isolation \
  'sglang @ git+https://github.com/FFY0/sglang.git@df2f34cca1f0b3200c6c0b2e2411916b1f0909c8#subdirectory=python'
python -m pip install --no-compile --no-deps --no-build-isolation -e "$ROOT"
python -m visionweave.check
