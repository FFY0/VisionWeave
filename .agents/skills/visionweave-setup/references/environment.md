# Environment setup and validation

[Skill](../SKILL.md) · [Project README](../../../../README.md)

Use this reference when installing or repairing the environment. Run commands from the repository root. For a requested inference task, continue with [checkpoints and inference](inference.md) after validation.

[Reference environment](#reference-environment) · [Installation contract](#repository-and-installation-contract) · [Setup](#configure-the-environment) · [Validation](#verify-the-installed-source-and-runtime) · [Pip/network diagnostics](#installation-diagnostics) · [Runtime failures](#resolve-setup-failures-within-the-pinned-profile) · [Optional isolation](#optional-checkout-only-isolation)

## Reference environment

**This is a reference configuration.** Users must configure CUDA and native dependencies for their own GPU architecture, NVIDIA driver, operating system, and compiler, and build compatible components from source where needed. The pinned versions do not guarantee that prebuilt wheels or attention kernels will work on every machine.

The installer targets Linux x86_64, Python 3.12, and CUDA 12.9 wheels; Ubuntu 24.04 with Python 3.12.3 is the host reference. [requirements.txt](../../../../requirements.txt) declares Python dependencies, [requirements-lock.txt](../../../../requirements-lock.txt) pins the resolved direct and transitive versions, and [scripts/install.sh](../../../../scripts/install.sh) pins the SGLang Git revision and installation sequence. Use those files as the version source of truth; Python package requirements do not provision the host driver or a complete CUDA compiler toolchain.

You need access to an NVIDIA GPU and a driver compatible with CUDA 12.9. The CUDA version displayed by `nvidia-smi` describes driver support; the PyTorch runtime should report `torch.version.cuda == "12.9"`. GPU memory requirements depend on the checkpoint, image/video budgets, and context length. See the inference reference for [two-GPU and single-GPU launch examples](inference.md#launch-the-sglang-service).

Serving may compile CUDA kernels at startup. Provide a CUDA development toolkit (`nvcc` and headers) and a compatible C++ compiler matched to the PyTorch CUDA runtime; the locked runtime wheels alone do not supply the full toolchain. Keep the pinned source and dependency versions when building for this profile. A different CUDA/dependency profile needs its own compatibility validation.

The launcher defaults to **FlashAttention-3 (`fa3`)**. Check support for your GPU and installed SGLang/kernel build. If necessary, build compatible kernels or select another supported backend with `--attention-backend` (for example, `triton`) and validate it with actual image/video requests. Backend changes can affect performance.

## Repository and installation contract

Locate the VisionWeave checkout from the working directory or the user's path. Confirm that it contains `pyproject.toml` with project name `visionweave`, `scripts/install.sh`, and `requirements-lock.txt`. Run the commands below from that checkout's root. Links assume this reference's bundled location; if the skill was copied elsewhere, resolve repository file links against the checkout.

Read these before installing:

- [scripts/install.sh](../../../../scripts/install.sh): installation order and SGLang source pin.
- [requirements-lock.txt](../../../../requirements-lock.txt): exact runtime dependency versions.
- [pyproject.toml](../../../../pyproject.toml): Python version and editable package configuration.
- [visionweave/check.py](../../../../visionweave/check.py): environment and integration checks.

The required SGLang source is **https://github.com/FFY0/sglang.git**, commit **`df2f34cca1f0b3200c6c0b2e2411916b1f0909c8`**, with package subdirectory **`python`**. Keep this pin and the lock file intact during environment setup or repair. Changing the pin requires a separate compatibility task.

The installer first installs the lock file, then installs SGLang with `--no-deps --no-build-isolation` and `SGLANG_BUILD_RUST_EXTS=none`, then installs VisionWeave in editable mode and runs its check. This preserves the CUDA 12.9 profile despite the pinned SGLang source's CUDA 13 dependency defaults. Do not replace this sequence with an unpinned SGLang install or generate a fresh dependency resolution.

## Configure the environment

When the user requires all setup and inference artifacts inside the checkout, apply the [optional isolation notes](#optional-checkout-only-isolation) before provisioning Python, installing packages, or importing CUDA libraries. A fresh venv still inherits pip constraints, library paths, and default cache locations. Keep that isolation scope throughout retries, builds, and service launches.

1. Inspect the host and selected Python interpreter. The target is Linux x86_64 and Python 3.12; Ubuntu 24.04 with Python 3.12.3 is a reference. Confirm Git, venv/pip, available disk space, GPU model/compute capability, NVIDIA driver version, and the task's GPU visibility. Inspect `nvcc --version` and the C++ compiler: serving may require JIT compilation, and the locked runtime wheels do not provide the full CUDA development toolkit. `nvidia-smi` reports driver capability; `torch.version.cuda` identifies the wheel's CUDA runtime. Neither verifies the development toolkit or attention-kernel compatibility.
2. Prefer a project-local `.venv`. Reuse an existing environment only when it belongs to this checkout and uses Python 3.12; the installer requires `sys.prefix != sys.base_prefix`. Honor a user-selected project virtualenv. Do not delete an existing environment or mutate the shared system Python to resolve conflicts; use a fresh project environment when needed and report its path. Driver changes and container GPU attachment are host prerequisites outside this virtualenv setup.
3. For a new `.venv`, run:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   bash scripts/install.sh
   ```

   If terminal calls do not preserve activation, activate in each call or consistently invoke the chosen environment's Python. The installer uses `python` from `PATH`, so ensure it points into that environment.

If `python3.12` is missing and `uv` is available, provision a user-scoped interpreter with `uv python install 3.12` and create the new environment with `uv venv --python 3.12 --seed .venv`. `--seed` supplies pip for the existing installer. Then activate the environment and run `scripts/install.sh` as above. If no suitable interpreter can be provisioned, report that prerequisite without falling back to another Python version.

For an existing project environment, first run the verification below. If it already passes, reuse it unless the user requested a fresh environment. Otherwise repair it through `scripts/install.sh` after diagnosing the failure. Keep installation logs; follow a running installer to completion before starting another one.

The project is installed in editable mode (`pip install -e .`), so the runtime uses the source under `visionweave/`. Restart the service after changing Python files to load your edits.

SGLang itself is installed into the selected environment's `site-packages`; the installer does not keep a separate editable SGLang checkout. Locate the installed package without loading the model:

```bash
python -c "from importlib.util import find_spec; print(find_spec('sglang').origin)"
```

## Verify the installed source and runtime

Check the full installed commit using pip's source metadata, with the selected environment active:

```bash
python - <<'PY'
import json
import sys
from importlib import metadata, util

expected_url = "https://github.com/FFY0/sglang.git"
expected = "df2f34cca1f0b3200c6c0b2e2411916b1f0909c8"
dist = metadata.distribution("sglang")
raw = dist.read_text("direct_url.json")
if not raw:
    raise SystemExit("SGLang source metadata is missing; reinstall with scripts/install.sh")
source = json.loads(raw)
actual_url = source.get("url")
actual = source.get("vcs_info", {}).get("commit_id")
if actual_url != expected_url or actual != expected or source.get("subdirectory") != "python":
    raise SystemExit(f"Unexpected SGLang source: url={actual_url!r}, commit={actual!r}, subdirectory={source.get('subdirectory')!r}")
print(f"Python: {sys.executable}")
print(f"SGLang package: {util.find_spec('sglang').origin}")
print(f"Source: {actual_url}")
print(f"SGLang {dist.version}; commit {actual}; subdirectory python")
PY
python -m visionweave.check
```

`visionweave.check` must exit successfully. It checks pinned package versions, the SGLang build identifier, CUDA availability, and model/processor/video-decoder/patch imports. It does not load a checkpoint or establish that a model has served a request. Confirm `torch.version.cuda` is `12.9` from the environment and retain the check output.

Record the environment path, Python version, GPU model/compute capability, driver and CUDA toolkit/compiler versions, installed SGLang commit, any local build commands or library-path overrides, and validation outcome. For an inference task, also record the selected attention backend and actual request results. For a reproducible record, save the selected environment's package inventory under the ignored `artifacts/` directory:

```bash
mkdir -p artifacts
python -m pip freeze > artifacts/visionweave-environment.txt
```

Treat a missing GPU or failed import as an incomplete runtime check, even if packages installed successfully. Report the exact failing stage and the remaining prerequisite.

## Installation diagnostics

A new venv inherits environment variables and pip configuration. If an install conflicts with the lock file, inspect `PIP_CONSTRAINT`, `PIP_BUILD_CONSTRAINT`, `PIP_REQUIREMENT`, `PIP_TARGET`, and `PIP_PREFIX` before changing dependencies. `PIP_CONFIG_FILE=/dev/null` does not clear environment-variable constraints. For example, if inherited constraints are unrelated to this project, rerun the installer with only those constraints removed:

```bash
env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT bash scripts/install.sh
```

Keep the intended package sources, proxy, and trusted CA configuration. The lock file includes CUDA-specific package indexes, so replacing a single pip index may not fix all downloads. Retry the same versions or use verified local wheels, recording their source URLs and published hashes where available. Resolve certificate errors through the machine's CA/proxy configuration. Allow disk space for downloaded CUDA wheels, installed packages, build caches, and checkpoint files.

## Resolve setup failures within the pinned profile

- **Wrong Python or global pip:** inspect `sys.executable` and `sys.prefix`; select the project environment and rerun the installer there.
- **Pip configuration or download failure:** follow [installation diagnostics](#installation-diagnostics) to check inherited settings, package sources, certificates, and cached downloads before changing dependencies. Preserve the full commit and locked versions when retrying.
- **Wrong SGLang build or missing source metadata:** use the installer in a clean project environment if the current installation cannot be repaired in place. Do not edit installed version strings or bypass `visionweave.check`.
- **CUDA unavailable:** inspect driver access and `CUDA_VISIBLE_DEVICES` within the actual container/session. Report a host prerequisite when the GPU is not attached or the driver is incompatible; package installation alone cannot validate CUDA serving.
- **CUDA compilation or native-library failure:** provide `nvcc`, CUDA headers, and a supported C++ compiler for the selected PyTorch runtime. Configure toolkit and library paths explicitly and build the affected native components for the target GPU as needed, preserving pinned sources and versions. If different CUDA wheels or dependency versions are required, handle that as a separate compatibility profile; the existing installer and checks validate only the pinned profile.
- **Attention backend failure:** the launcher defaults to FlashAttention-3 (`fa3`). Check GPU support and the SGLang/kernel build combination before attributing a failure to hardware or drivers. During a requested inference task, build compatible kernels or select a supported backend (for example, `--attention-backend triton`) and verify actual image/video requests. Report the override; success with another backend does not validate FA3 or its performance.
- **TorchCodec/FFmpeg loading:** inspect the check's selected video backend. The pinned profile includes a Decord fallback; use the backend accepted by the existing decoder check. Preserve video validation and the pinned PyTorch and decoder package versions.
- **Upstream dependency diagnostics:** the project intentionally installs SGLang without its default dependency resolution. If inspecting `pip check`, distinguish those upstream declarations from this repository's CUDA profile; record relevant diagnostics and use the repository's runtime checks without changing the pins to silence warnings.

## Optional checkout-only isolation

Use this mode when the user requires a fresh reproduction with environments, toolchains, downloads, caches, logs, and outputs inside the checkout. Ordinary setup can use a compatible host CUDA toolkit; strict isolation requires a separately provisioned toolkit inside the checkout. The operating system, system Python used to create the venv, host compiler, and NVIDIA kernel driver remain host prerequisites.

Before installing packages or provisioning Python with `uv`, run the following from the repository root in each setup/serving shell, or retain these settings in a local launch wrapper:

```bash
export VISIONWEAVE_ARTIFACTS="$PWD/artifacts/setup"
mkdir -p "$VISIONWEAVE_ARTIFACTS/tmp" "$VISIONWEAVE_ARTIFACTS/cache"
export TMPDIR="$VISIONWEAVE_ARTIFACTS/tmp"
export XDG_CACHE_HOME="$VISIONWEAVE_ARTIFACTS/cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$VISIONWEAVE_ARTIFACTS/python"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export FLASHINFER_WORKSPACE_BASE="$XDG_CACHE_HOME/flashinfer"
export TVM_FFI_CACHE_DIR="$XDG_CACHE_HOME/tvm_ffi"
export SGLANG_CACHE_DIR="$XDG_CACHE_HOME/sglang"
export SGLANG_JIT_CACHE_DIR="$XDG_CACHE_HOME/sglang_jit"
export PYTHONNOUSERSITE=1
```

Use a fresh project venv and inspect inherited `PYTHONPATH`, pip settings, `PATH`, compiler flags, and library paths for references to other environments. Select project-local CUDA toolkit and runtime paths explicitly through `CUDA_HOME`/`CUDA_PATH`, `CUDACXX`, `CUDA_LIB_PATH`, and the compiler/linker search paths. In the pinned FlashInfer version, `CUDA_LIB_PATH` controls a runtime-library preload independently of `CUDA_HOME`. `SGLANG_JIT_CACHE_DIR` likewise does not inherit `SGLANG_CACHE_DIR`.

These cache settings alone do not prove isolation. Check build logs and the libraries loaded by serving processes, for example through `/proc/<pid>/maps` on Linux. After correcting toolchain or library paths, use fresh build caches to verify that the service can compile and run without earlier artifacts. Record the selected paths and any host prerequisites used.
