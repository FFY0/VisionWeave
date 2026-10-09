---
name: visionweave-setup
description: Set up, verify, or repair the pinned VisionWeave environment and reproduce dense Qwen3.5 image/video inference. Use for this repository's Python/CUDA setup, checkpoint preparation, serving, or inference validation.
---

# VisionWeave setup and inference

Configure the checked-out repository and verify the outcome requested by the user. An environment-only request ends after environment validation. If inference is requested, initialize or adapt the checkpoint as needed, then launch the service and complete the requested image/video checks. Prefer supplied local checkpoints and preserve their source files.

## Reference documents

Read the reference needed for the current stage; detailed commands and diagnostics live here:

| Reference | When to read |
| --- | --- |
| [Environment](references/environment.md) | Before installing or repairing Python/CUDA dependencies; includes pinned versions, source checks, host compatibility, and optional isolation |
| [Inference](references/inference.md) | When preparing checkpoints, selecting a GPU layout/backend, or verifying image/video responses |
| [Development](references/development.md) | When changing the integration or running FastV/VisionZip; includes architecture, repository layout, and relevant checks |

## Installation contract

Locate the checkout from the working directory or the user's path. Confirm that it contains `pyproject.toml` with project name `visionweave`, [scripts/install.sh](../../../scripts/install.sh), and [requirements-lock.txt](../../../requirements-lock.txt). Run commands from that checkout's root. Repository links assume this skill's bundled location; resolve them against the checkout if the skill was copied elsewhere.

The reference SGLang source is **https://github.com/FFY0/sglang.git**, commit **`df2f34cca1f0b3200c6c0b2e2411916b1f0909c8`**, package subdirectory **`python`**. Keep the source pin and lock file intact during reference-profile setup or repair. A different CUDA/dependency profile is a separate compatibility task and requires its own validation.

Use `scripts/install.sh` in the selected project virtualenv. It installs the lock file, SGLang with `--no-deps --no-build-isolation` and Rust extensions disabled, then the editable project and its checks. The pinned SGLang source defaults to CUDA 13 dependencies; replacing this sequence with upstream dependency resolution changes the CUDA 12.9 reference profile.

## Workflow

1. Read the environment reference and inspect Python, GPU model/compute capability, driver, CUDA toolkit, compiler, and available disk space. These are reference versions, not a guarantee of compatibility on every machine. Configure or build native components for the host as needed; preserve evidence when a failure's cause is unresolved.
2. Honor the user's isolation scope before provisioning Python or installing packages. If all artifacts must remain inside the checkout, apply [checkout-only isolation](references/environment.md#optional-checkout-only-isolation) first and retain it through retries and service launches. A new venv still inherits pip settings, library paths, and cache defaults.
3. Create or repair a project-local virtualenv using the installer. Reuse an environment only when it belongs to this checkout and meets the required profile; honor a request for a fresh environment. Keep logs and follow a running installation to completion before retrying.
4. Run the [source and runtime checks](references/environment.md#verify-the-installed-source-and-runtime). Verify the source URL, full commit, and package subdirectory, and retain the interpreter/package paths and check output. Diagnose failures through the environment reference; do not bypass checks or alter pins to silence errors.
5. If inference is requested, follow the inference reference using the supplied checkpoint. A native checkpoint needs initialization; its added router/pooler weights are untrained. Serve the initializer's output directly. Use [optional preparation](references/inference.md#optional-checkpoint-preparation) when an imported checkpoint's model/processor metadata needs normalization or a separate serving directory is useful. Launch with a backend supported by the host and verify each requested modality. Default `fa3` and any alternative backend must be validated on the actual machine.

## Validation and reporting

Report environment checks, service health, and inference results separately. A missing GPU or failed import leaves runtime validation incomplete. `/health` alone does not verify inference.

For each requested modality, check a nonempty response with `finish_reason="stop"` and inspect its content against the input; `length` means truncation. The [request examples](references/inference.md#send-requests) disable thinking for short smoke tests. A completed request verifies the serving path, not paper-level quality or performance. Success with another backend does not validate FA3.

Record the environment/package paths, GPU/driver/toolkit/compiler versions, full SGLang commit, local build commands or library overrides, and validation results. For inference, also retain launch commands, backend, input paths, responses, and any unresolved errors. Save logs and the package inventory under the project's ignored `artifacts/` directory.
