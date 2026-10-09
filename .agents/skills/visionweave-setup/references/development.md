# Development and baseline adapters

[Skill](../SKILL.md) · [Project README](../../../../README.md) · [Paper PDF](../../../../paper/2610.07987v1.pdf)

Read this reference when working on the integration, running FastV/VisionZip, or changing the code. For ordinary VisionWeave inference, use [checkpoints and inference](inference.md). Run commands from the repository root after [environment setup](environment.md).

[Integration](#integration-overview) · [FastV / VisionZip](#fastv--visionzip) · [Development checks](#development) · [Repository layout](#repository-layout)

## Integration Overview

```text
Image / video → Encoder: ViT → Router → Compress / retain → embeddings + route
                                                                  ↓
                    Language: exact placeholders + M-RoPE → SGLang → response
```

Each routing block contains four native visual tokens. The router either retains all four or compresses them into one. Given N native tokens and C compressed blocks, the output length is `N - 3C`. The vision encoder must therefore run before the language service finalizes sequence lengths, positional encodings, and KV-cache allocation.

Full retention and full compression use the same routing mechanism. The service uses SGLang's separate encoder and language interfaces, transferring embeddings and routing metadata through `zmq_to_tokenizer`. External model/processor registration and in-process patches connect the extension to SGLang.

## FastV / VisionZip

These adapters use **native dense Qwen3.5 checkpoints** and do not require routed checkpoint preparation:

```bash
python -m visionweave.serve \
  --model-path /path/to/native-Qwen3.5 \
  --method fastv --gpus 0 --keep-ratio 0.5

python -m visionweave.serve \
  --model-path /path/to/native-Qwen3.5 \
  --method visionzip --gpus 0 --keep-ratio 0.5 --contextual-fraction 0.16
```

Run these commands separately. Their default served model names are `fastv` and `visionzip`.

The FastV adapter scores importance using the last ViT attention layer and retains top-k merger tokens. This is the **post-ViT variant** used in the paper; the original FastV prunes at an intermediate LLM layer. The VisionZip adapter retains dominant tokens and aggregates other tokens into contextual tokens using key similarity. Both process each frame separately and restore the selected tokens' spatial positions. `--keep-ratio 1` uses the native path.

Both adapters require a full prefill, single-request scheduling, and disabled radix caching and overlap scheduling. The launcher sets these options automatically. Decode can use a full CUDA graph; prefill and ViT graphs are disabled.

## Development

After installing the runtime environment, install development dependencies and run checks from the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check visionweave scripts tests
```

Unit tests use synthetic tensors to cover checkpoint initialization, routing layouts, payloads, positional encoding, adapter algorithms, and launch configuration. Run relevant tests and Ruff after changes. Model or SGLang interface changes also require restarting the service and validating actual requests.

When changing runtime dependencies, update both `requirements.txt` and `requirements-lock.txt`, then rerun `bash scripts/install.sh` to synchronize the environment. After editing command entry points or installation settings in `pyproject.toml`, register the project again in the active environment:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

## Repository Layout

```text
.agents/skills/visionweave-setup/  Agent setup and inference skill
├── SKILL.md                     Workflow and validation requirements
├── references/                  Environment, inference, and development details
└── agents/                      Skill invocation metadata
visionweave/                     Inference integration, routing, positions, and serving
├── models/                      Model registration, weight loading, and forward pass
├── processors/                  Visual placeholders and M-RoPE alignment
└── baselines/                   FastV / VisionZip token selection and aggregation
    ├── models/                  Baseline model integration
    └── processors/              Baseline placeholder and position handling
scripts/
├── install.sh                   Environment installation entry point
└── init_visionweave_checkpoint.py  Native Qwen3.5 → VisionWeave initialization
tests/                          Unit and interface tests
LICENSES/                       Third-party license texts
THIRD_PARTY_NOTICES.md           Third-party attribution and license notes
pyproject.toml                  Package installation, entry points, and Ruff settings
requirements.txt                Inference dependency profile
requirements-lock.txt           Pinned runtime dependencies
requirements-dev.txt            Development dependencies
assets/                         Paper figures and README assets
paper/                          Paper PDF
README.md                       Paper overview, results, and usage entry points
```

`.venv/` contains the local Python environment; `*.egg-info/` contains installation metadata. `__pycache__/` and `.ruff_cache/` are tool caches. Model serving directories under `models/`, logs under `logs/`, and validation artifacts under `artifacts/` are also ignored by `.gitignore`. These directories are created locally as needed.
