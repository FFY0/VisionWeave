# Checkpoints and inference

[Skill](../SKILL.md) · [Project README](../../../../README.md)

This reference covers dense Qwen3.5 image/video serving. First complete [environment setup and validation](environment.md), then run the commands below from the repository root with the same environment and CUDA/library settings. Trained weights are supplied separately. The paper's Qwen3.8-27B integration, training pipeline, and full evaluation pipeline are not included in this release.

[Checkpoint initialization](#initialize-a-visionweave-checkpoint) · [Launch options](#launch-the-sglang-service) · [Image/video requests](#send-requests) · [Optional preparation](#optional-checkpoint-preparation) · [Limitations](#supported-features-and-limitations)

For native Qwen3.5 weights, initialize the added modules and then launch that output directory directly. For an existing compatible VisionWeave checkpoint, go straight to [Launch the SGLang Service](#launch-the-sglang-service). Use [optional preparation](#optional-checkpoint-preparation) if imported model/processor metadata needs normalization or you want a separate serving directory. Prefer a supplied local checkpoint and preserve its source files.

## Initialize a VisionWeave Checkpoint

With a local safetensors checkpoint of native `Qwen/Qwen3.5-4B`, run the following in the installed environment:

```bash
python scripts/init_visionweave_checkpoint.py \
  --source /path/to/Qwen3.5-4B \
  --destination ./models/visionweave-init \
  --seed 0
```

The script initializes the added modules on CPU and copies native weights file by file, without loading the full 4B model into memory or onto a GPU. Both single-file and sharded safetensors checkpoints are supported. The destination must be new and outside the source directory. The source checkpoint is preserved.

By default, the output is a standalone directory containing weights, an index, configuration, and tokenizer/processor files. Allow additional disk space for the native weights and new parameters. Pass `--link-base-weights` to reuse native weights through symbolic links; keep the source files available when using this option.

Initialization follows these rules:

- Native ViT, merger, and language-model weights are preserved.
- The pooler uses transparent initialization: it initially applies LayerNorm to each of four ViT features, then averages them.
- The router uses six local/global cross-attention layers and samples depths `[0, 12, 24]` for the 4B model. Linear weights use a normal distribution with the vision configuration's `initializer_range`; the query and linear biases start at zero.
- The classification head starts with zero weights. `last_layer_bias` sets the initial compression probability, and `router_bias` starts at zero. Defaults are `P(compress)=0.1` and a routing threshold of `0.5`, so the initial service retains all native visual tokens. Adjust these with `--initial-compression-prob` and `--route-threshold`; compression uses `p >= threshold`.
- Added router/pooler weights follow the native vision merger's dtype; the classification head and routing biases remain FP32. A fixed `--seed` reproduces the added weights, and the configuration records initialization parameters.

**This is an untrained VisionWeave initialization checkpoint.** It can be used for subsequent training or to validate the serving pipeline. Compression behavior must be learned through training. This repository provides initialization and inference; a training implementation must use matching VisionWeave modules and weight names. The native Transformers Qwen3.5 classes do not automatically load these added modules.

## Launch the SGLang Service

The initializer writes the VisionWeave architecture and routing threshold into its output, so `prepare` is not a required step for that output. These examples launch `./models/visionweave-init` directly; replace the model path when using another compatible checkpoint or the output of optional preparation. Keep the supplied processor files compatible with the selected image/video budgets.

This example uses two GPUs: GPU 0 runs the ViT and router, and GPU 1 runs the language model.

```bash
source .venv/bin/activate
python -m visionweave.serve \
  --model-path ./models/visionweave-init \
  --encoder-gpus 0 \
  --gpus 1 \
  --host 127.0.0.1 \
  --port 30000
```

For language-model TP2, pass `--gpus 0,1`. You can also pass `--encoder-gpus 0,1` to launch one encoder on each of those GPUs, sharing GPU memory with the language service.

To share a single GPU:

```bash
python -m visionweave.serve \
  --model-path ./models/visionweave-init \
  --gpus 0 --encoder-gpus 0 \
  --mem-fraction 0.60 --encoder-mem-fraction 0.10
```

Logs are written to `logs/encoder-0.log` and `logs/language.log`. The launcher prints `LANGUAGE SERVICE READY` when all roles pass `/health`. Ctrl-C stops all processes started by the launcher; if any role exits, the remaining roles are stopped as well.

Encoders listen on localhost only, starting at port 31000 by default. The language service defaults to port 30000. Pass `--host 0.0.0.0` to make the language service accessible from other machines.

Common options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--image-tokens` | 512 | Input native visual token budget per image |
| `--video-tokens` | 512 | Input native visual token budget per frame |
| `--max-frames` | 64 | Maximum sampled video frames; even and at least 4 |
| `--context-length` | 32768 | Total context length |
| `--route-threshold` | Checkpoint metadata | Compress a block when `p >= threshold` |
| `--max-running-requests` | 8 | VisionWeave language-service concurrency limit |
| `--attention-backend` | `fa3` | Attention backend; select one supported by your GPU and SGLang/kernel build |
| `--encoder-port` | 31000 | First encoder port; later encoders use successive ports |
| `--dry-run` | Off | Validate configuration and print commands without loading a model or starting services |

Input budgets are converted to pixel limits as `token count × 1024`, with image dimensions rounded to multiples of 64. The token count after routing depends on content. A threshold of `0` compresses all blocks; a finite threshold greater than `1` retains all blocks.

## Send Requests

The service exposes SGLang's OpenAI-compatible API. Check its health first:

```bash
curl --fail http://127.0.0.1:30000/health
```

In another terminal, activate the same project environment and apply any CUDA/library settings used for setup. The following examples disable Qwen3.5 thinking and request short answers. For longer reasoning tasks, enable thinking explicitly and increase `max_tokens`.

For an image request, replace `example.jpg` with a local JPEG:

```bash
python - <<'PY'
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY", timeout=600.0, max_retries=0)
image = base64.b64encode(Path("example.jpg").read_bytes()).decode()
response = client.chat.completions.create(
    model="visionweave",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Briefly describe the main objects and their colors."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
    ]}],
    max_tokens=256,
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
choice = response.choices[0]
print(choice.message.content)
print("finish_reason:", choice.finish_reason)
if choice.finish_reason != "stop" or not (choice.message.content or "").strip():
    raise RuntimeError("Expected a complete, nonempty response; inspect the response and server logs.")
PY
```

For a video request, replace `example.mp4` with a short local MP4. SGLang accepts the `video_url` content type through the same endpoint:

```bash
python - <<'PY'
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY", timeout=600.0, max_retries=0)
video = base64.b64encode(Path("example.mp4").read_bytes()).decode()
response = client.chat.completions.create(
    model="visionweave",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Briefly describe the main object and how it moves in this video."},
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video}"}},
    ]}],
    max_tokens=256,
    temperature=0,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
choice = response.choices[0]
print(choice.message.content)
print("finish_reason:", choice.finish_reason)
if choice.finish_reason != "stop" or not (choice.message.content or "").strip():
    raise RuntimeError("Expected a complete, nonempty response; inspect the response and server logs.")
PY
```

Video processing samples frames, routes each frame, and constructs temporal positions from timestamps. Use the matching MIME type if you change the input format. A `finish_reason` of `length` means the answer was truncated; inspect it and adjust the output budget or thinking setting. Check response content against the actual input as well: a nonempty response ending with `stop` verifies the serving path, not model quality or paper-level benchmark results.

## Optional Checkpoint Preparation

Use `prepare` for exported **dense Qwen3.5 VisionWeave checkpoints** whose model or processor metadata needs conversion to the serving format, or when you want a separate serving directory. It also accepts initialized checkpoints, but is optional for the initializer output used above. It does not add or train router/pooler weights. The loader requires:

- `vision_config` with `spatial_merge_size=2`, `compression_spatial_merge_size=2`, `effective_compression_spatial_merge_size=1`, `patch_size=16`, and no deepstack.
- A router with six local/global cross-attention layers and three sampled ViT depths.
- Weights containing `model.visual.compression_projector.*`, `model.visual.router_cross_attn.*`, `model.visual.mask_net.*`, and `model.visual.last_layer_bias`.
- A zero-valued `model.visual.router_bias`. Missing required weights or a nonzero routing bias cause loading to fail.
- Hugging Face tokenizer, image/video processor files, and safetensors weights.

Create a separate serving directory:

```bash
python -m visionweave.prepare \
  --source /path/to/visionweave-checkpoint \
  --destination ./models/visionweave
```

Launch this prepared output with `--model-path ./models/visionweave`. The tool symlinks weights and tokenizer assets, writes a serving configuration with architecture `Qwen3_5VisionWeaveForConditionalGeneration`, and normalizes image/video processor settings and nested processor metadata. It preserves the source directory and excludes training scripts, training state, and training argument files from the output. You do not need to rename the source checkpoint's architecture manually. Keep the source weights available because the serving directory links to them.

The routing threshold is read from `visionweave.route_threshold` in `config.json`, or from `router_hard_threshold` in the source checkpoint's `args.json`. Only the threshold value is copied into the new configuration. If this metadata is absent, pass the training threshold explicitly with `--route-threshold`.

If the exported tokenizer is incomplete or has been reserialized, use `--tokenizer-source /path/to/original-tokenizer` to supply the tokenizer/processor used during training. The destination directory must not already exist.

## Supported Features and Limitations

The integration supports dense Qwen3.5 image/video inference with language TP1/TP2. MoE, deepstack, audio, quantized vision towers, ViT CUDA graphs, and the `flashinfer_cudnn` vision backend are unsupported.

VisionWeave disables embedding caching and adaptive dispatch that bypasses the encoder, keeping routing metadata and sequence lengths consistent. Patches check the interfaces of the pinned SGLang revision and stop startup if those interfaces have changed.
