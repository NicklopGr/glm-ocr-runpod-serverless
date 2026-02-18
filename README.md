# GLM-OCR RunPod Serverless (Full Layout Mode)

This worker is configured for **official GLM-OCR full layout parsing** to better match PaddleOCR-VL 1.5 style outputs:
- layout detection enabled (`PP-DocLayoutV3_safetensors`)
- OCR over detected regions via local `vllm serve`
- output format `both` (JSON + Markdown)
- optional crop image export (`imgs/`)

Architecture:
- Process 1: `vllm serve zai-org/GLM-OCR` on `localhost:8080`
- Process 2: RunPod `handler.py` that calls `glmocr parse` with runtime config

## Hardware Note
RunPod `RTX A40` is usually **48GB VRAM** (not 40GB).

## Repo Files
- `Dockerfile`: installs runtime + official GLM-OCR SDK
- `start.sh`: starts vLLM, loads official `glmocr/config.yaml` template, applies runtime overrides, starts handler
- `handler.py`: accepts PDF/images, runs full layout parse, returns JSON/Markdown + optional image artifacts
- `.env.a40`: A40 preset (single worker)
- `.env.h100`: H100 preset (single worker, higher concurrency)

## Build
```bash
docker build -t <dockerhub_user>/glm-ocr-runpod:latest .
docker push <dockerhub_user>/glm-ocr-runpod:latest
```

Pinned refs and compatibility:
- `GLMOCR_REF=529a0c7ee9aecf55095e6fa6d9da08e4bb3bc2a9` in `Dockerfile`
- Source commit: https://github.com/zai-org/GLM-OCR/commit/529a0c7ee9aecf55095e6fa6d9da08e4bb3bc2a9
- vLLM base image pinned by commit + digest:
  - `VLLM_BASE_IMAGE=vllm/vllm-openai:nightly-d00df624f313a6a5a7a6245b71448b068b080cd7@sha256:3f5ad92f63e3f4b0073cca935a976d4cbf2a21e22cf06a95fd6df47759e10e04`
- Runtime pins used globally (single shared Python env for vLLM + handler):
  - `TRANSFORMERS_VERSION=5.2.0`
  - `TOKENIZERS_VERSION=0.22.2`
  - `HUGGINGFACE_HUB_VERSION=1.4.1`
  - `TQDM_VERSION=4.67.1`
- Why this matters:
  - GLM-OCR model type (`glm_ocr`) is present in Transformers `>=5.1.0`
  - if vLLM and handler use different Python environments, startup can fail with:
    `model type 'glm_ocr' but Transformers does not recognize this architecture`
  - this image avoids split-runtime drift by installing all runtime deps globally
- Build-time compatibility gates:
  - verifies native GLM-OCR files and registry entry exist in vLLM:
    - `vllm/model_executor/models/glm_ocr.py`
    - `vllm/model_executor/models/glm_ocr_mtp.py`
    - `GlmOcrForConditionalGeneration` mapping in `registry.py`
  - enforces critical dependency edges:
    - `vllm -> tokenizers`
    - `transformers -> huggingface_hub, tokenizers, tqdm`
  - enforces `vllm -> transformers` with an explicit allowlist only for:
    - vLLM nightly metadata still advertising `<5` while runtime is validated
      for GLM-OCR using Transformers `>=5.1.0`
  - verifies installed Transformers exposes `glm_ocr` model type
  - verifies `transformers >= 5.1.0`
  - verifies `glmocr` package import/version in final runtime
  - intentionally does not use raw `pip check` as a build gate because it fails
    on the same known allowlisted metadata mismatch and unrelated system deps
- GLM-OCR model snapshot pin:
  - default in `start.sh` (used when endpoint env is missing)
  - `MODEL_REVISION=e9134f400acad80346162536e043def285fa1022`
  - same value in `.env.a40` and `.env.h100` presets
  - Source commit: https://huggingface.co/zai-org/GLM-OCR/commit/e9134f400acad80346162536e043def285fa1022

To compare builds across versions, override explicitly:
```bash
docker build \
  --build-arg VLLM_BASE_IMAGE=<image-tag-or-digest> \
  --build-arg GLMOCR_REF=<commit-sha> \
  --build-arg TRANSFORMERS_VERSION=<transformers-version> \
  --build-arg TOKENIZERS_VERSION=<tokenizers-version> \
  --build-arg HUGGINGFACE_HUB_VERSION=<huggingface_hub-version> \
  --build-arg TQDM_VERSION=<tqdm-version> \
  -t <dockerhub_user>/glm-ocr-runpod:<tag> .
```

For strict immutability, prefer a digest:
```bash
docker build \
  --build-arg VLLM_BASE_IMAGE=vllm/vllm-openai:<tag>@sha256:<digest> \
  --build-arg GLMOCR_REF=<commit-sha> \
  --build-arg TRANSFORMERS_VERSION=<transformers-version> \
  --build-arg TOKENIZERS_VERSION=<tokenizers-version> \
  --build-arg HUGGINGFACE_HUB_VERSION=<huggingface_hub-version> \
  --build-arg TQDM_VERSION=<tqdm-version> \
  -t <dockerhub_user>/glm-ocr-runpod:<tag> .
```

And pin model snapshot in endpoint env:
```env
MODEL_REVISION=<hf_commit_or_tag>
```

## RunPod Endpoint Setup
Recommended:
- `GPU`: A40 (48GB) or H100 (80GB)
- `Active workers`: `1`
- `Max workers`: `1`
- `Network volume`: enabled
- `Container disk`: at least 40GB

Environment presets:
- A40: paste from `.env.a40`
- H100: paste from `.env.h100`

## GitHub Workflow (Build First On GitHub)
This repo now includes:
- `.github/workflows/ghcr-build.yml`

Behavior:
- on each push to `main`, GitHub Actions builds this Docker image first
- if build succeeds, it pushes to GHCR:
  - `ghcr.io/nicklopgr/glm-ocr-runpod-serverless:latest`
  - `ghcr.io/nicklopgr/glm-ocr-runpod-serverless:main`
  - `ghcr.io/nicklopgr/glm-ocr-runpod-serverless:sha-<commit>`
- on pull requests, it validates build without pushing

Recommended RunPod setup with this workflow:
- set endpoint container image to:
  - `ghcr.io/nicklopgr/glm-ocr-runpod-serverless:latest`
- for strict reproducibility, use:
  - `ghcr.io/nicklopgr/glm-ocr-runpod-serverless:sha-<commit>`
- set GHCR package visibility to public, or configure registry credentials in RunPod

## GitHub Workflow (Auto RunPod Rebuild From GitHub Source)
This repo now includes:
- `.github/workflows/runpod-release.yml`

Behavior:
- on each push to `main`, it creates a new **pre-release**
- this pre-release triggers RunPod GitHub integration to rebuild/deploy
- manual trigger is also available via `workflow_dispatch`

Skip one auto-deploy:
- include `[skip runpod]` in the commit message

## Input Formats
Supported:
- `pdf_url` or `pdf_base64`
- `image_urls` or `images_base64`
- `source_url` or `source_base64` (auto-detect PDF vs image)

Example:
```json
{
  "input": {
    "pdf_url": "https://.../statement.pdf",
    "start_page": 1,
    "end_page": 12,
    "return_crop_images": true,
    "crop_images_base64": false,
    "max_crop_images": 50,
    "return_layout_visualization": true,
    "layout_visualization_base64": false
  }
}
```

Warmup:
```json
{"input":{"warmup":true}}
```

## Output Behavior
- `glmocr_ref`: pinned SDK commit used by this image
- `vllm_base_image_ref`: pinned vLLM base image reference used by this image
- `model_revision`: pinned model revision if set
- `json_result`: structured full-layout result from GLM-OCR pipeline
- `markdown_result`: markdown output (tables typically appear as markdown/HTML blocks from formatter)
- `crop_images`: optional extracted region images from `imgs/`
- `layout_visualizations`: optional visualization images if generated by pipeline run

## GLM-OCR SDK Behavior (Official)
The SDK pipeline in self-hosted mode is:
1. Load pages/images (`PageLoader`)
2. Detect layout regions (`PPDocLayoutV3`)
3. Map each region label to a task via `pipeline.layout.label_task_mapping`
4. OCR regions in parallel (`OCRClient`)
5. Format to JSON/Markdown (`ResultFormatter`)

Important routing rules from official config:
- `text`, `table`, `formula`: region is sent to VLM with task-specific prompt
- `skip`: region is kept in output, OCR is not run for that region
- `abandon`: region is dropped

Image/cheque note:
- By default, `image`/`chart` labels are in `skip`, so they will not be OCRed as text.
- If cheque-like regions must be OCRed, map those labels to `text` in `pipeline.layout.label_task_mapping`, or run with layout disabled for that request.

Stability note:
- This image now builds runtime config from the official SDK template instead of a hand-written subset, preserving required layout fields like `pipeline.layout.id2label` and `pipeline.layout.label_task_mapping`.

## Throughput Tuning Order
1. Increase `GLMOCR_MAX_WORKERS` and `GLMOCR_CONNECTION_POOL_SIZE`.
2. Increase `MAX_NUM_SEQS` / `MAX_NUM_BATCHED_TOKENS` if GPU headroom exists.
3. Increase `WORKER_MAX_CONCURRENCY` only after step 1-2 are stable.

If unstable/OOM:
- reduce `WORKER_MAX_CONCURRENCY`
- reduce `GLMOCR_MAX_WORKERS`
- reduce `MAX_NUM_SEQS`
- reduce `MAX_MODEL_LEN`

## Official Prompt Mapping Note
In this full layout mode, task prompts are handled inside GLM-OCR config mapping:
- `Text Recognition:`
- `Table Recognition:`
- `Formula Recognition:`

For raw model-only calls (without layout pipeline), these prompts are sent directly to VLM.
