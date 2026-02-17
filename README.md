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
- `start.sh`: starts vLLM, writes `glmocr.config.yaml`, starts handler
- `handler.py`: accepts PDF/images, runs full layout parse, returns JSON/Markdown + optional image artifacts
- `.env.a40`: A40 preset (single worker)
- `.env.h100`: H100 preset (single worker, higher concurrency)

## Build
```bash
docker build -t <dockerhub_user>/glm-ocr-runpod:latest .
docker push <dockerhub_user>/glm-ocr-runpod:latest
```

Pinned SDK ref:
- `GLMOCR_REF=529a0c7ee9aecf55095e6fa6d9da08e4bb3bc2a9` in `Dockerfile`
- Source commit: https://github.com/zai-org/GLM-OCR/commit/529a0c7ee9aecf55095e6fa6d9da08e4bb3bc2a9
- Transformers pinned for `glm_ocr` architecture support in vLLM:
  - `TRANSFORMERS_REF=372c27e71f80e64571ac1149d1708e641d7d44da`
  - Source repo: https://github.com/huggingface/transformers
- Compatibility patch in Docker build:
  - patches `vllm/transformers_utils/tokenizer.py` to fall back from
    `all_special_tokens_extended` to `all_special_tokens` (required with Transformers v5)
- vLLM base image pinned digest:
  - `VLLM_BASE_IMAGE=vllm/vllm-openai@sha256:2a503ea85ae35f6d556cbb12309c628a0a02af85a3f3c527ad4c0c7788553b92`
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
  --build-arg TRANSFORMERS_REF=<transformers-commit-sha> \
  -t <dockerhub_user>/glm-ocr-runpod:<tag> .
```

For strict immutability, prefer a digest:
```bash
docker build \
  --build-arg VLLM_BASE_IMAGE=vllm/vllm-openai@sha256:<digest> \
  --build-arg GLMOCR_REF=<commit-sha> \
  --build-arg TRANSFORMERS_REF=<transformers-commit-sha> \
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
