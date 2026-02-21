#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# vLLM + handler + GLM-OCR runtime config.
# Non-secret values are intentionally hardcoded here (endpoint env optional).
# -----------------------------------------------------------------------------
MODEL_NAME="zai-org/GLM-OCR"
MODEL_REVISION="e9134f400acad80346162536e043def285fa1022"
SERVED_MODEL_NAME="glm-ocr"
VLLM_HOST="0.0.0.0"
VLLM_PORT="8080"
VLLM_DTYPE="float16"
GPU_MEMORY_UTILIZATION="0.92"
MAX_MODEL_LEN="16384"
MAX_NUM_SEQS="96"
MAX_NUM_BATCHED_TOKENS="32768"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
VLLM_HEALTH_TIMEOUT="420"

WORKER_MAX_CONCURRENCY="2"
DOWNLOAD_CONCURRENCY="16"
MAX_PAGES_PER_JOB="128"
IMAGE_MAX_SIDE="2200"
IMAGE_JPEG_QUALITY="90"
GLMOCR_PARSE_TIMEOUT_SECONDS="1800"
GLMOCR_LOG_LEVEL="INFO"
RETURN_CROP_IMAGES_DEFAULT="false"
MAX_CROP_IMAGES="200"

GLMOCR_CONFIG_PATH="/app/glmocr.config.yaml"
GLMOCR_ENABLE_LAYOUT="true"
GLMOCR_OUTPUT_FORMAT="both"
GLMOCR_MAX_WORKERS="16"
GLMOCR_CONNECTION_POOL_SIZE="128"
GLMOCR_PAGE_MAXSIZE="100"
GLMOCR_REGION_MAXSIZE="800"
GLMOCR_MAX_TOKENS_PER_PAGE="4096"
GLMOCR_TEMPERATURE="0.01"
GLMOCR_TOP_P="0.9"
GLMOCR_DEFAULT_PROMPT="Recognize all text, formulas, and tables in the document."

GLMOCR_CONNECT_TIMEOUT="30"
GLMOCR_REQUEST_TIMEOUT="180"
GLMOCR_RETRY_MAX_ATTEMPTS="3"
GLMOCR_RETRY_BACKOFF_BASE_SECONDS="0.75"

GLMOCR_LAYOUT_MODEL_DIR="PaddlePaddle/PP-DocLayoutV3_safetensors"
GLMOCR_LAYOUT_THRESHOLD="0.3"
GLMOCR_LAYOUT_BATCH_SIZE="1"
GLMOCR_LAYOUT_WORKERS="1"
GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES="0"

export MODEL_NAME MODEL_REVISION SERVED_MODEL_NAME VLLM_HOST VLLM_PORT VLLM_DTYPE
export GPU_MEMORY_UTILIZATION MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
export LIMIT_MM_PER_PROMPT SPECULATIVE_CONFIG VLLM_HEALTH_TIMEOUT

export WORKER_MAX_CONCURRENCY DOWNLOAD_CONCURRENCY MAX_PAGES_PER_JOB IMAGE_MAX_SIDE IMAGE_JPEG_QUALITY
export GLMOCR_PARSE_TIMEOUT_SECONDS GLMOCR_LOG_LEVEL RETURN_CROP_IMAGES_DEFAULT MAX_CROP_IMAGES

export GLMOCR_CONFIG_PATH GLMOCR_ENABLE_LAYOUT GLMOCR_OUTPUT_FORMAT GLMOCR_MAX_WORKERS
export GLMOCR_CONNECTION_POOL_SIZE GLMOCR_PAGE_MAXSIZE GLMOCR_REGION_MAXSIZE
export GLMOCR_MAX_TOKENS_PER_PAGE GLMOCR_TEMPERATURE GLMOCR_TOP_P GLMOCR_DEFAULT_PROMPT
export GLMOCR_CONNECT_TIMEOUT GLMOCR_REQUEST_TIMEOUT GLMOCR_RETRY_MAX_ATTEMPTS GLMOCR_RETRY_BACKOFF_BASE_SECONDS
export GLMOCR_LAYOUT_MODEL_DIR GLMOCR_LAYOUT_THRESHOLD GLMOCR_LAYOUT_BATCH_SIZE GLMOCR_LAYOUT_WORKERS GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES

echo "[start.sh] Model: ${MODEL_NAME} -> ${SERVED_MODEL_NAME}"
if [ -n "${MODEL_REVISION}" ]; then
  echo "[start.sh] Model revision pinned: ${MODEL_REVISION}"
else
  echo "[start.sh] WARNING: MODEL_REVISION is empty (model snapshot is not pinned)"
fi
if vllm --version >/tmp/vllm_version.txt 2>/dev/null; then
  echo "[start.sh] vLLM(runtime): $(cat /tmp/vllm_version.txt)"
fi
if python3 -c 'import transformers; print(transformers.__version__)' >/tmp/transformers_version.txt 2>/dev/null; then
  echo "[start.sh] Transformers(global): $(cat /tmp/transformers_version.txt)"
fi
if python3 -c 'import importlib.metadata as m; print(m.version("mistral-common"))' >/tmp/mistral_common_version.txt 2>/dev/null; then
  echo "[start.sh] mistral-common(global): $(cat /tmp/mistral_common_version.txt)"
fi
if python3 -c 'import importlib.metadata as m; print(m.version("huggingface_hub"))' >/tmp/hf_hub_version.txt 2>/dev/null; then
  echo "[start.sh] huggingface_hub(global): $(cat /tmp/hf_hub_version.txt)"
fi
if python3 -c 'import importlib.metadata as m; print(m.version("tqdm"))' >/tmp/tqdm_version.txt 2>/dev/null; then
  echo "[start.sh] tqdm(global): $(cat /tmp/tqdm_version.txt)"
fi
echo "[start.sh] vLLM: dtype=${VLLM_DTYPE}, gpu_mem=${GPU_MEMORY_UTILIZATION}, max_seqs=${MAX_NUM_SEQS}, max_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "[start.sh] GLMOCR: enable_layout=${GLMOCR_ENABLE_LAYOUT}, output_format=${GLMOCR_OUTPUT_FORMAT}, max_workers=${GLMOCR_MAX_WORKERS}, conn_pool=${GLMOCR_CONNECTION_POOL_SIZE}"
echo "[start.sh] Handler: worker_concurrency=${WORKER_MAX_CONCURRENCY}, max_pages_per_job=${MAX_PAGES_PER_JOB}"

# Persist caches to RunPod network volume when available.
VOLUME_PATH="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
if [ -d "${VOLUME_PATH}" ]; then
  export HF_HOME="${VOLUME_PATH}/huggingface"
  export HUGGINGFACE_HUB_CACHE="${VOLUME_PATH}/huggingface/hub"
  export XDG_CACHE_HOME="${VOLUME_PATH}/xdg-cache"
  mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${XDG_CACHE_HOME}/vllm"
  echo "[start.sh] Using cache volume: ${VOLUME_PATH}"
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  echo "[start.sh] HF auth: enabled via HF_TOKEN"
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "[start.sh] HF auth: enabled via HUGGING_FACE_HUB_TOKEN"
else
  echo "[start.sh] WARNING: HF auth is not configured; downloads may be rate-limited"
fi

# Generate GLM-OCR SDK config from the official package template, then apply
# runtime overrides. This preserves required layout keys (id2label, mappings).
python3 - <<'PY'
import os
from pathlib import Path

import yaml
from glmocr.config import GlmOcrConfig


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


cfg_path = Path(env("GLMOCR_CONFIG_PATH", "/app/glmocr.config.yaml"))
template_path = Path(GlmOcrConfig.default_path())
if not template_path.exists():
    raise SystemExit(f"[start.sh] GLM-OCR template config not found: {template_path}")

data = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
pipeline = data.setdefault("pipeline", {})
ocr_api = pipeline.setdefault("ocr_api", {})
page_loader = pipeline.setdefault("page_loader", {})
result_formatter = pipeline.setdefault("result_formatter", {})
layout = pipeline.setdefault("layout", {})

pipeline["enable_layout"] = as_bool(os.environ.get("GLMOCR_ENABLE_LAYOUT"), True)
pipeline["max_workers"] = int(env("GLMOCR_MAX_WORKERS", "16"))
pipeline["page_maxsize"] = int(env("GLMOCR_PAGE_MAXSIZE", "100"))
pipeline["region_maxsize"] = int(env("GLMOCR_REGION_MAXSIZE", "800"))

ocr_api["api_host"] = env("VLLM_HOST", "127.0.0.1")
ocr_api["api_port"] = int(env("VLLM_PORT", "8080"))
ocr_api["model"] = env("SERVED_MODEL_NAME", "glm-ocr")
ocr_api["api_mode"] = "openai"
ocr_api["connect_timeout"] = int(env("GLMOCR_CONNECT_TIMEOUT", "30"))
ocr_api["request_timeout"] = int(env("GLMOCR_REQUEST_TIMEOUT", "180"))
ocr_api["retry_max_attempts"] = int(env("GLMOCR_RETRY_MAX_ATTEMPTS", "3"))
ocr_api["retry_backoff_base_seconds"] = float(
    env("GLMOCR_RETRY_BACKOFF_BASE_SECONDS", "0.75")
)
ocr_api["connection_pool_size"] = int(env("GLMOCR_CONNECTION_POOL_SIZE", "128"))

page_loader["max_tokens"] = int(env("GLMOCR_MAX_TOKENS_PER_PAGE", "4096"))
page_loader["temperature"] = float(env("GLMOCR_TEMPERATURE", "0.01"))
page_loader["top_p"] = float(env("GLMOCR_TOP_P", "0.9"))
page_loader["image_format"] = "JPEG"
page_loader["default_prompt"] = env(
    "GLMOCR_DEFAULT_PROMPT", page_loader.get("default_prompt", "")
)

result_formatter["output_format"] = env("GLMOCR_OUTPUT_FORMAT", "both")

layout["model_dir"] = env(
    "GLMOCR_LAYOUT_MODEL_DIR", "PaddlePaddle/PP-DocLayoutV3_safetensors"
)
layout["threshold"] = float(env("GLMOCR_LAYOUT_THRESHOLD", "0.3"))
layout["batch_size"] = int(env("GLMOCR_LAYOUT_BATCH_SIZE", "1"))
layout["workers"] = int(env("GLMOCR_LAYOUT_WORKERS", "1"))
layout["cuda_visible_devices"] = env("GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES", "0")

if pipeline["enable_layout"]:
    missing = [key for key in ("id2label", "label_task_mapping") if not layout.get(key)]
    if missing:
        joined = ", ".join(f"pipeline.layout.{key}" for key in missing)
        raise SystemExit(
            "[start.sh] invalid GLM-OCR layout config: missing required fields: "
            + joined
        )

cfg_path.parent.mkdir(parents=True, exist_ok=True)
cfg_path.write_text(
    yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
print(f"[start.sh] Loaded GLM-OCR template: {template_path}")
print(f"[start.sh] Wrote GLM-OCR config: {cfg_path}")
if pipeline["enable_layout"]:
    print(f"[start.sh] Layout classes: {len(layout.get('id2label', {}))}")
PY

VLLM_PID=""
cleanup() {
  if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "[start.sh] Stopping vLLM (pid=${VLLM_PID})"
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

echo "[start.sh] Starting vLLM server on :${VLLM_PORT}"
vllm_args=(
  serve "${MODEL_NAME}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --dtype "${VLLM_DTYPE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --allowed-local-media-path /tmp
)

if [ -n "${MODEL_REVISION}" ]; then
  vllm_args+=(--revision "${MODEL_REVISION}")
fi

if [ -n "${LIMIT_MM_PER_PROMPT}" ]; then
  if python3 -c 'import json,sys; json.loads(sys.argv[1]); print("ok")' "${LIMIT_MM_PER_PROMPT}" >/dev/null 2>&1; then
    vllm_args+=(--limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}")
    echo "[start.sh] Using --limit-mm-per-prompt=${LIMIT_MM_PER_PROMPT}"
  else
    echo "[start.sh] WARNING: LIMIT_MM_PER_PROMPT is not valid JSON; skipping flag (value=${LIMIT_MM_PER_PROMPT})"
  fi
fi

if [ -n "${SPECULATIVE_CONFIG}" ]; then
  vllm_args+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

vllm_env=(
  env
  -u VLLM_BASE_IMAGE_REF
  -u VLLM_DTYPE
  -u VLLM_HEALTH_TIMEOUT
  -u VLLM_HOST
  -u VLLM_PORT
)

"${vllm_env[@]}" vllm "${vllm_args[@]}" &
VLLM_PID=$!

HEALTH_URL="http://127.0.0.1:${VLLM_PORT}/health"
echo "[start.sh] Waiting for health: ${HEALTH_URL}"
for i in $(seq 1 "${VLLM_HEALTH_TIMEOUT}"); do
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    wait "${VLLM_PID}" || true
    echo "[start.sh] ERROR: vLLM process exited before becoming healthy"
    exit 1
  fi
  if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "[start.sh] vLLM healthy after ${i}s"
    break
  fi
  if [ "${i}" -eq "${VLLM_HEALTH_TIMEOUT}" ]; then
    echo "[start.sh] ERROR: vLLM failed to become healthy within ${VLLM_HEALTH_TIMEOUT}s"
    exit 1
  fi
  sleep 1
done

echo "[start.sh] Starting RunPod handler"
exec python3 -u /app/handler.py
