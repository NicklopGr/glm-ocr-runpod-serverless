#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# vLLM defaults (A40-optimized baseline). Override via RunPod env vars.
# -----------------------------------------------------------------------------
: "${MODEL_NAME:=zai-org/GLM-OCR}"
: "${MODEL_REVISION:=}"
: "${SERVED_MODEL_NAME:=glm-ocr}"
: "${VLLM_HOST:=0.0.0.0}"
: "${VLLM_PORT:=8080}"
: "${VLLM_DTYPE:=float16}"
: "${TRUST_REMOTE_CODE:=true}"
: "${GPU_MEMORY_UTILIZATION:=0.92}"
: "${MAX_MODEL_LEN:=16384}"
: "${MAX_NUM_SEQS:=96}"
: "${MAX_NUM_BATCHED_TOKENS:=32768}"
: "${LIMIT_MM_PER_PROMPT:=}"
: "${SPECULATIVE_CONFIG:=}"
: "${VLLM_HEALTH_TIMEOUT:=420}"

# -----------------------------------------------------------------------------
# RunPod handler + GLM-OCR full-layout pipeline defaults.
# -----------------------------------------------------------------------------
: "${WORKER_MAX_CONCURRENCY:=1}"
: "${DOWNLOAD_CONCURRENCY:=16}"
: "${MAX_PAGES_PER_JOB:=128}"
: "${IMAGE_MAX_SIDE:=2200}"
: "${IMAGE_JPEG_QUALITY:=90}"
: "${GLMOCR_PARSE_TIMEOUT_SECONDS:=1800}"
: "${GLMOCR_LOG_LEVEL:=INFO}"
: "${RETURN_CROP_IMAGES_DEFAULT:=false}"
: "${MAX_CROP_IMAGES:=200}"

: "${GLMOCR_CONFIG_PATH:=/app/glmocr.config.yaml}"
: "${GLMOCR_ENABLE_LAYOUT:=true}"
: "${GLMOCR_OUTPUT_FORMAT:=both}"
: "${GLMOCR_MAX_WORKERS:=16}"
: "${GLMOCR_CONNECTION_POOL_SIZE:=128}"
: "${GLMOCR_PAGE_MAXSIZE:=100}"
: "${GLMOCR_REGION_MAXSIZE:=800}"
: "${GLMOCR_MAX_TOKENS_PER_PAGE:=4096}"
: "${GLMOCR_TEMPERATURE:=0.01}"
: "${GLMOCR_TOP_P:=0.9}"
: "${GLMOCR_DEFAULT_PROMPT:=Recognize all text, formulas, and tables in the document.}"

: "${GLMOCR_CONNECT_TIMEOUT:=30}"
: "${GLMOCR_REQUEST_TIMEOUT:=180}"
: "${GLMOCR_RETRY_MAX_ATTEMPTS:=3}"
: "${GLMOCR_RETRY_BACKOFF_BASE_SECONDS:=0.75}"

: "${GLMOCR_LAYOUT_MODEL_DIR:=PaddlePaddle/PP-DocLayoutV3_safetensors}"
: "${GLMOCR_LAYOUT_THRESHOLD:=0.3}"
: "${GLMOCR_LAYOUT_BATCH_SIZE:=1}"
: "${GLMOCR_LAYOUT_WORKERS:=1}"
: "${GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES:=0}"

export MODEL_NAME MODEL_REVISION SERVED_MODEL_NAME VLLM_HOST VLLM_PORT VLLM_DTYPE TRUST_REMOTE_CODE
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

# Generate GLM-OCR SDK config for full layout parsing.
cat > "${GLMOCR_CONFIG_PATH}" <<YAML
pipeline:
  maas:
    enabled: false
  ocr_api:
    api_host: 127.0.0.1
    api_port: ${VLLM_PORT}
    model: ${SERVED_MODEL_NAME}
    api_mode: openai
    connect_timeout: ${GLMOCR_CONNECT_TIMEOUT}
    request_timeout: ${GLMOCR_REQUEST_TIMEOUT}
    retry_max_attempts: ${GLMOCR_RETRY_MAX_ATTEMPTS}
    retry_backoff_base_seconds: ${GLMOCR_RETRY_BACKOFF_BASE_SECONDS}
    retry_status_codes: [429, 500, 502, 503, 504]
    connection_pool_size: ${GLMOCR_CONNECTION_POOL_SIZE}
    max_workers: ${GLMOCR_MAX_WORKERS}
    page_maxsize: ${GLMOCR_PAGE_MAXSIZE}
    region_maxsize: ${GLMOCR_REGION_MAXSIZE}
  page_loader:
    max_tokens: ${GLMOCR_MAX_TOKENS_PER_PAGE}
    temperature: ${GLMOCR_TEMPERATURE}
    top_p: ${GLMOCR_TOP_P}
    image_format: JPEG
    max_pixels: 71372800
  result_formatter:
    output_format: ${GLMOCR_OUTPUT_FORMAT}
  task_prompt_mapping:
    text: "Text Recognition:"
    formula: "Formula Recognition:"
    table: "Table Recognition:"
  default_prompt: "${GLMOCR_DEFAULT_PROMPT}"
  enable_layout: ${GLMOCR_ENABLE_LAYOUT}
  layout:
    model_dir: ${GLMOCR_LAYOUT_MODEL_DIR}
    threshold: ${GLMOCR_LAYOUT_THRESHOLD}
    batch_size: ${GLMOCR_LAYOUT_BATCH_SIZE}
    workers: ${GLMOCR_LAYOUT_WORKERS}
    cuda_visible_devices: "${GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES}"
YAML

echo "[start.sh] Wrote GLM-OCR config: ${GLMOCR_CONFIG_PATH}"

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

if [ "${TRUST_REMOTE_CODE,,}" = "true" ]; then
  vllm_args+=(--trust-remote-code)
fi

vllm "${vllm_args[@]}" &
VLLM_PID=$!

HEALTH_URL="http://127.0.0.1:${VLLM_PORT}/health"
echo "[start.sh] Waiting for health: ${HEALTH_URL}"
for i in $(seq 1 "${VLLM_HEALTH_TIMEOUT}"); do
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
