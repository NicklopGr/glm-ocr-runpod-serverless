# GLM-OCR RunPod serverless worker:
# - Process 1: local vLLM OpenAI server (zai-org/GLM-OCR)
# - Process 2: RunPod handler that runs full layout parsing via glmocr SDK
#
# Pin to an immutable image digest for reproducibility.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai@sha256:2a503ea85ae35f6d556cbb12309c628a0a02af85a3f3c527ad4c0c7788553b92
FROM ${VLLM_BASE_IMAGE}
ARG VLLM_BASE_IMAGE

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG GLMOCR_REF=529a0c7ee9aecf55095e6fa6d9da08e4bb3bc2a9
ARG TRANSFORMERS_REF=372c27e71f80e64571ac1149d1708e641d7d44da
ARG MISTRAL_COMMON_VERSION=1.8.6
ENV GLMOCR_REF=${GLMOCR_REF}
ENV TRANSFORMERS_REF=${TRANSFORMERS_REF}
ENV MISTRAL_COMMON_VERSION=${MISTRAL_COMMON_VERSION}
ENV VLLM_BASE_IMAGE_REF=${VLLM_BASE_IMAGE}
ENV VENV_PATH=/opt/venv
ENV PATH=${VENV_PATH}/bin:${PATH}

# Install a GLM-OCR-capable Transformers build in the same Python environment
# used by the global `vllm` binary from the base image.
RUN python3 -m pip install --upgrade \
      "https://github.com/huggingface/transformers/archive/${TRANSFORMERS_REF}.zip"

# Align tokenizer dependency expected by current Transformers GLM/Mistral stack.
RUN python3 -m pip install --upgrade \
      "mistral-common==${MISTRAL_COMMON_VERSION}"

# vLLM 0.11.x expects `all_special_tokens_extended`, removed in Transformers v5.
# Patch vLLM tokenizer helper with a fallback to `all_special_tokens`.
RUN python3 - <<'PY'
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/tokenizer.py")
src = p.read_text(encoding="utf-8")
old = "tokenizer_all_special_tokens_extended = tokenizer.all_special_tokens_extended"
new = (
    "tokenizer_all_special_tokens_extended = "
    "getattr(tokenizer, 'all_special_tokens_extended', None)\n"
    "    if tokenizer_all_special_tokens_extended is None:\n"
    "        tokenizer_all_special_tokens_extended = tokenizer.all_special_tokens"
)

if old in src:
    src = src.replace(old, new, 1)
    p.write_text(src, encoding="utf-8")
    print("[patch] Applied vLLM tokenizer compatibility patch for Transformers v5")
else:
    print("[patch] vLLM tokenizer patch not needed (pattern missing)")
PY

# Transformers v5 may default to fast image processors that are incompatible
# with GLM-OCR multimodal token estimation in this vLLM build.
RUN python3 - <<'PY'
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/processor.py")
src = p.read_text(encoding="utf-8")
orig = src

marker1 = '"""Load an image processor for the given model name via HuggingFace."""\n'
inject = marker1 + '    kwargs.setdefault("use_fast", False)\n'
if marker1 in src and 'kwargs.setdefault("use_fast", False)' not in src:
    src = src.replace(marker1, inject, 1)

marker2 = '"""Load a processor for the given model name via HuggingFace."""\n'
if marker2 in src and src.count('kwargs.setdefault("use_fast", False)') < 2:
    src = src.replace(marker2, marker2 + '    kwargs.setdefault("use_fast", False)\n', 1)

if src != orig:
    p.write_text(src, encoding="utf-8")
    print("[patch] Applied vLLM processor patch: force use_fast=False")
else:
    print("[patch] vLLM processor patch not needed (pattern missing/already patched)")
PY

RUN python3 -m venv ${VENV_PATH}

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt && \
    python -m pip install "https://github.com/zai-org/GLM-OCR/archive/${GLMOCR_REF}.zip"

COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENTRYPOINT []
CMD ["/app/start.sh"]
