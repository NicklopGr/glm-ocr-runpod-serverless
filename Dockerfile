# GLM-OCR RunPod serverless worker:
# - Process 1: local vLLM OpenAI server (zai-org/GLM-OCR)
# - Process 2: RunPod handler that runs full layout parsing via glmocr SDK
#
# Pin to an immutable image digest for reproducibility.
# NOTE: v0.15.x does not ship native GLM-OCR support. This pinned nightly
# commit includes native GlmOcrForConditionalGeneration + MTP support.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai:nightly-d00df624f313a6a5a7a6245b71448b068b080cd7@sha256:3f5ad92f63e3f4b0073cca935a976d4cbf2a21e22cf06a95fd6df47759e10e04
FROM ${VLLM_BASE_IMAGE}
ARG VLLM_BASE_IMAGE

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG GLMOCR_REF=529a0c7ee9aecf55095e6fa6d9da08e4bb3bc2a9
ARG TRANSFORMERS_VERSION=5.2.0
ARG TOKENIZERS_VERSION=0.22.2
ARG HUGGINGFACE_HUB_VERSION=1.4.1
ARG TQDM_VERSION=4.67.1

ENV GLMOCR_REF=${GLMOCR_REF}
ENV TRANSFORMERS_VERSION=${TRANSFORMERS_VERSION}
ENV TOKENIZERS_VERSION=${TOKENIZERS_VERSION}
ENV HUGGINGFACE_HUB_VERSION=${HUGGINGFACE_HUB_VERSION}
ENV TQDM_VERSION=${TQDM_VERSION}
ENV VLLM_BASE_IMAGE_REF=${VLLM_BASE_IMAGE}

COPY requirements.txt /tmp/requirements.txt

# Verify the pinned vLLM base actually contains native GLM-OCR support.
RUN python3 - <<'PY'
from pathlib import Path

import vllm

models_dir = Path(vllm.__file__).resolve().parent / "model_executor" / "models"
native_glm = models_dir / "glm_ocr.py"
native_glm_mtp = models_dir / "glm_ocr_mtp.py"
registry_py = models_dir / "registry.py"

if not native_glm.exists():
    raise SystemExit("[compat] missing native GLM-OCR model file: " + str(native_glm))
if not native_glm_mtp.exists():
    raise SystemExit("[compat] missing native GLM-OCR MTP model file: " + str(native_glm_mtp))
if "GlmOcrForConditionalGeneration" not in registry_py.read_text(encoding="utf-8"):
    raise SystemExit("[compat] GLM-OCR registry entry missing in " + str(registry_py))

print("[compat] native GLM-OCR and GLM-OCR MTP support verified in base image")
PY

# IMPORTANT: install runtime deps in the GLOBAL environment.
# vLLM is launched via /usr/local/bin/vllm (global interpreter), so splitting
# deps across a venv and global Python causes model-config mismatch at startup.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade \
      "transformers==${TRANSFORMERS_VERSION}" \
      "tokenizers==${TOKENIZERS_VERSION}" \
      "huggingface_hub==${HUGGINGFACE_HUB_VERSION}" \
      "tqdm==${TQDM_VERSION}" && \
    python3 -m pip install --upgrade --ignore-installed "blinker==1.9.0" && \
    python3 -m pip install -r /tmp/requirements.txt && \
    python3 -m pip install "https://github.com/zai-org/GLM-OCR/archive/${GLMOCR_REF}.zip" && \
    python3 -m pip check

# Validate the final runtime for GLM-OCR startup and handler imports.
RUN python3 - <<'PY'
import importlib.metadata as md
from pathlib import Path

import vllm
from packaging.version import Version
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

for pkg in ["vllm", "transformers", "tokenizers", "huggingface_hub", "tqdm", "glmocr"]:
    print(f"[runtime] {pkg}=={md.version(pkg)}")

if Version(md.version("transformers")) < Version("5.1.0"):
    raise SystemExit("[compat] transformers must be >=5.1.0 for glm_ocr support")

if "glm_ocr" not in CONFIG_MAPPING_NAMES:
    raise SystemExit("[compat] installed transformers does not expose model type glm_ocr")

models_dir = Path(vllm.__file__).resolve().parent / "model_executor" / "models"
registry_py = models_dir / "registry.py"
if "GlmOcrForConditionalGeneration" not in registry_py.read_text(encoding="utf-8"):
    raise SystemExit("[compat] GLM-OCR registry entry missing in " + str(registry_py))

print("[compat] shared global runtime validated for GLM-OCR")
PY

COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENTRYPOINT []
CMD ["/app/start.sh"]
