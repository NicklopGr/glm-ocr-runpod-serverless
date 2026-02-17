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
ENV GLMOCR_REF=${GLMOCR_REF}
ENV VLLM_BASE_IMAGE_REF=${VLLM_BASE_IMAGE}
ENV VENV_PATH=/opt/venv
ENV PATH=${VENV_PATH}/bin:${PATH}

# Validate that the base runtime has native GLM-OCR support and compatible
# critical dependency edges before installing app-level packages.
RUN python3 - <<'PY'
import importlib.metadata as md
from pathlib import Path

import vllm
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

runtime = {
    canonicalize_name("vllm"): md.version("vllm"),
    canonicalize_name("transformers"): md.version("transformers"),
    canonicalize_name("tokenizers"): md.version("tokenizers"),
    canonicalize_name("huggingface_hub"): md.version("huggingface_hub"),
    canonicalize_name("tqdm"): md.version("tqdm"),
}
print("[compat] global runtime versions:")
for name, version in sorted(runtime.items()):
    print(f"  - {name}=={version}")

edges = {
    "vllm": {"transformers", "tokenizers"},
    "transformers": {"huggingface_hub", "tokenizers", "tqdm"},
}
errors = []
env = default_environment()
env["extra"] = ""

for src, deps in edges.items():
    for raw in (md.requires(src) or []):
        req = Requirement(raw)
        dep = canonicalize_name(req.name)
        if dep not in deps:
            continue
        if req.marker and not req.marker.evaluate(env):
            continue
        if dep not in runtime:
            errors.append(f"{src} requires {dep} but {dep} is not installed")
            continue
        if req.specifier and not req.specifier.contains(runtime[dep], prereleases=True):
            errors.append(
                f"{src} requires {dep}{req.specifier}, installed {dep}=={runtime[dep]}"
            )

if errors:
    raise SystemExit("[compat] critical dependency mismatches:\n- " + "\n- ".join(errors))

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

print("[compat] native GLM-OCR and GLM-OCR MTP support verified")
PY

RUN python3 -m venv ${VENV_PATH}

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt && \
    python -m pip install "https://github.com/zai-org/GLM-OCR/archive/${GLMOCR_REF}.zip" && \
    python -m pip install -r /tmp/requirements.txt && \
    python -m pip check

COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENTRYPOINT []
CMD ["/app/start.sh"]
