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
ARG HUGGINGFACE_HUB_VERSION=1.4.1
ARG TQDM_VERSION=4.67.1
ENV GLMOCR_REF=${GLMOCR_REF}
ENV TRANSFORMERS_REF=${TRANSFORMERS_REF}
ENV MISTRAL_COMMON_VERSION=${MISTRAL_COMMON_VERSION}
ENV HUGGINGFACE_HUB_VERSION=${HUGGINGFACE_HUB_VERSION}
ENV TQDM_VERSION=${TQDM_VERSION}
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

# Keep HuggingFace stack pinned to a vLLM 0.11.x-compatible baseline.
RUN python3 -m pip install --upgrade \
      "huggingface_hub==${HUGGINGFACE_HUB_VERSION}" \
      "tqdm==${TQDM_VERSION}"

# Verify declared compatibility directly from installed package metadata.
RUN python3 - <<'PY'
import importlib.metadata as md
from packaging.requirements import Requirement
from packaging.version import Version

transformers_ver = Version(md.version("transformers"))
hub_ver = Version(md.version("huggingface_hub"))
tqdm_ver = Version(md.version("tqdm"))

hub_req = None
for raw in (md.requires("transformers") or []):
    req = Requirement(raw)
    if req.name.replace("-", "_") == "huggingface_hub":
        hub_req = req
        break

if hub_req is not None and hub_ver not in hub_req.specifier:
    raise SystemExit(
        f"Incompatible pins: transformers=={transformers_ver} requires "
        f"{hub_req.name}{hub_req.specifier}, but huggingface_hub=={hub_ver}"
    )

print(
    f"[compat] transformers={transformers_ver}, "
    f"huggingface_hub={hub_ver}, tqdm={tqdm_ver}"
)
PY

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

# huggingface_hub may pass `disable=` into tqdm class constructors.
# vLLM's DisabledTqdm wrapper already forces disable=True, which can raise:
#   TypeError: ... got multiple values for keyword argument 'disable'
RUN python3 - <<'PY'
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py")
src = p.read_text(encoding="utf-8")
orig = src

needle = "super().__init__(*args, **kwargs, disable=True)"
patch = 'kwargs.pop("disable", None)\n        super().__init__(*args, **kwargs, disable=True)'

if 'kwargs.pop("disable", None)' in src:
    print("[patch] vLLM DisabledTqdm patch already applied")
elif needle in src:
    src = src.replace(needle, patch, 1)
    p.write_text(src, encoding="utf-8")
    print("[patch] Applied vLLM DisabledTqdm compatibility patch")
else:
    print("[patch] vLLM DisabledTqdm patch not needed (pattern missing)")
PY

# GLM-OCR checkpoints include extra MTP-only weights under
# `model.language_model.layers.16.*`. This vLLM base falls back to the
# generic Transformers backend, which is strict by default and crashes on
# those unexpected keys. Ignore that prefix only for model_type=glm_ocr.
RUN python3 - <<'PY'
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/transformers/base.py")
src = p.read_text(encoding="utf-8")
orig = src

needle = '                self.ignore_unexpected_suffixes.append(".bias")\n'
inject = (
    needle
    + "\n"
    + '        if getattr(self.config, "model_type", None) == "glm_ocr":\n'
    + "            self.ignore_unexpected_prefixes.extend([\n"
    + '                "model.language_model.layers.16",\n'
    + '                "model.language_model.layers.16.",\n'
    + "            ])\n"
)

if '"model.language_model.layers.16."' in src:
    print("[patch] vLLM GLM-OCR unexpected-weight ignore patch already applied")
elif needle in src:
    src = src.replace(needle, inject, 1)
    p.write_text(src, encoding="utf-8")
    print("[patch] Applied vLLM GLM-OCR unexpected-weight ignore patch")
else:
    print("[patch] vLLM GLM-OCR patch not needed (pattern missing)")
PY

# Fail fast on declared dependency incompatibilities in the global runtime
# environment used by `vllm`, and ensure GLM-OCR fallback compatibility is in
# place when native `glm_ocr` model support is absent in this vLLM build.
RUN python3 - <<'PY'
import importlib.metadata as md
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

watch = {
    canonicalize_name(name): name
    for name in [
        "vllm",
        "transformers",
        "huggingface_hub",
        "tokenizers",
        "tqdm",
        "mistral-common",
        "torch",
        "numpy",
        "requests",
        "pillow",
        "safetensors",
    ]
}

installed: dict[str, Version] = {}
for dist in md.distributions():
    name = canonicalize_name(dist.metadata.get("Name", ""))
    if name in watch:
        installed[name] = Version(dist.version)

print("[compat] key runtime versions:")
for name in sorted(installed):
    print(f"  - {watch[name]}=={installed[name]}")

checks: list[tuple[str, str, str, str, str]] = []
for src in ["vllm", "transformers", "huggingface_hub", "mistral-common"]:
    try:
        reqs = md.requires(src) or []
    except md.PackageNotFoundError:
        continue
    for raw in reqs:
        req = Requirement(raw)
        dep = canonicalize_name(req.name)
        if dep not in installed:
            continue

        env = default_environment()
        env["extra"] = ""
        if req.marker and not req.marker.evaluate(env):
            continue

        spec = str(req.specifier) if str(req.specifier) else "*"
        ok = installed[dep] in req.specifier if req.specifier else True
        if not ok:
            checks.append((src, watch.get(dep, dep), spec, str(installed[dep]), raw))

if checks:
    print("[compat] incompatible dependency pins detected:")
    for src, dep, spec, got, raw in checks:
        print(f"  - {src} requires {dep}{spec}, installed {dep}=={got} (from `{raw}`)")
    raise SystemExit(1)

models_dir = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models")
native_glm_ocr = (models_dir / "glm_ocr.py").exists()
if native_glm_ocr:
    print("[compat] native vLLM glm_ocr model support detected")
else:
    base_py = models_dir / "transformers" / "base.py"
    src = base_py.read_text(encoding="utf-8")
    marker = "model.language_model.layers.16"
    if marker not in src:
        raise SystemExit(
            "[compat] vLLM has no native glm_ocr support and fallback ignore "
            "patch is missing; GLM-OCR startup will fail on MTP-only weights"
        )
    print("[compat] no native glm_ocr model; fallback MTP-weight ignore patch verified")
PY

RUN python3 -m venv ${VENV_PATH}

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt && \
    python -m pip install "https://github.com/zai-org/GLM-OCR/archive/${GLMOCR_REF}.zip" && \
    python -m pip install --upgrade \
      "huggingface_hub==${HUGGINGFACE_HUB_VERSION}" \
      "tqdm==${TQDM_VERSION}" && \
    python -m pip check

COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENTRYPOINT []
CMD ["/app/start.sh"]
