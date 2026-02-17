# GLM-OCR RunPod serverless worker:
# - Process 1: local vLLM OpenAI server (zai-org/GLM-OCR)
# - Process 2: RunPod handler that runs full layout parsing via glmocr SDK
#
# Pin to an immutable image digest for reproducibility.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai@sha256:2a503ea85ae35f6d556cbb12309c628a0a02af85a3f3c527ad4c0c7788553b92
FROM ${VLLM_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG GLMOCR_REF=529a0c7ee9aecf55095e6fa6d9da08e4bb3bc2a9
ARG TRANSFORMERS_REF=372c27e71f80e64571ac1149d1708e641d7d44da
ENV GLMOCR_REF=${GLMOCR_REF}
ENV TRANSFORMERS_REF=${TRANSFORMERS_REF}
ENV VLLM_BASE_IMAGE_REF=${VLLM_BASE_IMAGE}
ENV VENV_PATH=/opt/venv
ENV PATH=${VENV_PATH}/bin:${PATH}

# Install a GLM-OCR-capable Transformers build in the same Python environment
# used by the global `vllm` binary from the base image.
RUN /usr/local/bin/python3 -m pip install --upgrade \
      "git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_REF}"

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
