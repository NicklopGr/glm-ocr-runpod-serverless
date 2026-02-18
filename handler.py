import asyncio
import base64
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import runpod
from PIL import Image
from pypdf import PdfReader, PdfWriter


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _to_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


WORKER_MAX_CONCURRENCY = max(1, _env_int("WORKER_MAX_CONCURRENCY", 1))
DOWNLOAD_CONCURRENCY = max(1, _env_int("DOWNLOAD_CONCURRENCY", 16))
MAX_PAGES_PER_JOB = max(1, _env_int("MAX_PAGES_PER_JOB", 128))
IMAGE_MAX_SIDE = max(512, _env_int("IMAGE_MAX_SIDE", 2200))
IMAGE_JPEG_QUALITY = max(40, min(95, _env_int("IMAGE_JPEG_QUALITY", 90)))

GLMOCR_CONFIG_PATH = os.environ.get("GLMOCR_CONFIG_PATH", "/app/glmocr.config.yaml")
GLMOCR_PARSE_TIMEOUT_SECONDS = max(60, _env_int("GLMOCR_PARSE_TIMEOUT_SECONDS", 1800))
GLMOCR_LOG_LEVEL = os.environ.get("GLMOCR_LOG_LEVEL", "INFO")

RETURN_CROP_IMAGES_DEFAULT = _env_bool("RETURN_CROP_IMAGES_DEFAULT", False)
MAX_CROP_IMAGES_DEFAULT = max(0, _env_int("MAX_CROP_IMAGES", 200))

MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "glm-ocr")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "")
GLMOCR_OUTPUT_FORMAT = os.environ.get("GLMOCR_OUTPUT_FORMAT", "both")
GLMOCR_ENABLE_LAYOUT = _env_bool("GLMOCR_ENABLE_LAYOUT", True)
GLMOCR_REF = os.environ.get("GLMOCR_REF", "unknown")
VLLM_BASE_IMAGE_REF = os.environ.get("VLLM_BASE_IMAGE_REF", "unknown")


class InputError(Exception):
    pass


def _b64decode(payload: str) -> bytes:
    data = (payload or "").strip()
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    if not data:
        raise InputError("Empty base64 payload")
    return base64.b64decode(data)


def _is_pdf_bytes(data: bytes) -> bool:
    return data.startswith(b"%PDF")


def _url_looks_pdf(url: str) -> bool:
    return ".pdf" in url.lower().split("?", 1)[0]


def _image_bytes_to_jpeg(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
        return out.getvalue()


def _slice_pdf_bytes(pdf_bytes: bytes, start_page: int, end_page: Optional[int]) -> Tuple[bytes, int, int]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)

    start_idx = max(0, start_page - 1)
    end_idx = total_pages if end_page is None else min(total_pages, end_page)

    if start_idx >= end_idx:
        raise InputError(
            f"Invalid page range: start_page={start_page}, end_page={end_page}, total_pages={total_pages}"
        )

    writer = PdfWriter()
    for idx in range(start_idx, end_idx):
        writer.add_page(reader.pages[idx])

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue(), total_pages, (end_idx - start_idx)


def _write_images_as_pdf(image_bytes_list: List[bytes], pdf_path: Path) -> int:
    if not image_bytes_list:
        raise InputError("No images available to create PDF")

    pil_pages: List[Image.Image] = []
    try:
        for raw in image_bytes_list:
            with Image.open(io.BytesIO(raw)) as im:
                pil_pages.append(im.convert("RGB"))

        first, *rest = pil_pages
        first.save(str(pdf_path), format="PDF", save_all=True, append_images=rest)
        return len(pil_pages)
    finally:
        for page in pil_pages:
            page.close()


async def _download_urls(urls: List[str]) -> List[bytes]:
    if not urls:
        return []

    timeout = aiohttp.ClientTimeout(total=180)
    sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def _fetch(url: str) -> bytes:
            async with sem:
                async with session.get(url) as resp:
                    body = await resp.read()
                    if resp.status >= 400:
                        raise InputError(f"Failed to download URL ({resp.status}): {url}")
                    return body

        return await asyncio.gather(*[_fetch(url) for url in urls])


async def _collect_document(job_input: Dict[str, Any]) -> Tuple[Path, int, Dict[str, Any], Path]:
    start_page = max(1, int(job_input.get("start_page", 1)))
    end_page_raw = job_input.get("end_page")
    end_page = int(end_page_raw) if end_page_raw is not None else None

    temp_dir = Path(tempfile.mkdtemp(prefix="glmocr_job_", dir="/tmp"))
    input_meta: Dict[str, Any] = {
        "start_page": start_page,
        "end_page": end_page,
    }

    pdf_url = job_input.get("pdf_url")
    pdf_base64 = job_input.get("pdf_base64")
    image_url = job_input.get("image_url")
    image_base64 = job_input.get("image_base64")
    image_urls = list(job_input.get("image_urls") or ([] if image_url is None else [image_url]))
    images_base64 = list(job_input.get("images_base64") or ([] if image_base64 is None else [image_base64]))
    source_url = job_input.get("source_url")
    source_base64 = job_input.get("source_base64")

    pdf_bytes: Optional[bytes] = None
    image_bytes_list: List[bytes] = []

    if pdf_url or pdf_base64:
        if pdf_url and pdf_base64:
            raise InputError("Provide only one of pdf_url or pdf_base64")
        if pdf_url:
            pdf_bytes = (await _download_urls([pdf_url]))[0]
            input_meta["input_type"] = "pdf_url"
            input_meta["source"] = pdf_url
        else:
            pdf_bytes = _b64decode(pdf_base64)
            input_meta["input_type"] = "pdf_base64"
    elif image_urls or images_base64:
        if image_urls and images_base64:
            raise InputError("Use either image_urls or images_base64 in one request")
        if image_urls:
            input_meta["input_type"] = "image_urls"
            input_meta["source_count"] = len(image_urls)
            image_bytes_list = await _download_urls(image_urls)
        else:
            input_meta["input_type"] = "images_base64"
            input_meta["source_count"] = len(images_base64)
            image_bytes_list = [_b64decode(item) for item in images_base64]
    elif source_url or source_base64:
        if source_url and source_base64:
            raise InputError("Provide only one of source_url or source_base64")
        if source_url:
            raw = (await _download_urls([source_url]))[0]
            input_meta["input_type"] = "source_url"
            input_meta["source"] = source_url
            if _is_pdf_bytes(raw) or _url_looks_pdf(source_url):
                pdf_bytes = raw
                input_meta["resolved_type"] = "pdf"
            else:
                image_bytes_list = [raw]
                input_meta["resolved_type"] = "image"
        else:
            raw = _b64decode(source_base64)
            input_meta["input_type"] = "source_base64"
            if _is_pdf_bytes(raw):
                pdf_bytes = raw
                input_meta["resolved_type"] = "pdf"
            else:
                image_bytes_list = [raw]
                input_meta["resolved_type"] = "image"
    else:
        raise InputError(
            "No input provided. Use one of: pdf_url/pdf_base64, image_urls/images_base64, source_url/source_base64"
        )

    if pdf_bytes is not None:
        if not _is_pdf_bytes(pdf_bytes):
            raise InputError("Input was treated as PDF but payload is not a valid PDF header")

        sliced_pdf_bytes, total_pages, selected_pages = _slice_pdf_bytes(pdf_bytes, start_page=start_page, end_page=end_page)
        if selected_pages > MAX_PAGES_PER_JOB:
            raise InputError(
                f"This job has {selected_pages} pages but MAX_PAGES_PER_JOB={MAX_PAGES_PER_JOB}. "
                "Split into smaller requests or increase MAX_PAGES_PER_JOB."
            )

        input_path = temp_dir / "input.pdf"
        input_path.write_bytes(sliced_pdf_bytes)

        input_meta["total_pdf_pages"] = total_pages
        input_meta["selected_pages"] = selected_pages
        return input_path, selected_pages, input_meta, temp_dir

    # Image path flow.
    normalized = [_image_bytes_to_jpeg(raw) for raw in image_bytes_list]
    page_count = len(normalized)
    if page_count == 0:
        raise InputError("No image pages resolved from input")

    if page_count > MAX_PAGES_PER_JOB:
        raise InputError(
            f"This job has {page_count} pages but MAX_PAGES_PER_JOB={MAX_PAGES_PER_JOB}. "
            "Split into smaller requests or increase MAX_PAGES_PER_JOB."
        )

    if page_count == 1:
        input_path = temp_dir / "input.jpg"
        input_path.write_bytes(normalized[0])
    else:
        input_path = temp_dir / "input.pdf"
        _write_images_as_pdf(normalized, input_path)

    input_meta["selected_pages"] = page_count
    return input_path, page_count, input_meta, temp_dir


async def _run_glmocr_parse(input_path: Path, output_dir: Path) -> Dict[str, str]:
    cmd = [
        "glmocr",
        "parse",
        str(input_path),
        "--config",
        GLMOCR_CONFIG_PATH,
        "--output",
        str(output_dir),
        "--log-level",
        GLMOCR_LOG_LEVEL,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_raw, stderr_raw = await asyncio.wait_for(proc.communicate(), timeout=GLMOCR_PARSE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(
            f"glmocr parse timed out after {GLMOCR_PARSE_TIMEOUT_SECONDS}s (input={input_path.name})"
        )

    stdout = stdout_raw.decode("utf-8", errors="ignore")
    stderr = stderr_raw.decode("utf-8", errors="ignore")

    if proc.returncode != 0:
        raise RuntimeError(
            f"glmocr parse failed (code={proc.returncode}). "
            f"stderr_head={stderr[:800]} stderr_tail={stderr[-1500:]} "
            f"stdout_tail={stdout[-1500:]}"
        )

    return {
        "stdout_tail": stdout[-1500:],
        "stderr_tail": stderr[-1500:],
    }


def _find_result_paths(output_dir: Path) -> Dict[str, Optional[Path]]:
    json_candidates = sorted(output_dir.rglob("result.json"))
    if not json_candidates:
        # GLM-OCR SDK 0.1.1 saves "<doc_stem>.json" instead of "result.json".
        json_candidates = sorted(output_dir.rglob("*.json"))
    if not json_candidates:
        raise RuntimeError(f"No JSON result found under {output_dir}")

    def _rank(path: Path) -> tuple[int, str]:
        if path.name == "result.json":
            return (0, str(path))
        md_peer = path.with_suffix(".md")
        if md_peer.exists():
            return (1, str(path))
        return (2, str(path))

    result_json = sorted(json_candidates, key=_rank)[0]
    doc_dir = result_json.parent

    result_md: Optional[Path] = doc_dir / "result.md"
    if not result_md.exists():
        stem_md = doc_dir / f"{result_json.stem}.md"
        result_md = stem_md if stem_md.exists() else None

    imgs_dir = doc_dir / "imgs"
    if not imgs_dir.exists() or not imgs_dir.is_dir():
        imgs_dir = None

    return {
        "doc_dir": doc_dir,
        "result_json": result_json,
        "result_md": result_md,
        "imgs_dir": imgs_dir,
    }


def _collect_image_artifacts(
    images_dir: Optional[Path],
    include_base64: bool,
    max_items: int,
) -> List[Dict[str, Any]]:
    if images_dir is None or max_items == 0:
        return []

    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in valid_ext]

    out: List[Dict[str, Any]] = []
    for idx, path in enumerate(files):
        if idx >= max_items:
            break

        payload: Dict[str, Any] = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
        }

        if include_base64:
            payload["image_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")

        out.append(payload)

    return out


def _find_layout_visualizations(doc_dir: Path) -> List[Path]:
    valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    out: List[Path] = []
    search_dirs: List[Path] = [doc_dir, doc_dir / "layout_vis"]
    for base in search_dirs:
        if not base.exists() or not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if not path.is_file() or path.suffix.lower() not in valid_ext:
                continue
            name = path.name.lower()
            if "layout" in name or "visual" in name or base.name.lower() == "layout_vis":
                out.append(path)
    return out


async def _warmup() -> Dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(prefix="glmocr_warmup_", dir="/tmp"))
    try:
        input_path = temp_dir / "warmup.jpg"
        img = Image.new("RGB", (128, 128), color=(255, 255, 255))
        img.save(str(input_path), format="JPEG", quality=80)
        img.close()

        output_dir = temp_dir / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

        run_info = await _run_glmocr_parse(input_path=input_path, output_dir=output_dir)

        return {
            "status": "success",
            "provider": "glm-ocr-sdk-layout-vllm",
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "glmocr_ref": GLMOCR_REF,
            "vllm_base_image_ref": VLLM_BASE_IMAGE_REF,
            "enable_layout": GLMOCR_ENABLE_LAYOUT,
            "output_format": GLMOCR_OUTPUT_FORMAT,
            "glmocr_config_path": GLMOCR_CONFIG_PATH,
            "worker_max_concurrency": WORKER_MAX_CONCURRENCY,
            "warmup_run": run_info,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
            "provider": "glm-ocr-sdk-layout-vllm",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    start_ts = time.time()
    job_input = event.get("input", {}) or {}

    if event.get("warmup") or job_input.get("warmup"):
        return {
            "status": "success",
            "result": await _warmup(),
        }

    temp_dir: Optional[Path] = None

    try:
        return_crop_images = _to_bool(job_input.get("return_crop_images"), RETURN_CROP_IMAGES_DEFAULT)
        crop_images_base64 = _to_bool(job_input.get("crop_images_base64"), False)
        max_crop_images = max(0, int(job_input.get("max_crop_images", MAX_CROP_IMAGES_DEFAULT)))

        return_layout_visualization = _to_bool(job_input.get("return_layout_visualization"), False)
        layout_visualization_base64 = _to_bool(job_input.get("layout_visualization_base64"), False)

        input_path, page_count, input_meta, temp_dir = await _collect_document(job_input)
        output_dir = temp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        run_info = await _run_glmocr_parse(input_path=input_path, output_dir=output_dir)
        paths = _find_result_paths(output_dir)

        result_json_path = paths["result_json"]
        if result_json_path is None:
            raise RuntimeError("glmocr parse completed but no result.json found")

        result_json = json.loads(result_json_path.read_text(encoding="utf-8"))

        markdown_output = ""
        if paths["result_md"] is not None:
            markdown_output = paths["result_md"].read_text(encoding="utf-8")

        crop_images = []
        if return_crop_images:
            crop_images = _collect_image_artifacts(
                images_dir=paths["imgs_dir"],
                include_base64=crop_images_base64,
                max_items=max_crop_images,
            )

        layout_visualizations: List[Dict[str, Any]] = []
        if return_layout_visualization:
            vis_paths = _find_layout_visualizations(paths["doc_dir"])
            for vis in vis_paths:
                item: Dict[str, Any] = {
                    "filename": vis.name,
                    "size_bytes": vis.stat().st_size,
                }
                if layout_visualization_base64:
                    item["image_base64"] = base64.b64encode(vis.read_bytes()).decode("ascii")
                layout_visualizations.append(item)

        elapsed_ms = int((time.time() - start_ts) * 1000)

        return {
            "status": "success",
            "result": {
                "provider": "glm-ocr-sdk-layout-vllm",
                "model": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "glmocr_ref": GLMOCR_REF,
                "vllm_base_image_ref": VLLM_BASE_IMAGE_REF,
                "enable_layout": GLMOCR_ENABLE_LAYOUT,
                "output_format": GLMOCR_OUTPUT_FORMAT,
                "page_count": page_count,
                "processing_time_ms": elapsed_ms,
                "input": input_meta,
                "json_result": result_json,
                "markdown_result": markdown_output,
                "crop_images": crop_images,
                "layout_visualizations": layout_visualizations,
                "run_info": run_info,
            },
        }
    except InputError as exc:
        return {
            "status": "error",
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(exc),
        }
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


def concurrency_modifier(current_concurrency: int) -> int:
    _ = current_concurrency
    return WORKER_MAX_CONCURRENCY


if __name__ == "__main__":
    print(
        "[GLM-OCR] starting runpod handler",
        {
            "provider": "glm-ocr-sdk-layout-vllm",
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "glmocr_ref": GLMOCR_REF,
            "vllm_base_image_ref": VLLM_BASE_IMAGE_REF,
            "enable_layout": GLMOCR_ENABLE_LAYOUT,
            "output_format": GLMOCR_OUTPUT_FORMAT,
            "glmocr_config_path": GLMOCR_CONFIG_PATH,
            "worker_max_concurrency": WORKER_MAX_CONCURRENCY,
            "max_pages_per_job": MAX_PAGES_PER_JOB,
        },
    )

    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": concurrency_modifier,
        }
    )
