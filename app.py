"""
OlmOCR API Server
Provides OCR capabilities using the allenai/olmOCR-2-7B-1025-FP8 model
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import io
import re
import torch
import logging
import traceback
import base64
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OlmOCR API",
    description="OCR API powered by allenai/olmOCR-2-7B-1025-FP8",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None
device = None

# Supported file types
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def get_ocr_prompt():
    """Get the official olmOCR v2 prompt (build_no_anchoring_v4_yaml_prompt).
    Uses the olmocr package if available, otherwise uses the exact prompt text
    from olmocr/prompts/prompts.py so no dependency on olmocr is needed."""
    try:
        from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
        return build_no_anchoring_v4_yaml_prompt()
    except ImportError:
        # Exact prompt from olmocr v0.4.25 - olmocr/prompts/prompts.py
        return (
            "Attached is one page of a document that you must process. "
            "Just return the plain text representation of this document as if you were reading it naturally. "
            "Convert equations to LateX and tables to HTML.\n"
            "If there are any figures or charts, label them with the following markdown syntax "
            "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
            "Return your output as markdown, with a front matter section on top specifying values for "
            "the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
        )


def render_pdf_page_to_base64(pdf_bytes, page_num=1, target_longest_dim=1288):
    """Render a PDF page to base64 PNG. Uses olmocr if available, otherwise falls back to pdf2image."""
    try:
        from olmocr.data.renderpdf import render_pdf_to_base64png
        # Write PDF bytes to a temp file since render_pdf_to_base64png expects a file path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        try:
            return render_pdf_to_base64png(tmp_path, page_num, target_longest_image_dim=target_longest_dim)
        finally:
            os.unlink(tmp_path)
    except ImportError:
        logger.info("olmocr.data.renderpdf not available, using pdf2image fallback")
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_bytes, first_page=page_num, last_page=page_num, dpi=200)
        if not images:
            raise ValueError(f"Could not render PDF page {page_num}")
        img = images[0]
        # Resize to target dimension
        max_dim = max(img.size)
        if max_dim > target_longest_dim:
            scale = target_longest_dim / max_dim
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


def image_to_base64(image: Image.Image, target_longest_dim=1288) -> str:
    """Convert a PIL Image to base64 PNG, resizing if needed."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    max_dim = max(image.size)
    if max_dim > target_longest_dim:
        scale = target_longest_dim / max_dim
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def parse_model_output(raw_text: str) -> dict:
    """Parse the model output, stripping YAML frontmatter and markdown fences."""
    text = raw_text.strip()

    # Remove outer markdown code fence if present (```markdown ... ```)
    fence_match = re.match(r"^```(?:markdown)?\s*\n?(.*?)```\s*$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    metadata = {}
    body = text

    # Extract YAML frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if fm_match:
        frontmatter = fm_match.group(1)
        body = fm_match.group(2).strip()
        for line in frontmatter.strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip()

    return {"text": body, "metadata": metadata}


def run_inference(image_base64: str) -> str:
    """Run the model on a base64-encoded image and return raw text output."""
    prompt = get_ocr_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(io.BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=0.8,
            max_new_tokens=4096,
            num_return_sequences=1,
            do_sample=True,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    decoded = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return decoded[0]


@app.on_event("startup")
async def load_model():
    """Load the OlmOCR model on startup"""
    global model, processor, device

    try:
        logger.info("Loading OlmOCR model...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model_name = "allenai/olmOCR-2-7B-1025-FP8"

        logger.info("Loading processor from Qwen/Qwen2.5-VL-7B-Instruct...")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        logger.info(f"Loading model from {model_name}...")
        if device.type == "cuda":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            logger.warning("No GPU detected! Running on CPU will be very slow.")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model.to(device)

        model.eval()
        logger.info("Model loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "running",
        "model": "allenai/olmOCR-2-7B-1025-FP8",
        "device": str(device),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or processor is None:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
    }


@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    """
    Perform OCR on an uploaded image or PDF.

    Accepts: JPEG, PNG, BMP, TIFF, WebP images and PDF files.
    For PDFs, processes the first page. Use /ocr/pdf for multi-page.

    Returns JSON with extracted text.
    """
    try:
        logger.info(f"Received file: {file.filename}")
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes")

        ext = os.path.splitext(file.filename or "")[1].lower()

        # Determine if it's a PDF or image
        if ext in PDF_EXTENSIONS or (file.content_type and "pdf" in file.content_type):
            logger.info("Processing as PDF (page 1)...")
            image_base64 = render_pdf_page_to_base64(contents, page_num=1)
        elif ext in IMAGE_EXTENSIONS or not ext:
            logger.info("Processing as image...")
            image = Image.open(io.BytesIO(contents))
            logger.info(f"Image size: {image.size}, mode: {image.mode}")
            image_base64 = image_to_base64(image)
        else:
            return JSONResponse(
                status_code=400,
                content={"text": "", "success": False, "message": f"Unsupported file type: {ext}"},
            )

        logger.info("Running inference...")
        raw_output = run_inference(image_base64)
        parsed = parse_model_output(raw_output)

        logger.info(f"OCR complete, extracted {len(parsed['text'])} chars")

        return {
            "text": parsed["text"],
            "metadata": parsed["metadata"],
            "success": True,
            "message": "OCR completed successfully",
        }

    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"text": "", "success": False, "message": f"OCR failed: {str(e)}"},
        )


@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...), pages: str = "1"):
    """
    Perform OCR on a PDF file, processing specified pages.

    Args:
        file: PDF file
        pages: Comma-separated page numbers or range (e.g. "1,2,3" or "1-5"). Default: "1"

    Returns JSON with per-page OCR results.
    """
    try:
        logger.info(f"Received PDF: {file.filename}, pages={pages}")
        contents = await file.read()

        # Parse page numbers
        page_nums = []
        for part in pages.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                page_nums.extend(range(int(start), int(end) + 1))
            else:
                page_nums.append(int(part))

        results = []
        for page_num in page_nums:
            try:
                logger.info(f"Processing page {page_num}...")
                image_base64 = render_pdf_page_to_base64(contents, page_num=page_num)
                raw_output = run_inference(image_base64)
                parsed = parse_model_output(raw_output)
                results.append({
                    "page": page_num,
                    "text": parsed["text"],
                    "metadata": parsed["metadata"],
                    "success": True,
                })
            except Exception as page_err:
                logger.error(f"Error on page {page_num}: {page_err}")
                results.append({
                    "page": page_num,
                    "text": "",
                    "success": False,
                    "error": str(page_err),
                })

        return {"results": results, "total_pages": len(page_nums)}

    except Exception as e:
        logger.error(f"PDF OCR error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"text": "", "success": False, "message": f"PDF OCR failed: {str(e)}"},
        )


@app.post("/ocr/batch")
async def batch_ocr(files: list[UploadFile] = File(...)):
    """
    Perform OCR on multiple files (images or PDFs, first page only).

    Returns list of OCR results.
    """
    results = []

    for file in files:
        try:
            contents = await file.read()
            ext = os.path.splitext(file.filename or "")[1].lower()

            if ext in PDF_EXTENSIONS or (file.content_type and "pdf" in file.content_type):
                image_base64 = render_pdf_page_to_base64(contents, page_num=1)
            else:
                image = Image.open(io.BytesIO(contents))
                image_base64 = image_to_base64(image)

            raw_output = run_inference(image_base64)
            parsed = parse_model_output(raw_output)

            results.append({
                "filename": file.filename,
                "text": parsed["text"],
                "metadata": parsed["metadata"],
                "success": True,
            })

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "text": "",
                "success": False,
                "error": str(e),
            })

    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
