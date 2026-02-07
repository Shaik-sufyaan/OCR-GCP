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
import torch
import logging
import traceback
import base64

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
    version="1.0.0"
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


# Default prompt for OCR (simplified version of build_no_anchoring_v4_yaml_prompt)
DEFAULT_PROMPT = """Please provide a high-quality, structured OCR transcription of the document in this image.

Guidelines:
- Preserve all text exactly as it appears
- Maintain document structure (headings, paragraphs, lists, tables)
- Use markdown formatting where appropriate
- Preserve equations in LaTeX format
- Maintain natural reading order

Document transcription:"""


@app.on_event("startup")
async def load_model():
    """Load the OlmOCR model on startup"""
    global model, processor, device
    
    try:
        logger.info("Loading OlmOCR model...")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load processor and model
        model_name = "allenai/olmOCR-2-7B-1025-FP8"
        
        logger.info(f"Loading processor from Qwen/Qwen2.5-VL-7B-Instruct...")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
        logger.info(f"Loading model from {model_name}...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
        
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
        "device": str(device)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or processor is None:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }


@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    """
    Perform OCR on uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with extracted text
    """
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Read image contents
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes")
        
        # Open and convert image
        try:
            image = Image.open(io.BytesIO(contents))
            logger.info(f"Image opened. Size: {image.size}, Mode: {image.mode}")
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                logger.info(f"Converting from {image.mode} to RGB")
                image = image.convert("RGB")
            
            # Resize if needed (longest dimension should be ~1288 pixels)
            max_dim = max(image.size)
            if max_dim > 1288:
                scale = 1288 / max_dim
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized to: {image.size}")
                
        except Exception as img_error:
            logger.error(f"Image error: {str(img_error)}")
            return JSONResponse(
                status_code=400,
                content={
                    "text": "",
                    "success": False,
                    "message": f"Invalid image: {str(img_error)}"
                }
            )
        
        # Convert image to base64 for the message format
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Build message in the format expected by Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DEFAULT_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ]
        
        logger.info("Processing with model...")
        
        # Prepare inputs using the processor's apply_chat_template
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logger.info("Inputs prepared successfully")
            
        except Exception as proc_error:
            logger.error(f"Processor error: {str(proc_error)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "text": "",
                    "success": False,
                    "message": f"Processing failed: {str(proc_error)}"
                }
            )
        
        # Generate text
        try:
            logger.info("Generating text...")
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=None,
                    top_p=None
                )
            logger.info("Generation complete")
            
        except Exception as gen_error:
            logger.error(f"Generation error: {str(gen_error)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "text": "",
                    "success": False,
                    "message": f"Generation failed: {str(gen_error)}"
                }
            )
        
        # Decode output
        try:
            # Get only the newly generated tokens (skip the input)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            generated_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Successfully extracted text (length: {len(generated_text)})")
            
        except Exception as decode_error:
            logger.error(f"Decode error: {str(decode_error)}")
            return JSONResponse(
                status_code=500,
                content={
                    "text": "",
                    "success": False,
                    "message": f"Decoding failed: {str(decode_error)}"
                }
            )
        
        return {
            "text": generated_text.strip(),
            "success": True,
            "message": "OCR completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "text": "",
                "success": False,
                "message": f"OCR failed: {str(e)}"
            }
        )


@app.post("/ocr/batch")
async def batch_ocr(files: list[UploadFile] = File(...)):
    """
    Perform OCR on multiple images
    
    Args:
        files: List of image files
    
    Returns:
        List of OCR results
    """
    results = []
    
    for file in files:
        try:
            # Process each file using the same logic as single OCR
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize if needed
            max_dim = max(image.size)
            if max_dim > 1288:
                scale = 1288 / max_dim
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": DEFAULT_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ]
            
            # Process
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            
            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            generated_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            results.append({
                "filename": file.filename,
                "text": generated_text.strip(),
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "text": "",
                "success": False,
                "error": str(e)
            })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 