from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, field_validator
import torch
from diffusers import DiffusionPipeline
import boto3
import os
import google.generativeai as genai
from PIL import Image
from typing import List, Optional
import re
import numpy as np
import io
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Flux Image Generation API")

# Configure AWS
s3_client = boto3.client('s3')

# Configure Google Gemini
genai.configure(api_key='AIzaSyAJsRbTRUcNAlYR2SWrR0qp0aX1j4M2TO8')
gemini_model = genai.GenerativeModel('gemini-nano')

# Global model initialization
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if MODEL_DEVICE == "cuda" else torch.float32
BASE_MODEL_NAME = "black-forest-labs/FLUX.1-schnell"

# Initialize base model once at startup
@app.on_event("startup")
async def load_base_model():
    logger.info("Loading base model...")
    try:
        app.state.pipe = DiffusionPipeline.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=MODEL_DTYPE,
            # variant="fp16" if MODEL_DEVICE == "cuda" else None
        ).to(MODEL_DEVICE)
        logger.info("Base model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load base model: {str(e)}")
        raise RuntimeError("Could not initialize base model")

# Thread pool for parallel image generation
app.state.executor = ThreadPoolExecutor(max_workers=4)

class GenerationRequest(BaseModel):
    prompt: str
    aspect_ratio: str 
    num_images: int = 1
    lora_urls: List[str] = []
    seed: Optional[int] = None
    guidance_scale: float = 3.5
    num_inference_steps: int = 50

    @field_validator('aspect_ratio')
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        if not re.match(r'^\d+:\d+$', v):
            raise ValueError('Aspect ratio must be in format "width:height" (e.g., "16:9")')
        return v
    
    @field_validator('num_images')
    @classmethod
    def validate_num_images(cls, v: int) -> int:
        if not 1 <= v <= 4:
            raise ValueError('Number of images must be between 1 and 4')
        return v
    
    @field_validator('guidance_scale')
    @classmethod
    def validate_guidance_scale(cls, v: float) -> float:
        if not 1.0 <= v <= 10.0:
            raise ValueError('Guidance scale must be between 1.0 and 10.0')
        return v
    
    @field_validator('num_inference_steps')
    @classmethod
    def validate_inference_steps(cls, v: int) -> int:
        if not 10 <= v <= 100:
            raise ValueError('Inference steps must be between 10 and 100')
        return v

def download_lora_from_s3(s3_url: str) -> str:
    """Download and cache LoRA weights from S3"""
    try:
        bucket_name = s3_url.split('/')[2]
        key = '/'.join(s3_url.split('/')[3:])
        local_path = f"/tmp/{key.split('/')[-1]}"
        
        if os.path.exists(local_path):
            logger.info(f"Using cached LoRA: {local_path}")
            return local_path
            
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, key, local_path)
        if os.path.getsize(local_path) > 0:
            logger.info(f"Downloaded LoRA to {local_path}")
            return local_path
        raise Exception("Empty LoRA file")
    except Exception as e:
        logger.error(f"LoRA download failed: {str(e)}")
        raise

async def load_lora_adapters(lora_urls: List[str], pipe: DiffusionPipeline):
    """Load multiple LoRA adapters dynamically"""
    loaded_adapters = []
    for i, url in enumerate(lora_urls):
        try:
            lora_path = await asyncio.to_thread(download_lora_from_s3, url)
            adapter_name = f"adapter_{i}_{os.path.basename(lora_path)}"
            
            if adapter_name in pipe.get_active_adapters():
                logger.info(f"Adapter {adapter_name} already loaded")
                continue
                
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            loaded_adapters.append(adapter_name)
            logger.info(f"Loaded LoRA adapter: {adapter_name}")
        except Exception as e:
            logger.error(f"Failed to load LoRA {url}: {str(e)}")
            raise
    return loaded_adapters

def generate_single_image(args):
    """Wrapper function for parallel execution"""
    pipe, prompt, width, height, seed, guidance, steps = args
    generator = torch.Generator(device=MODEL_DEVICE)
    if seed is not None:
        generator.manual_seed(seed)
        
    return pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator
    ).images[0]

async def generate_images_parallel(pipe, prompt, dimensions, num_images, seed, guidance, steps):
    """Generate multiple images in parallel"""
    loop = asyncio.get_running_loop()
    args = [(pipe, prompt, *dimensions, seed + i if seed else None, guidance, steps) 
            for i in range(num_images)]
    
    futures = [
        loop.run_in_executor(
            app.state.executor,
            generate_single_image,
            (app.state.pipe, prompt, *dimensions, seed + i if seed else None, guidance, steps)
        )
        for i in range(num_images)
    ]
    
    return await asyncio.gather(*futures)

def get_image_dimensions(aspect_ratio: str) -> tuple:
    """Calculate dimensions with safe defaults"""
    width_ratio, height_ratio = map(int, aspect_ratio.split(':'))
    base_size = 1024
    scale = base_size / max(width_ratio, height_ratio)
    width = int(width_ratio * scale) // 8 * 8
    height = int(height_ratio * scale) // 8 * 8
    return width, height

@app.post("/generate")
async def generate_images(request: GenerationRequest, background_tasks: BackgroundTasks):
    try:
        start_time = time.time()
        pipe = app.state.pipe
        
        # Load LoRA adapters
        if request.lora_urls:
            loaded_adapters = await load_lora_adapters(request.lora_urls, pipe)
            pipe.set_adapters(loaded_adapters)
            logger.info(f"Active adapters: {loaded_adapters}")
        else:
            pipe.disable_lora()
            
        # Enhanced prompt (temporarily disabled)
        # enhanced_prompt = gemini_model.generate_content(f"Enhance this prompt: {request.prompt}").text
        enhanced_prompt = request.prompt
        
        # Get dimensions
        width, height = get_image_dimensions(request.aspect_ratio)
        
        # Generate images in parallel
        images = await generate_images_parallel(
            pipe=pipe,
            prompt=enhanced_prompt,
            dimensions=(width, height),
            num_images=request.num_images,
            seed=request.seed,
            guidance=request.guidance_scale,
            steps=request.num_inference_steps
        )
        
        # Convert images to base64
        encoded_images = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            encoded_images.append(base64.b64encode(buffered.getvalue()).decode())
        
        # Cleanup LoRA adapters
        if request.lora_urls:
            background_tasks.add_task(pipe.disable_lora)
            
        logger.info(f"Generated {request.num_images} images in {time.time()-start_time:.2f}s")
        return {
            "status": "success",
            "dimensions": {"width": width, "height": height},
            "images": encoded_images,
            "prompt_info": {
                "original": request.prompt,
                "enhanced": enhanced_prompt
            }
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, 'pipe'),
        "device": MODEL_DEVICE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)