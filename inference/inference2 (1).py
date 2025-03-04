from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import torch
from diffusers import DiffusionPipeline
import boto3
import os
from typing import List
import re
import io
import base64
from contextlib import asynccontextmanager
from functools import lru_cache
import logging
import time
import uuid
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_client = boto3.client('s3')

genai.configure(api_key='your_key')
model = genai.GenerativeModel('gemini-1.5-pro')


def enhance_prompt(prompt: str) -> str:
    """Enhance the prompt using Google Gemini"""
    response = model.generate_content(
        f"Expand and refine the user prompt into a highly detailed, cinematic description suitable for an image generation model. The final prompt must vividly describe the scene, background, lighting, atmosphere, and key details to guide the model effectively. Ensure that the provided trigger word is naturally incorporated based on its category. If a focal length is given, adjust the perspective accordingly. The final output must remain within 77 tokens while being visually compelling and well-structured. Avoid redundancy, and optimize for clarity and realism: {prompt}"
    )
    return response.text

# Define default LoRA configuration
DEFAULT_LORA_URL = "s3://sagemaker-us-east-1-274412008471/trained_models/Default-Lora/Realism Lora.safetensors"
DEFAULT_LORA_SCALE = 0.6  # Adjust scale as needed for your use case

# Lifespan context manager to load the base model at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the base Flux model at startup and clean up on shutdown."""
    logger.info("Starting server and loading base model...")
    start_time = time.time()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        app.state.pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        ).to(device)
        
        # Preload default LoRA at startup to improve performance
        try:
            default_lora_path = get_lora_path(DEFAULT_LORA_URL)
            app.state.default_lora_path = default_lora_path
            logger.info(f"Default LoRA loaded from: {default_lora_path}")
        except Exception as e:
            logger.warning(f"Could not preload default LoRA: {str(e)}")
            app.state.default_lora_path = None
            
        duration = time.time() - start_time
        logger.info(f"Base model loaded successfully in {duration:.2f} seconds.")
        yield
    except Exception as e:
        logger.error(f"Failed to load base model: {str(e)}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down server and cleaning up...")
        if hasattr(app.state, 'pipe'):
            del app.state.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

app = FastAPI(title="Flux Image Generation API", lifespan=lifespan)

# Updated GenerationRequest model with option to disable default LoRA
class GenerationRequest(BaseModel):
    prompt: str
    aspect_ratio: str
    num_images: int = 1
    lora_urls: List[str] = []
    use_default_lora: bool = True
    enhance_prompt: bool = True# New flag to control default LoRA usage

    @field_validator('aspect_ratio')
    def validate_aspect_ratio(cls, v: str) -> str:
        if not re.match(r'^\d+:\d+$', v):
            raise ValueError('Aspect ratio must be in format "width:height" (e.g., "16:9")')
        return v

    @field_validator('num_images')
    def validate_num_images(cls, v: int) -> int:
        if v < 1:
            raise ValueError('Number of images must be at least 1')
        if v > 10:
            raise ValueError('Maximum 10 images per request')
        return v

def download_lora_from_s3(s3_url: str) -> str:
    """Download LoRA weights from S3 and return local path."""
    try:
        bucket_name = s3_url.split('/')[2]
        key = '/'.join(s3_url.split('/')[3:])
        local_path = f"/tmp/{key.split('/')[-1]}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, key, local_path)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            logger.info(f"Successfully downloaded LoRA to {local_path}")
            return local_path
        else:
            raise Exception(f"LoRA file download failed or file is empty: {local_path}")
    except Exception as e:
        logger.error(f"Error downloading LoRA from {s3_url}: {str(e)}")
        raise

@lru_cache(maxsize=10)
def get_lora_path(s3_url: str) -> str:
    """Cache and return the local path of downloaded LoRA weights."""
    return download_lora_from_s3(s3_url)

def get_image_dimensions(aspect_ratio: str, base_size: int = 1024) -> tuple:
    """Convert aspect ratio string to dimensions while maintaining max dimension."""
    width_ratio, height_ratio = map(int, aspect_ratio.split(':'))
    if width_ratio >= height_ratio:
        width = base_size
        height = int((height_ratio / width_ratio) * base_size)
    else:
        height = base_size
        width = int((width_ratio / height_ratio) * base_size)
    # Ensure dimensions are even
    width = (width // 2) * 2
    height = (height // 2) * 2
    return width, height

@app.post("/generate")
async def generate_images(request: GenerationRequest):
    """Generate images using the pre-loaded Flux model with LoRA adapters."""
    try:
        start_time = time.time()
        pipe = app.state.pipe

        if request.enhance_prompt:
            try:
                original_prompt=request.prompt
                enhanced_prompt=enhance_prompt(original_prompt)
                logger.info(f"Enhanced prompt from: '{original_prompt}' to: '{enhanced_prompt}'")
                processed_prompt = enhanced_prompt
            except Exception as e:
                logger.error(f"Failed to enhance prompt: {str(e)}. Using original prompt.")
                processed_prompt = request.prompt

        else:
            processed_prompt = request.prompt
            logger.info("Using original prompt without enhancement")

            
                
                
                

        # Generate unique request ID to use for adapter names
        request_id = str(uuid.uuid4())[:8]
        
        # Load and apply LoRA weights with unique adapter names
        loaded_adapters = []
        adapter_scales = {}
        
        # First load the default LoRA if enabled
        if request.use_default_lora:
            try:
                # Use the preloaded path if available, otherwise download
                if hasattr(app.state, 'default_lora_path') and app.state.default_lora_path:
                    default_lora_path = app.state.default_lora_path
                else:
                    default_lora_path = get_lora_path(DEFAULT_LORA_URL)
                
                default_adapter_name = f"default_lora_{request_id}"
                logger.info(f"Loading default LoRA from {default_lora_path} as {default_adapter_name}")
                pipe.load_lora_weights(default_lora_path, adapter_name=default_adapter_name)
                loaded_adapters.append(default_adapter_name)
                adapter_scales[default_adapter_name] = DEFAULT_LORA_SCALE
                logger.info(f"Default LoRA loaded with scale {DEFAULT_LORA_SCALE}")
            except Exception as e:
                logger.error(f"Failed to load default LoRA, continuing without it: {str(e)}")
        
        # Now load user-provided LoRAs
        for i, lora_url in enumerate(request.lora_urls):
            # Use unique adapter names for each request
            adapter_name = f"lora_{request_id}_{i}"
            lora_path = get_lora_path(lora_url)
            logger.info(f"Loading LoRA weights from {lora_path} as {adapter_name}")
            pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            loaded_adapters.append(adapter_name)
            adapter_scales[adapter_name] = 1.0  # Default scale for user LoRAs is 1.0

        # Set all adapters with their respective scales
        if loaded_adapters:
            pipe.set_adapters(loaded_adapters, adapter_weights=[adapter_scales.get(name, 1.0) for name in loaded_adapters])
            logger.info(f"Activated LoRA adapters: {loaded_adapters}")

        # Get image dimensions
        width, height = get_image_dimensions(request.aspect_ratio)

        # Generate images in batches for parallelism
        max_batch_size = 2  # Adjust based on GPU memory
        images = []
        generators = [torch.Generator(device="cpu").manual_seed(i) for i in range(request.num_images)]
        with torch.inference_mode():
            for i in range(0, request.num_images, max_batch_size):
                batch_size = min(max_batch_size, request.num_images - i)
                batch_prompts = [request.prompt] * batch_size
                batch_generators = generators[i:i + batch_size]
                logger.info(f"Generating batch of {batch_size} images...")
                batch_images = pipe(
                    batch_prompts,
                    height=height,
                    width=width,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    generator=batch_generators
                ).images
                images.extend(batch_images)

        # Convert images to base64
        encoded_images = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            encoded_images.append(img_str)

        # Clean up LoRA weights
        pipe.unload_lora_weights()
        
        # Explicitly remove adapters by name
        for adapter_name in loaded_adapters:
            if adapter_name in pipe.get_list_adapters():
                pipe.delete_adapter(adapter_name)
                logger.info(f"Deleted adapter: {adapter_name}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("LoRA weights unloaded and GPU memory cleared.")

        duration = time.time() - start_time
        logger.info(f"Generated {request.num_images} images in {duration:.2f} seconds")

        return {
            "status": "success",
            "prompt":{
                "original": request.prompt,
                "enhanced": processed_prompt if request.enhance_prompt else None,
                "enhancement_used": request.enhance_prompt
            },
            "lora_status": {
                "loaded_loras": loaded_adapters,
                "default_lora_used": request.use_default_lora,
                "lora_count": len(loaded_adapters)
            },
            "dimensions": {
                "width": width,
                "height": height
            },
            "images": encoded_images
        }

    except Exception as e:
        logger.error(f"Error in generate_images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint with GPU memory usage."""
    gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return {
        "status": "healthy",
        "gpu_memory": f"{gpu_memory:.2f} GB"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
