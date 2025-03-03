from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import torch
from diffusers import DiffusionPipeline
import boto3
import os
import google.generativeai as genai
from PIL import Image
from typing import List
import re
from pydantic import BaseModel, field_validator
import numpy as np
import cv2
import io
import base64
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Initialize FastAPI app
app = FastAPI(title="Flux Image Generation API")

# Configure AWS
s3_client = boto3.client('s3')

# Configure Google Gemini
genai.configure(api_key='AIzaSyAJsRbTRUcNAlYR2SWrR0qp0aX1j4M2TO8')
model = genai.GenerativeModel('gemini-nano')

# RealESRGAN models configurations
models = {
    'realesrgan-x4plus': {
        'model_path': 'weights/RealESRGAN_x4plus.pth',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'arch': RRDBNet,
        'arch_args': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'num_grow_ch': 32,
            'scale': 4
        }
    },
    'realesrgan-x4plus-anime': {
        'model_path': 'weights/RealESRGAN_x4plus_anime_6B.pth',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
        'arch': SRVGGNetCompact,
        'arch_args': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_conv': 16,
            'upscale': 4,
            'act_type': 'prelu'
        }
    }
}

class GenerationRequest(BaseModel):
    prompt: str
    aspect_ratio: str 
    num_images: int = 1
    lora_urls: List[str]
    upscale_model: str = 'realesrgan-x4plus'  # Default model
    upscale_factor: int = 2  # Default upscale factor

    @field_validator('aspect_ratio')
    @classmethod  # Required for field_validator
    def validate_aspect_ratio(cls, v: str) -> str:
        # Check if the aspect ratio is in the correct format (e.g., "16:9", "4:3", etc.)
        if not re.match(r'^\d+:\d+$', v):
            raise ValueError('Aspect ratio must be in format "width:height" (e.g., "16:9")')
        return v
    
    @field_validator('num_images')
    @classmethod  # Required for field_validator
    def validate_num_images(cls, v: int) -> int:
        if v < 1:
            raise ValueError('Number of images must be at least 1')
        if v > 10:  # Optional: limit maximum number of images
            raise ValueError('Maximum 10 images per request')
        return v
    
    @field_validator('upscale_model')
    @classmethod
    def validate_upscale_model(cls, v: str) -> str:
        valid_models = ['realesrgan-x4plus', 'realesrgan-x4plus-anime']
        if v not in valid_models:
            raise ValueError(f'Upscale model must be one of: {", ".join(valid_models)}')
        return v
    
    @field_validator('upscale_factor')
    @classmethod
    def validate_upscale_factor(cls, v: int) -> int:
        if v < 1 or v > 4:
            raise ValueError('Upscale factor must be between 1 and 4')
        return v

def download_lora_from_s3(s3_url: str) -> str:
    """Download LoRA weights from S3 and return local path"""
    try:
        bucket_name = s3_url.split('/')[2]
        key = '/'.join(s3_url.split('/')[3:])
        local_path = f"/tmp/{key.split('/')[-1]}"
    
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, key, local_path)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"Successfully downloaded LoRA to {local_path}")
            return local_path
        else:
            raise Exception(f"LoRA file download failed or file is empty: {local_path}")
            
    except Exception as e:
        print(f"Error downloading LoRA: {str(e)}")
        raise

def enhance_prompt(prompt: str) -> str:
    """Enhance the prompt using Google Gemini"""
    response = model.generate_content(
        f"Enhance this image generation prompt to be more detailed and creative: {prompt}"
    )
    return response.text

def get_image_dimensions(aspect_ratio: str, base_size: int = 1024) -> tuple:
    """Convert aspect ratio string to dimensions while maintaining max dimension"""
    width_ratio, height_ratio = map(int, aspect_ratio.split(':'))
    
    # Calculate scaling factor to maintain max dimension
    if width_ratio >= height_ratio:
        width = base_size
        height = int((height_ratio / width_ratio) * base_size)
    else:
        height = base_size
        width = int((width_ratio / height_ratio) * base_size)
    
    # Ensure dimensions are even numbers for the model
    width = (width // 2) * 2
    height = (height // 2) * 2
    
    return width, height

def initialize_realesrgan(model_name='realesrgan-x4plus', tile=0, tile_pad=10, pre_pad=0):
    """Initialize RealESRGAN upscaler with specified model"""
    # Check if model exists
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported")
    
    model_info = models[model_name]
    
    # Check if model file exists
    os.makedirs('weights', exist_ok=True)
    if not os.path.isfile(model_info['model_path']):
        print(f"Downloading {model_name} model...")
        load_file_from_url(url=model_info['url'], model_dir='weights', progress=True)
    
    # Check for available devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model architecture
    model = model_info['arch'](**model_info['arch_args'])
    
    # Initialize RealESRGANer
    upscaler = RealESRGANer(
        scale=4,  # Models are 4x scale internally
        model_path=model_info['model_path'],
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        device=device,
        half=False if device == 'cpu' else True  # Use half precision if GPU is available
    )
    
    return upscaler

def upscale_image(image: Image.Image, model_name='realesrgan-x4plus', outscale=2) -> Image.Image:
    """Upscale image using RealESRGAN with the specified model"""
    try:
        # Convert PIL Image to numpy array
        img_np = np.array(image)
        
        # Initialize RealESRGAN upscaler
        upscaler = initialize_realesrgan(model_name=model_name, tile=512)  # Use tile size 512 for large images
        
        # Perform upscaling
        output, _ = upscaler.enhance(img_np, outscale=outscale)
        
        # Convert back to PIL Image
        return Image.fromarray(output)
    
    except Exception as e:
        print(f"Error in RealESRGAN upscaling: {str(e)}")
        # Fallback to CV2 upscaling if RealESRGAN fails
        print("Falling back to CV2 upscaling")
        img_np = np.array(image)
        outscale = int(outscale)  # Ensure outscale is an integer
        upscaled = cv2.resize(img_np, (image.width * outscale, image.height * outscale), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(upscaled)

@app.post("/generate")
async def generate_images(request: GenerationRequest):
    try:
        # Initialize the pipeline with Flux Schnell
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",  
            torch_dtype=torch.bfloat16
        ).to(device)

        loaded_loras = []
        # Load all LoRA weights
        for i, lora_url in enumerate(request.lora_urls):
            try:
                lora_path = download_lora_from_s3(lora_url)
                weight_name = f"lora_{i}" 
                pipe.load_lora_weights(lora_path, weight_name)
                loaded_loras.append(weight_name)
                print(f"LoRA {weight_name} loaded successfully")

            except Exception as e:
                print(f"Failed to load LoRA {lora_url}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"LoRA loading failed for {lora_url}"
                )

        if loaded_loras:
            pipe.set_adapters(loaded_loras)
            print(f"Activated LoRAs: {loaded_loras}")
            
        # Enhance the prompt
        enhanced_prompt = request.prompt
        # enhanced_prompt = enhance_prompt(request.prompt)
        
        # Get dimensions based on aspect ratio
        width, height = get_image_dimensions(request.aspect_ratio)
        
        # Generate images
        generator = torch.Generator(device="cpu").manual_seed(0)
        images = []
        
        for _ in range(request.num_images):
            generated_image = pipe(
                enhanced_prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=generator
            ).images[0]
            
            # Upscale the image using RealESRGAN
            upscaled_image = upscale_image(
                generated_image, 
                model_name=request.upscale_model,
                outscale=request.upscale_factor
            )
            
            # Convert to base64 for API response
            buffered = io.BytesIO()
            upscaled_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_str)
        
        return {
            "status": "success",
            "lora_status":{
                "loaded_loras": loaded_loras,
                "lora_count": len(loaded_loras)
            },
            "original_prompt": request.prompt,
            "enhanced_prompt": enhanced_prompt,
            "dimensions": {
                "width": width,
                "height": height
            },
            "upscale_info": {
                "model": request.upscale_model,
                "factor": request.upscale_factor
            },
            "images": images
        }
        
    except Exception as e:
        print(f"Error in generate_images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)