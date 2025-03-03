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

# check
# Initialize FastAPI app
app = FastAPI(title="Flux Image Generation API")

# Configure AWS
s3_client = boto3.client('s3')

# Configure Google Gemini
genai.configure(api_key='key')
model = genai.GenerativeModel('gemini-1.5-pro')

class GenerationRequest(BaseModel):
    prompt: str
    aspect_ratio: str 
    num_images: int = 1
    lora_urls: List[str]  

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
            
        
    except Exeption as e:
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

def upscale_image(image: Image.Image, scale: int = 2) -> Image.Image:
    img_np = np.array(image)  # Convert PIL to NumPy
    upscaled = cv2.resize(img_np, (image.width * scale, image.height * scale), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(upscaled)  # Convert back to PIL

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
        enhanced_prompt = enhance_prompt(request.prompt)
        #enhanced_prompt = request.prompt
        
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
            
            # Upscale the image
            upscaled_image = upscale_image(generated_image)
            
            # Convert to base64 for API response
            import io
            import base64
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
