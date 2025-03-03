import boto3
import sagemaker
from fastapi import FastAPI, UploadFile, File, HTTPException,Form
from typing import List
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import yaml
from pathlib import Path
import shutil
import subprocess
import asyncio
from asyncio import StreamReader
import signal
from dotenv import load_dotenv
import re
import time
import GPUtil
import psutil

load_dotenv()

app = FastAPI()

# nothing

class TrainingProgress:
    def __init__(self):
        self.current_progress = 0.0
        self.status = "idle"
        self.current_lora_path = None
        self.s3_upload_path = None
        self.model_config = {
            "model_type": None,
            "base_model": None,
            "learning_rate": None,
            "optimizer": None,
            "batch_size": None
        }
        
        self.training_details = {
            "start_time": None,
            "end_time": None,
            "total_epochs": 0,
            "current_epoch": 0,
            "total_steps": 0,
            "current_step": 0,
            "current_loss": None,
            "loss_history": [],
            "epoch_progress": 0.0,
            "overall_progress": 0.0,
            "estimated_time_remaining": None,
            "formatted_time_remaining": None,
            "elapsed_time": None,
            "formatted_elapsed_time": None,
            "validation_metrics": {
                "validation_loss": None,
                "accuracy": None,
                "f1_score": None,
                "precision": None,
                "recall": None
            },
            "gpu_memory_usage": [],
            "cpu_utilization": [],
            "training_duration": None,
            "formatted_duration": None,
            "error": None,
            "error_details": None,
            "debug_logs": []
        }

    def format_time(self, seconds):
        """Format time in seconds to HH:MM:SS"""
        if seconds is None:
            return None
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update_time_metrics(self):
        """Update all time-related metrics"""
        current_time = time.time()
        
        # Update elapsed time
        if self.training_details['start_time']:
            elapsed_seconds = current_time - self.training_details['start_time']
            self.training_details['elapsed_time'] = elapsed_seconds
            self.training_details['formatted_elapsed_time'] = self.format_time(elapsed_seconds)
        
        # Update training duration
        if self.training_details['start_time'] and self.training_details['end_time']:
            duration = self.training_details['end_time'] - self.training_details['start_time']
            self.training_details['training_duration'] = duration
            self.training_details['formatted_duration'] = self.format_time(duration)
        
        # Update remaining time estimate
        if self.training_details['estimated_time_remaining']:
            self.training_details['formatted_time_remaining'] = self.format_time(
                self.training_details['estimated_time_remaining']
            )

    def update_resource_metrics(self):
        """Update GPU and CPU utilization metrics"""
        try:
            # GPU metrics
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assuming using first GPU
                self.training_details["gpu_memory_usage"].append({
                    "timestamp": time.time(),
                    "used_memory": gpu.memoryUsed,
                    "total_memory": gpu.memoryTotal,
                    "utilization": gpu.memoryUtil * 100
                })
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            self.training_details["cpu_utilization"].append({
                "timestamp": time.time(),
                "utilization": cpu_percent
            })
            
            # Limit the size of metrics arrays to prevent memory issues
            max_metrics_length = 100
            if len(self.training_details["gpu_memory_usage"]) > max_metrics_length:
                self.training_details["gpu_memory_usage"] = self.training_details["gpu_memory_usage"][-max_metrics_length:]
            if len(self.training_details["cpu_utilization"]) > max_metrics_length:
                self.training_details["cpu_utilization"] = self.training_details["cpu_utilization"][-max_metrics_length:]
                
        except Exception as e:
            self.training_details["debug_logs"].append(f"Error updating resource metrics: {str(e)}")

    def update_progress(self, current_step: int = None, total_steps: int = None, 
                       current_epoch: int = None, total_epochs: int = None):
        """Update training progress percentages and time estimates"""
        try:
            if current_step is not None and total_steps is not None:
                self.training_details['current_step'] = current_step
                self.training_details['total_steps'] = total_steps
                self.training_details['epoch_progress'] = (current_step / total_steps) * 100

            if current_epoch is not None and total_epochs is not None:
                self.training_details['current_epoch'] = current_epoch
                self.training_details['total_epochs'] = total_epochs

            # Calculate overall progress
            if self.training_details['total_epochs'] > 0:
                epoch_fraction = self.training_details['epoch_progress'] / 100
                completed_epochs = self.training_details['current_epoch'] - 1
                self.training_details['overall_progress'] = (
                    (completed_epochs + epoch_fraction) / 
                    self.training_details['total_epochs'] * 100
                )
                self.current_progress = round(self.training_details['overall_progress'], 2)

            # Calculate time estimates
            if self.training_details['start_time'] and self.training_details['overall_progress'] > 0:
                elapsed_time = time.time() - self.training_details['start_time']
                progress_fraction = self.training_details['overall_progress'] / 100
                if progress_fraction > 0:
                    total_estimated_time = elapsed_time / progress_fraction
                    remaining_time = total_estimated_time - elapsed_time
                    self.training_details['estimated_time_remaining'] = remaining_time
                    
            # Update all time-related formatting
            self.update_time_metrics()

        except Exception as e:
            error_msg = f"Error updating progress: {str(e)}"
            print(error_msg)
            self.training_details['debug_logs'].append(error_msg)

training_state = TrainingProgress()

def get_or_create_default_bucket():
    try:
        sagemaker_session = sagemaker.Session()
        bucket = sagemaker_session.default_bucket()
        return bucket
    except Exception as e:
        print(f"Error getting default bucket: {e}")
        return None



def generate_caption(image_path: str) -> str:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", 
            torch_dtype=torch_dtype, 
            trust_remote_code=True
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", 
            trust_remote_code=True
        )
        
        image = Image.open(image_path).convert("RGB")
        prompt = "<DETAILED_CAPTION>"
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        
        # Use model's generate method directly
        generated_ids = model.generate(
            input_ids=inputs["input_ids"], 
            pixel_values=inputs["pixel_values"], 
            max_new_tokens=1024, 
            num_beams=3
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        
            
        torch.cuda.empty_cache()
        return caption_text
    
    except Exception as e:
        print(f"Caption generation error: {str(e)}")
        return "A detailed image description"

def prepare_training_data(image_paths: List[str], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    config_path = "config/train_lora_flux.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    trigger_word = config.get('config', {}).get('trigger_word', '')

    
    for img_path in image_paths:
        filename = Path(img_path).name
        new_img_path = os.path.join(save_dir, filename)
        shutil.copy2(img_path, new_img_path)
        caption = generate_caption(img_path)

        clean_caption = caption.replace(f"[{trigger_word}]", "").strip()
        formatted_caption = f"[{trigger_word}] {clean_caption}" if trigger_word else clean_caption
        
        
            
        txt_path = os.path.join(save_dir, Path(filename).stem + '.txt')
        with open(txt_path, 'w') as f:
            f.write(formatted_caption)

def update_yaml_config(training_data_path: str, trigger_word: str, training_steps: int):
    config_path = "config/train_lora_flux.yaml"
    with open(config_path, 'r') as f:
        config_content = f.read()

    config_content = config_content.replace('${trigger_word}', trigger_word)
    config = yaml.safe_load(config_content)

    config['config']['name'] = trigger_word
    config['config']['process'][0]['datasets'][0]['folder_path'] = training_data_path
    config['config']['trigger_word'] = trigger_word
    config['config']['training_folder'] = f"output/{trigger_word}"

    config['config']['process'][0]['train']['steps'] = training_steps
    

    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def parse_training_output(line: str, training_state: TrainingProgress):
    """Enhanced parsing of training output with improved regex patterns"""
    try:
        # Parse epoch information
        epoch_match = re.search(r'Epoch[:\s]+(\d+)(?:/|\s+of\s+)(\d+)', line, re.IGNORECASE)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            training_state.update_progress(current_epoch=current_epoch, total_epochs=total_epochs)
        
        # Parse step progress
        step_match = re.search(r'(?:Step|Iteration)[:\s]+(\d+)(?:/|\s+of\s+)(\d+)', line, re.IGNORECASE)
        if step_match:
            current_step = int(step_match.group(1))
            total_steps = int(step_match.group(2))
            training_state.update_progress(current_step=current_step, total_steps=total_steps)
        
        # Parse loss values with better pattern matching
        loss_match = re.search(r'(?:Loss|Training Loss)[:\s]+([\d.]+)', line, re.IGNORECASE)
        if loss_match:
            current_loss = float(loss_match.group(1))
            training_state.training_details['current_loss'] = current_loss
            training_state.training_details['loss_history'].append({
                "step": training_state.training_details['current_step'],
                "loss": current_loss,
                "timestamp": time.time()
            })
        
        # Parse learning rate
        lr_match = re.search(r'Learning Rate[:\s]+([\d.e-]+)', line, re.IGNORECASE)
        if lr_match:
            training_state.model_config['learning_rate'] = float(lr_match.group(1))
        
        # Parse validation metrics
        val_loss_match = re.search(r'Validation Loss[:\s]+([\d.]+)', line, re.IGNORECASE)
        if val_loss_match:
            training_state.training_details['validation_metrics']['validation_loss'] = float(val_loss_match.group(1))
        
        # Update resource metrics periodically
        if not hasattr(parse_training_output, 'last_resource_update') or \
           time.time() - parse_training_output.last_resource_update > 30:
            training_state.update_resource_metrics()
            parse_training_output.last_resource_update = time.time()
        
    except Exception as e:
        error_msg = f"Error parsing training output: {str(e)}"
        print(error_msg)
        training_state.training_details['debug_logs'].append(error_msg)


async def read_stream(stream: StreamReader, callback) -> None:
    """Read from stream line by line until EOF, calling callback for each line"""
    while True:
        line = await stream.readline()
        if not line:
            break
        callback(line.decode().strip())
        
async def run_training(config_path: str):
    training_state.status = "training"
    training_state.current_progress = 0
    training_state.training_details['start_time'] = time.time()
    training_state.training_details['error'] = None
    
    try:
        # Parse training configuration 
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        training_folder = config.get('config', {}).get('training_folder', 'output/fieldpics')
        project_name = config.get('config', {}).get('name', 'unnamed_project')
        
        # Create process with managed subprocess
        process = await asyncio.create_subprocess_shell(
            f"python run.py {config_path}", 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=os.setsid,  # Create new process group
            limit=1024*1024*8  
        )
        
        # Setup concurrent stream readers
        stdout_reader = asyncio.create_task(
            read_stream(process.stdout, lambda line: parse_training_output(line, training_state))
        )
        stderr_reader = asyncio.create_task(
            read_stream(process.stderr, lambda line: print(f"Error: {line}"))
        )
        
        # Wait for process and stream readers to complete
        try:
            await asyncio.gather(
                process.wait(),
                stdout_reader,
                stderr_reader
            )
        except asyncio.CancelledError:
            # Ensure clean process termination
            if process.returncode is None:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            raise
        
        if process.returncode != 0:
            raise Exception(f"Training failed with return code {process.returncode}")
        
        # Find and upload latest model file
        latest_model = None
        max_step = -1
        
        for root, dirs, files in os.walk(training_folder):
            for file in files:
                if file.endswith('.safetensors'):
                    try:
                        step = int(file.split('_')[-1].split('.')[0])
                        if step > max_step:
                            max_step = step
                            latest_model = os.path.join(root, file)
                    except (IndexError, ValueError):
                        continue
        
        if latest_model:
            bucket = get_or_create_default_bucket()
            if bucket:
                s3_client = boto3.client('s3')
                file_name = Path(latest_model).name
                s3_key = f"trained_models/{project_name}/{file_name}"
                
                s3_client.upload_file(latest_model, bucket, s3_key)
                s3_url = s3_client.generate_presigned_url(
                    'get_object', 
                    Params={'Bucket': bucket, 'Key': s3_key}, 
                    ExpiresIn=3600
                )
                
                training_state.current_lora_path = latest_model
                training_state.s3_upload_path = s3_url
        
        training_state.status = "completed"
        training_state.current_progress = 100
        training_state.training_details['end_time'] = time.time()
        
    except Exception as e:
        training_state.status = "failed"
        training_state.training_details['error'] = str(e)
        training_state.training_details['end_time'] = time.time()
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure memory cleanup
        if 'process' in locals():
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except:
                pass
        torch.cuda.empty_cache()

@app.post("/upload-images")
async def upload_images(images: List[UploadFile] = File(...), trigger_word: str = Form(...),training_steps: int = Form(2000)):

    if training_steps<=0:
        raise HTTPException(
            status_code=400,
            detail="Training steps must be a positive integer"
        )
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    image_paths = []
    try:
        for img in images:
            path = os.path.join(temp_dir, img.filename)
            with open(path, "wb") as buffer:
                shutil.copyfileobj(img.file, buffer)
            image_paths.append(path)
        
        training_dir = "training_data"
        prepare_training_data(image_paths, training_dir)
        update_yaml_config(training_dir,trigger_word,training_steps)
        config_path = "config/train_lora_flux.yaml"
        asyncio.create_task(run_training(config_path))
        return {"message": "Upload successful, training started",
                "configuration":{
                    "trigger_word": trigger_word,
                    "training_steps": training_steps,
                    "number_of_images": len(image_paths)
                }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.get("/get-training-progress")
async def get_progress():
    """Enhanced progress endpoint with detailed metrics"""
    training_state.update_time_metrics()  # Ensure time metrics are up to date
    
    progress_info = {
        "progress": training_state.current_progress,
        "status": training_state.status,
        "time_metrics": {
            "elapsed_time": training_state.training_details['formatted_elapsed_time'],
            "estimated_time_remaining": training_state.training_details['formatted_time_remaining'],
            "training_duration": training_state.training_details['formatted_duration']
        },
        "training_metrics": {
            "current_epoch": training_state.training_details['current_epoch'],
            "total_epochs": training_state.training_details['total_epochs'],
            "current_step": training_state.training_details['current_step'],
            "total_steps": training_state.training_details['total_steps'],
            "epoch_progress": round(training_state.training_details['epoch_progress'], 2),
            "overall_progress": round(training_state.training_details['overall_progress'], 2),
            "current_loss": training_state.training_details['current_loss']
        },
        "resource_metrics": {
            "gpu_memory": training_state.training_details['gpu_memory_usage'][-1] if training_state.training_details['gpu_memory_usage'] else None,
            "cpu_utilization": training_state.training_details['cpu_utilization'][-1] if training_state.training_details['cpu_utilization'] else None
        },
        "validation_metrics": training_state.training_details['validation_metrics'],
        "error": training_state.training_details['error']
    }
    
    return progress_info

@app.get("/get-lora-file")
async def get_lora_file():
    if training_state.status == "idle":
        raise HTTPException(status_code=400, detail="No training started")
    
    if training_state.status == "training":
        return {
            "status": "in_progress",
            "progress": training_state.current_progress,
            "details": training_state.training_details
        }
    
    if training_state.status == "failed":
        raise HTTPException(
            status_code=500, 
            detail={
                "status": "failed",
                "error": training_state.training_details.get('error', 'Unknown error')
            }
        )
    
    if training_state.s3_upload_path is None:
        raise HTTPException(status_code=404, detail="LoRA file not found")
    
    return {
        "lora_local_path": training_state.current_lora_path, 
        "lora_s3_url": training_state.s3_upload_path,
        "training_details": training_state.training_details
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
