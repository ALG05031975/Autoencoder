import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import io
import gc
from datetime import datetime
import json
import uuid

app = FastAPI(title="Autoencoder Defect Detection API")
from fastapi.middleware.cors import CORSMiddleware

UPLOAD_DIR = Path("api/uploads")
RESULT_DIR = Path("api/results")
MODEL_DIR = Path("api/models")
DEBUG_DIR = Path("api/debug_logs")

# Create directories for uploads and results
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
MODEL_DIR = Path("models")
DEBUG_DIR = Path("debug_logs")

for directory in [UPLOAD_DIR, RESULT_DIR, MODEL_DIR, DEBUG_DIR]:
    directory.mkdir(exist_ok=True)

# Mount static files for serving results
app.mount("/results", StaticFiles(directory="results"), name="results")

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class DetectionRequest(BaseModel):
    heatmap_threshold: int = 40
    min_defect_area: int = 1
    show_boxes: bool = True
    force_cpu: bool = False
    debug_mode: bool = False

class DetectionResult(BaseModel):
    status: str
    defect_count: int
    result_path: Optional[str] = None
    debug_info: Optional[dict] = None
    error: Optional[str] = None

# Global state for the model
global_state = {
    "model_loaded": False,
    "autoencoder": None,
    "device": None,
    "debug_mode": False,
    "last_debug_info": {}
}

def debug_dump(error_msg=""):
    debug_dir = Path("debug_logs")
    debug_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file = debug_dir / f"debug_{timestamp}.txt"
    
    debug_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": error_msg,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "debug_mode": global_state["debug_mode"],
    }
    
    try:
        with open(debug_file, "w") as f:
            for key, value in debug_info.items():
                f.write(f"{key}: {value}\n")
        
        global_state["last_debug_info"] = debug_info
        if global_state["debug_mode"]:
            print(f"Debug info saved to {debug_file}")
    except Exception as e:
        if global_state["debug_mode"]:
            print(f"Failed to save debug info: {e}")

def get_defects(image, heatmap_threshold, min_defect_area):
    try:
        if image is None:
            return [], np.zeros((1,1))
            
        h, w = image.shape[:2]
        
        resized = cv2.resize(image, (256, 256))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        tensor = tensor.to(global_state["device"])
        
        with torch.no_grad():
            recon = global_state["autoencoder"](tensor)
            diff = torch.abs(tensor - recon).mean(1).squeeze().cpu().numpy()
            heatmap = cv2.resize(diff, (w, h))
        
        _, mask = cv2.threshold((heatmap * 255).astype(np.uint8), heatmap_threshold, 255, cv2.THRESH_BINARY)
        
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_defect_area:
                x, y, w, h = cv2.boundingRect(cnt)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(image.shape[1], x+w), min(image.shape[0], y+h)
                boxes.append((x1, y1, x2, y2))
        
        if global_state["debug_mode"]:
            print(f"Processing complete. Found {len(boxes)} anomalies")
        
        return boxes, heatmap
    except Exception as e:
        error_msg = f"Processing error: {e}"
        if global_state["debug_mode"]:
            print(error_msg)
        return [], np.zeros((h,w) if 'h' in locals() else (1,1))

def visualize_results(image, boxes, show_boxes):
    if image is None:
        return np.zeros((100,100,3), dtype=np.uint8)
        
    vis_img = image.copy()
    h, w = image.shape[:2]
    
    base_width = 1000
    scale_factor = min(2.0, max(0.5, w / base_width))
    
    font_scale = 0.6 * scale_factor
    text_thickness = max(1, int(1.5 * scale_factor))
    line_thickness = max(1, int(2 * scale_factor))
    
    if show_boxes:
        for box in boxes:
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), line_thickness)
            
            text = "DEFECT"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            
            y_pos = max(text_height + 3, box[1] - 3)
            x_pos = max(0, min(box[0], w - text_width - 5))
            
            cv2.putText(vis_img, text, (x_pos, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                      (0, 0, 255), text_thickness, cv2.LINE_AA)
    
    status_text = "Status: DEFECT" if len(boxes) > 0 else "Status: NO-DEFECT"
    status_font_scale = 1.0 * scale_factor
    status_thickness = max(2, int(2.5 * scale_factor))
    
    cv2.putText(vis_img, status_text, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, 
               (0, 255, 0) if len(boxes) == 0 else (0, 0, 255), 
               status_thickness, cv2.LINE_AA)
    
    return vis_img

@app.post("/load_model/")
async def load_model(model_file: UploadFile = File(...), force_cpu: bool = False, debug_mode: bool = False):
    try:
        global_state["debug_mode"] = debug_mode
        model_path = MODEL_DIR / model_file.filename
        
        # Save the uploaded model file
        with open(model_path, "wb") as buffer:
            buffer.write(model_file.file.read())
        
        # Device selection
        global_state["device"] = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if global_state["debug_mode"]:
            print(f"Loading model on device: {global_state['device']}")
            print(f"Autoencoder: {model_path}")
        
        # Load autoencoder
        global_state["autoencoder"] = Autoencoder().to(global_state["device"])
        state_dict = torch.load(str(model_path), map_location=torch.device(global_state["device"]))
        global_state["autoencoder"].load_state_dict(state_dict)
        global_state["autoencoder"].eval()
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        global_state["model_loaded"] = True
        
        if global_state["debug_mode"]:
            print("Model loaded successfully")
        
        return JSONResponse(content={"status": "success", "message": "Model loaded successfully"})
    
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        debug_dump(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/detect_defects/")
async def detect_defects(
    image: UploadFile = File(...),
    heatmap_threshold: int = Form(40),
    min_defect_area: int = Form(1),
    show_boxes: bool = Form(True),
    debug_mode: bool = Form(False)
) -> DetectionResult:
    try:
        if not global_state["model_loaded"]:
            raise HTTPException(status_code=400, detail="Model not loaded. Please load a model first.")
        
        global_state["debug_mode"] = debug_mode
        
        # Read and process the image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        if global_state["debug_mode"]:
            print(f"Image loaded successfully. Shape: {original_image.shape}")
        
        # Process the image
        boxes, heatmap = get_defects(original_image, heatmap_threshold, min_defect_area)
        result_image = visualize_results(original_image, boxes, show_boxes)
        
        # Save the result
        result_id = str(uuid.uuid4())
        result_filename = f"result_{result_id}.png"
        result_path = RESULT_DIR / result_filename
        cv2.imwrite(str(result_path), result_image)
        
        status = "DEFECT" if len(boxes) > 0 else "NO-DEFECT"
        
        if global_state["debug_mode"]:
            print(f"Processing completed successfully. Status: {status}")
        
        return DetectionResult(
            status=status,
            defect_count=len(boxes),
            result_path=f"/results/{result_filename}",
            debug_info=global_state["last_debug_info"] if debug_mode else None
        )
    
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        debug_dump(error_msg)
        return DetectionResult(
            status="ERROR",
            defect_count=0,
            error=error_msg,
            debug_info=global_state["last_debug_info"] if debug_mode else None
        )

@app.get("/get_result/")
async def get_result(result_path: str):
    full_path = RESULT_DIR / Path(result_path).name
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(full_path)

@app.get("/get_debug_info/")
async def get_debug_info():
    if not global_state["last_debug_info"]:
        raise HTTPException(status_code=404, detail="No debug information available")
    return JSONResponse(content=global_state["last_debug_info"])

@app.get("/system_info/")
async def get_system_info():
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": global_state["model_loaded"],
        "device": global_state["device"] if "device" in global_state else None
    }
    return JSONResponse(content=info)

@app.get("/")
async def root():
    return {"message": "Autoencoder Defect Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)