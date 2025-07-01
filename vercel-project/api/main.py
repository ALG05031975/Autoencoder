import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import gc
from datetime import datetime
import uuid

app = FastAPI(title="Autoencoder Defect Detection API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
MODEL_DIR = Path("models")
DEBUG_DIR = Path("debug_logs")

for directory in [UPLOAD_DIR, RESULT_DIR, MODEL_DIR, DEBUG_DIR]:
    directory.mkdir(exist_ok=True)

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
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

# Global state
global_state = {
    "model_loaded": False,
    "autoencoder": None,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "debug_mode": False,
    "last_debug_info": {}
}

def debug_dump(error_msg=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file = DEBUG_DIR / f"debug_{timestamp}.txt"
    
    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "error": error_msg,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "device": str(global_state["device"]),
        "model_loaded": global_state["model_loaded"]
    }
    
    try:
        with open(debug_file, "w") as f:
            json.dump(debug_info, f, indent=2)
        global_state["last_debug_info"] = debug_info
    except Exception as e:
        if global_state["debug_mode"]:
            print(f"Failed to save debug info: {e}")

def get_defects(image, heatmap_threshold=40, min_defect_area=1):
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
                boxes.append((max(0, x), max(0, y), 
                            min(image.shape[1], x+w), min(image.shape[0], y+h)))
        
        return boxes, heatmap
    except Exception as e:
        error_msg = f"Processing error: {e}"
        debug_dump(error_msg)
        return [], np.zeros((h,w) if 'h' in locals() else (1,1))

@app.post("/api/load_model/")
async def load_model(
    model_file: UploadFile = File(...),
    force_cpu: bool = Form(False),
    debug_mode: bool = Form(False)
):
    try:
        global_state["debug_mode"] = debug_mode
        model_path = MODEL_DIR / model_file.filename
        
        with open(model_path, "wb") as buffer:
            buffer.write(await model_file.read())
        
        global_state["device"] = torch.device("cpu" if force_cpu else "cuda" if torch.cuda.is_available() else "cpu")
        
        global_state["autoencoder"] = Autoencoder().to(global_state["device"])
        state_dict = torch.load(str(model_path), map_location=global_state["device"])
        global_state["autoencoder"].load_state_dict(state_dict)
        global_state["autoencoder"].eval()
        global_state["model_loaded"] = True
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return JSONResponse(content={"status": "success", "message": "Model loaded successfully"})
    
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        debug_dump(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/detect_defects/")
async def detect_defects(
    image: UploadFile = File(...),
    heatmap_threshold: int = Form(40),
    min_defect_area: int = Form(1),
    show_boxes: bool = Form(True),
    debug_mode: bool = Form(False)
):
    try:
        if not global_state["model_loaded"]:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        global_state["debug_mode"] = debug_mode
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        boxes, heatmap = get_defects(original_image, heatmap_threshold, min_defect_area)
        
        # Visualization
        vis_img = original_image.copy()
        if show_boxes:
            for box in boxes:
                cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(vis_img, "DEFECT", (box[0], box[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        status = "DEFECT" if boxes else "NO-DEFECT"
        cv2.putText(vis_img, f"Status: {status}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (0,255,0) if status == "NO-DEFECT" else (0,0,255), 2)
        
        # Save result
        result_id = str(uuid.uuid4())
        result_filename = f"result_{result_id}.png"
        result_path = RESULT_DIR / result_filename
        cv2.imwrite(str(result_path), vis_img)
        
        return JSONResponse(content={
            "status": status,
            "defect_count": len(boxes),
            "result_path": f"/api/get_result/?result_path={result_filename}",
            "debug_info": global_state["last_debug_info"] if debug_mode else None
        })
    
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        debug_dump(error_msg)
        return JSONResponse(content={
            "status": "ERROR",
            "error": error_msg,
            "debug_info": global_state["last_debug_info"] if debug_mode else None
        }, status_code=500)

@app.get("/api/get_result/")
async def get_result(result_path: str):
    full_path = RESULT_DIR / Path(result_path).name
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(full_path)

@app.get("/api/health")
async def health_check():
    return JSONResponse(content={
        "status": "OK",
        "model_loaded": global_state["model_loaded"],
        "device": str(global_state["device"])
    })

@app.get("/api/system_info")
async def system_info():
    return JSONResponse(content={
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(global_state["device"])
    })