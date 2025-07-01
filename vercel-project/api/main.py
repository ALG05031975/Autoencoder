import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import gc
from datetime import datetime
import uuid
import json

app = FastAPI(title="Autoencoder Defect Detection API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory Setup (Vercel-compatible paths)
UPLOAD_DIR = Path("/tmp/uploads")  # Using Vercel's /tmp for persistence
RESULT_DIR = Path("/tmp/results")
MODEL_DIR = Path("/tmp/models")
DEBUG_DIR = Path("/tmp/debug_logs")

for directory in [UPLOAD_DIR, RESULT_DIR, MODEL_DIR, DEBUG_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

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

# Global state with Vercel-compatible initialization
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
        "system": {
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "device": str(global_state["device"])
        },
        "state": {
            "model_loaded": global_state["model_loaded"],
            "last_action": "debug_dump"
        }
    }
    
    try:
        with open(debug_file, "w") as f:
            json.dump(debug_info, f, indent=2)
        global_state["last_debug_info"] = debug_info
        return str(debug_file)
    except Exception as e:
        print(f"DEBUG_SAVE_ERROR: {str(e)}")
        return None

@app.post("/api/load_model")
async def load_model(
    model_file: UploadFile = File(...),
    force_cpu: bool = Form(False),
    debug_mode: bool = Form(False)
):
    try:
        global_state["debug_mode"] = debug_mode
        
        # Save model to temporary location
        model_path = MODEL_DIR / f"model_{uuid.uuid4().hex}.pth"
        with open(model_path, "wb") as buffer:
            buffer.write(await model_file.read())
        
        # Device configuration
        global_state["device"] = torch.device("cpu" if force_cpu else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Model loading
        global_state["autoencoder"] = Autoencoder().to(global_state["device"])
        state_dict = torch.load(model_path, map_location=global_state["device"])
        global_state["autoencoder"].load_state_dict(state_dict)
        global_state["autoencoder"].eval()
        global_state["model_loaded"] = True
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return JSONResponse({
            "status": "success",
            "message": f"Model loaded on {global_state['device']}",
            "model_size": f"{os.path.getsize(model_path)/1024/1024:.2f}MB"
        })
    
    except Exception as e:
        debug_path = debug_dump(f"Load Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "debug_file": debug_path,
                "device": str(global_state["device"])
            }
        )

@app.post("/api/detect")
async def detect_defects(
    image: UploadFile = File(...),
    heatmap_threshold: int = Form(40),
    min_defect_area: int = Form(10),
    debug_mode: bool = Form(False)
):
    try:
        if not global_state["model_loaded"]:
            raise HTTPException(400, "Model not loaded. Please load model first.")
            
        global_state["debug_mode"] = debug_mode
        
        # Image processing
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_img is None:
            raise HTTPException(400, "Invalid image file")
        
        # Detection logic
        h, w = original_img.shape[:2]
        resized = cv2.resize(original_img, (256, 256))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        tensor = tensor.to(global_state["device"])
        
        with torch.no_grad():
            recon = global_state["autoencoder"](tensor)
            diff = torch.abs(tensor - recon).mean(1).squeeze().cpu().numpy()
            heatmap = cv2.resize(diff, (w, h))
        
        _, mask = cv2.threshold((heatmap * 255).astype(np.uint8), heatmap_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defects = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_defect_area:
                x, y, w, h = cv2.boundingRect(cnt)
                defects.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": float(cv2.contourArea(cnt))
                })
        
        # Generate result image
        result_img = original_img.copy()
        for defect in defects:
            cv2.rectangle(result_img, 
                         (defect["x"], defect["y"]), 
                         (defect["x"]+defect["width"], defect["y"]+defect["height"]), 
                         (0, 0, 255), 2)
        
        # Save result
        result_id = uuid.uuid4().hex
        result_path = RESULT_DIR / f"result_{result_id}.jpg"
        cv2.imwrite(str(result_path), result_img)
        
        return JSONResponse({
            "status": "success",
            "defect_count": len(defects),
            "defects": defects,
            "result_url": f"/api/result/{result_id}",
            "heatmap_size": f"{heatmap.nbytes/1024:.2f}KB",
            "debug": global_state["last_debug_info"] if debug_mode else None
        })
        
    except Exception as e:
        debug_path = debug_dump(f"Detection Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "debug_file": debug_path,
                "input_size": f"{len(contents)/1024:.2f}KB" if 'contents' in locals() else None
            }
        )

@app.get("/api/result/{result_id}")
async def get_result(result_id: str):
    result_path = RESULT_DIR / f"result_{result_id}.jpg"
    if not result_path.exists():
        raise HTTPException(404, "Result not found or expired")
    return FileResponse(result_path)

@app.get("/api/system")
async def system_info():
    return JSONResponse({
        "system": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device": str(global_state["device"])
        },
        "model": {
            "loaded": global_state["model_loaded"],
            "type": "Autoencoder",
            "last_action": datetime.now().isoformat()
        },
        "paths": {
            "uploads": str(UPLOAD_DIR),
            "results": str(RESULT_DIR),
            "models": str(MODEL_DIR)
        }
    })

@app.get("/api/health")
async def health_check():
    return JSONResponse({
        "status": "operational",
        "model_ready": global_state["model_loaded"],
        "timestamp": datetime.now().isoformat()
    })