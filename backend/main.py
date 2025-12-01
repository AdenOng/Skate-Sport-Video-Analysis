from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shutil
import os
import time
import traceback
import subprocess  # <--- NEW IMPORT
from pathlib import Path
from analyzer import analyze_video_v2

app = FastAPI()

# Paths
YOLO_PATH = "models/yolox.onnx"
RTM_PATH = "models/rtmpose.onnx"
TEMP_DIR = Path("temp_videos")
TEMP_DIR.mkdir(exist_ok=True)

app.mount("/results", StaticFiles(directory="temp_videos"), name="results")

@app.post("/analyze")
def analyze_endpoint(file: UploadFile = File(...)):
    timestamp = int(time.time())
    safe_filename = f"{timestamp}_{file.filename}"
    input_path = TEMP_DIR / f"in_{safe_filename}"
    
    # We save the raw OpenCV output to a temp file first
    temp_output_filename = f"temp_out_{safe_filename}"
    temp_output_path = TEMP_DIR / temp_output_filename
    
    # The final web-ready file
    final_output_filename = f"out_{safe_filename}"
    final_output_path = TEMP_DIR / final_output_filename
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print(f"Starting analysis...")
        
        # 1. Run Analysis (Saves to temp_output_path)
        analyze_video_v2(
            video_path=str(input_path),
            yolo_path=YOLO_PATH,
            rtm_path=RTM_PATH,
            out_path=str(temp_output_path),
            process_every_n=2,
            device='cuda',
            show_window=False
        )

        # 2. Convert to H.264 for Browser Playback using FFmpeg
        print("Converting video for web playback...")
        subprocess.run([
            "ffmpeg", "-y", 
            "-i", str(temp_output_path), 
            "-vcodec", "libx264", 
            "-acodec", "aac", 
            str(final_output_path)
        ], check=True)
        
        # Remove the temp raw file to save space
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

        print("Conversion complete.")

    except Exception as e:
        print("!!!!!!!!!!! ERROR !!!!!!!!!!!")
        traceback.print_exc()
        return JSONResponse(content={"error": str(e), "trace": traceback.format_exc()}, status_code=500)

    return {
        "status": "success",
        "original_name": file.filename,
        # Point to the FFmpeg converted file
        "video_url": f"/api/results/{final_output_filename}",
        "processed_at": timestamp
    }