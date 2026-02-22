import logging
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles

from analyzer import analyze_video_v2, detect_persons_first_frame

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

YOLO_PATH = "models/yolox.onnx"
RTM_PATH = "models/rtmpose.onnx"
TEMP_DIR = Path("temp_videos")
TEMP_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024
FILE_RETENTION_HOURS = 24
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app.mount("/results", StaticFiles(directory="temp_videos"), name="results")


def sanitize_filename(filename: str) -> str:
    safe_name = Path(filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File extension '{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}"
        )
    unique_id = uuid.uuid4().hex[:8]
    return f"{unique_id}_{safe_name}"


def cleanup_old_files():
    now = datetime.now()
    cutoff = now - timedelta(hours=FILE_RETENTION_HOURS)
    cleaned = 0
    for f in TEMP_DIR.iterdir():
        if f.is_file():
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                try:
                    f.unlink()
                    cleaned += 1
                    logger.debug(f"Cleaned up old file: {f}")
                except OSError as e:
                    logger.warning(f"Failed to delete {f}: {e}")
    if cleaned > 0:
        logger.info(f"Cleaned up {cleaned} old files")
    return cleaned


@app.on_event("startup")
async def startup_cleanup():
    cleanup_old_files()


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/cleanup")
def cleanup_endpoint():
    cleaned = cleanup_old_files()
    return {"status": "success", "files_removed": cleaned}


@app.post("/detect")
def detect_endpoint(file: UploadFile = File(...)):
    input_path = None
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        try:
            safe_filename = sanitize_filename(file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        input_path = TEMP_DIR / f"detect_{safe_filename}"

        content_length = file.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({int(content_length) / (1024 * 1024):.1f}MB). Max: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB",
            )

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = input_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size / (1024 * 1024):.1f}MB). Max: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB",
            )

        logger.info(f"Detecting persons in first frame of {safe_filename}")

        result = detect_persons_first_frame(
            video_path=str(input_path),
            yolo_path=YOLO_PATH,
            device="cuda",
        )

        return {
            "status": "success",
            "original_name": file.filename,
            **result,
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during detection")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during detection"
        )
    finally:
        if input_path and input_path.exists():
            try:
                input_path.unlink()
            except OSError:
                pass


@app.post("/analyze")
def analyze_endpoint(
    file: UploadFile = File(...),
    target_bbox: Optional[str] = Form(None),
):
    input_path = None
    temp_output_path = None
    final_output_path = None

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        try:
            safe_filename = sanitize_filename(file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        parsed_bbox = None
        if target_bbox:
            try:
                parts = [float(x.strip()) for x in target_bbox.split(",")]
                if len(parts) != 4:
                    raise ValueError(
                        "target_bbox must have exactly 4 values: x1,y1,x2,y2"
                    )
                x1, y1, x2, y2 = parts
                if x1 >= x2 or y1 >= y2:
                    raise ValueError(
                        "Invalid bbox: x1 must be less than x2, y1 must be less than y2"
                    )
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    raise ValueError(
                        "Invalid bbox: all coordinates must be non-negative"
                    )
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width < 10 or box_height < 10:
                    raise ValueError(
                        "Invalid bbox: box is too small (minimum 10x10 pixels)"
                    )
                parsed_bbox = tuple(parts)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid target_bbox format: {e}. Expected: x1,y1,x2,y2",
                )

        timestamp = int(time.time())
        input_path = TEMP_DIR / f"in_{safe_filename}"

        temp_output_filename = f"temp_out_{safe_filename}"
        temp_output_path = TEMP_DIR / temp_output_filename

        final_output_filename = f"out_{safe_filename}"
        final_output_path = TEMP_DIR / final_output_filename

        content_length = file.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({int(content_length) / (1024 * 1024):.1f}MB). Max: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB",
            )

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = input_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size / (1024 * 1024):.1f}MB). Max: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB",
            )

        logger.info(
            f"Starting analysis of {safe_filename} ({file_size / (1024 * 1024):.1f}MB)"
            + (f" with target_bbox={parsed_bbox}" if parsed_bbox else "")
        )

        analyze_video_v2(
            video_path=str(input_path),
            yolo_path=YOLO_PATH,
            rtm_path=RTM_PATH,
            out_path=str(temp_output_path),
            process_every_n=2,
            device="cuda",
            show_window=False,
            target_bbox=parsed_bbox,
        )

        if not temp_output_path.exists():
            raise RuntimeError("Analysis completed but output file was not created")

        logger.info("Converting video for web playback...")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_output_path),
                "-vcodec",
                "libx264",
                "-acodec",
                "aac",
                "-movflags",
                "+faststart",
                str(final_output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Conversion complete.")

        return {
            "status": "success",
            "original_name": file.filename,
            "video_url": f"/api/results/{final_output_filename}",
            "processed_at": timestamp,
        }

    except HTTPException:
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        raise HTTPException(status_code=500, detail="Video conversion failed")
    except RuntimeError as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during analysis")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during analysis"
        )
    finally:
        if input_path and input_path.exists():
            try:
                input_path.unlink()
            except OSError:
                pass
        if temp_output_path and temp_output_path.exists():
            try:
                temp_output_path.unlink()
            except OSError:
                pass
