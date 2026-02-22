# Skate Sport Video Analysis

AI-powered video analysis tool for skating sports. Automatically detects persons, tracks their movements, estimates poses, and analyzes skating-specific metrics like foot contact, posture, and knee angles.

## Features

- **Person Detection & Tracking** - Robust tracking that handles fast movements, occlusions, and multi-person scenarios
- **Pose Estimation** - 26-keypoint skeleton with foot keypoints for detailed skating analysis
- **Skating Metrics**
  - Foot contact detection (AIR, TOE, HEEL, FLAT)
  - Posture analysis with lean angle
  - Knee angle tracking with real-time graphs
  - Wheeling angle calculation
- **Multi-Person Support** - Select which skater to track in multi-person videos
- **Web Interface** - Dark-themed UI with video playback and history
- **GPU Accelerated** - CUDA support for fast video processing

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### Run with Docker

```bash
# Clone the repository
git clone https://github.com/AdenOng/Skate-Sport-Video-Analysis.git
cd Skate-Sport-Video-Analysis

# Start the application
docker compose up --build
```

Access the application at `http://localhost`

### Local Development (without Docker)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run the API
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

1. **Upload Video** - Click "Upload Video" and select a skating video (MP4, MOV, AVI, MKV, WebM supported)
2. **Select Skater** - If multiple persons detected, click on the skater you want to track
3. **Analyze** - Wait for processing (time depends on video length and GPU speed)
4. **View Results** - Watch the annotated video with metrics overlay
5. **Download** - Click "Download MP4" to save the processed video

### Metrics Displayed

| Metric | Description |
|--------|-------------|
| STATE | Overall foot contact state (Airborne, 1-Foot, 2-Foot) |
| Posture | Body lean classification (Upright, Slight Lean, Aggressive) |
| L/R-Knee | Left/Right knee angle in degrees |
| L/R-Ankle | Ankle wheeling angle for edge control analysis |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/detect` | POST | Detect persons in first frame |
| `/analyze` | POST | Analyze video and return processed video URL |
| `/cleanup` | POST | Manually trigger cleanup of old temporary files |

### Example: Analyze Video

```bash
curl -X POST "http://localhost/api/analyze" \
  -F "file=@skating_video.mp4"
```

### Example: Analyze with Target Selection

```bash
curl -X POST "http://localhost/api/analyze" \
  -F "file=@skating_video.mp4" \
  -F "target_bbox=100,200,400,600"
```

## Architecture

```
Browser ──► nginx (port 80) ──► FastAPI Backend (port 8000)
                 │                      │
                 │                      ├── YOLOX (person detection)
                 │                      ├── RTMPose (pose estimation)
                 │                      └── FFmpeg (video encoding)
                 │
                 └── Serves static frontend
```

### Tech Stack

- **Backend**: FastAPI, OpenCV, RTMLib (YOLOX + RTMPose)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Proxy**: nginx
- **Containerization**: Docker Compose with GPU support

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_SIZE` | 500MB | Maximum upload file size |
| `FILE_RETENTION_HOURS` | 24 | Hours before temp files are cleaned |

### Model Paths

Models are mounted from `./Models/` to `/app/models/` in the container:
- `yolox.onnx` - Person detection model
- `rtmpose.onnx` - Pose estimation model

## Project Structure

```
Skate-Sport-Video-Analysis/
├── backend/
│   ├── main.py          # FastAPI endpoints
│   ├── analyzer.py      # Core analysis pipeline
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile       # Backend container
├── frontend/
│   └── index.html       # Web UI
├── nginx/
│   └── nginx.conf       # Reverse proxy config
├── Models/
│   ├── yolox.onnx       # Detection model (Git LFS)
│   └── rtmpose.onnx     # Pose model (Git LFS)
├── docker-compose.yml   # Service orchestration
└── Original_tested_ipynb_file.ipynb  # Research prototype
```

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Video Processing Fails

- Check video format is supported (MP4, MOV, AVI, MKV, WebM)
- Ensure video is not corrupted
- Check backend logs: `docker compose logs -f backend`

### Upload Timeout

- Large videos may take several minutes to process
- nginx timeout is set to 300 seconds
- Increase in `nginx/nginx.conf` if needed

## License

This project is open source. Models used:
- YOLOX: Apache 2.0 License
- RTMPose: Apache 2.0 License

## Acknowledgments

- [RTMLib](https://github.com/Tau-J/rtmlib) - YOLOX and RTMPose wrappers
- [MMPose](https://github.com/open-mmlab/mmpose) - Pose estimation framework
- [OpenCV](https://opencv.org/) - Computer vision library
