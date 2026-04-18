# Face Detection & Recognition

A real-time facial detection and recognition application using MediaPipe and face_recognition.

## Features

- **Real-time Face Detection** - Detect faces in video stream using MediaPipe
- **Face Recognition** - Recognize and identify saved faces with similarity matching
- **Multiple Camera Support** - Switch between different video sources
- **Face Management** - Add, save, and delete face references
- **Web Interface** - Clean, professional UI for easy management
- **Docker Support** - Run anywhere with containerization

## Tech Stack

- **Backend**: Flask
- **Face Detection**: MediaPipe
- **Face Recognition**: face_recognition (dlib)
- **Computer Vision**: OpenCV
- **Frontend**: HTML5, CSS3, JavaScript
- **Containerization**: Docker

## Installation

### Requirements
- Python 3.9+
- Webcam access
- (Optional) Docker

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/cdegrel/face-detection.git
cd face-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python3 -m src
```

The app will be available at `http://localhost:5005`

## Usage

### Web Interface

1. Open `http://localhost:5005` in your browser
2. Select video source from the sidebar
3. Use the controls to:
   - **Add Person**: Capture and save a new face
   - **View Saved Faces**: See all recognized individuals
   - **Delete Face**: Remove a saved reference

### Adding a New Face

1. Position yourself in front of the camera
2. Click "Add a person" button
3. Enter your name
4. Click "Save"

The system will now recognize you in future frames.

## Project Structure

```
face-detection/
├── src/
│   ├── __init__.py
│   ├── __main__.py          # Entry point (python3 -m src)
│   ├── main.py              # Application entry point
│   ├── server.py            # Flask application
│   ├── config.py            # Configuration
│   └── core/
│       ├── detection.py      # Face detection logic
│       └── recognition.py    # Face recognition logic
├── templates/
│   └── index.html           # Web UI
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── .github/
│   └── workflows/
│       └── docker-build.yml # CI/CD pipeline
└── README.md              # This file
```

## Docker

### Build Image

```bash
docker build -t face-detection .
```

### Run Container

**Linux/Server:**
```bash
docker run -p 5005:5005 --device /dev/video0 face-detection
```

**macOS/Windows:**
```bash
docker run -p 5005:5005 face-detection
```

### Docker Compose

```bash
docker-compose up
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| GET | `/video_feed` | Video stream (MJPEG) |
| GET | `/get_cameras` | List available cameras |
| POST | `/set_camera` | Change video source |
| GET | `/get_references` | Get saved faces |
| POST | `/save_reference` | Save a new face |
| POST | `/delete_reference` | Delete a face |

## Configuration

Edit `src/config.py` to customize:

```python
MIN_DETECTION_CONFIDENCE = 0.5    # Face detection threshold
RECOGNITION_THRESHOLD = 0.6       # Face recognition similarity threshold
PORT = 5005                       # Server port
HOST = '0.0.0.0'                 # Server host
```

## GitHub Actions

The project includes a CI/CD pipeline that automatically builds and pushes Docker images to GitHub Container Registry (GHCR) on every push to `main`.

View workflows: [.github/workflows/](.github/workflows/)

## Permissions

On macOS, you may need to grant camera access:
1. System Settings > Privacy & Security > Camera
2. Grant permission to Python or Terminal

## Performance Notes

- Video resolution: Full HD (1920x1080)
- Face encoding: 25% downsampling for speed
- Detection confidence: 50% (adjustable)
- Recognition threshold: 60% similarity (adjustable)

## Troubleshooting

**Camera not accessible:**
- macOS: Grant camera permissions in System Settings
- Linux: Check device permissions for `/dev/video0`

**Slow performance:**
- Reduce video resolution
- Increase detection confidence threshold
- Close other applications

**Face not recognized:**
- Ensure good lighting
- Face should be clearly visible
- Try adding face again with different angles

## License

MIT License - Feel free to use this project for personal or commercial use.

## Author

Created with love for real-time face detection and recognition.

---

For issues and feature requests, please open an issue on GitHub.
