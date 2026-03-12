# Parking Spot Detector

A real-time **smart parking spot detection** system using a custom-trained YOLOv5 model. A Flask web dashboard displays live annotated camera feeds with counts of occupied vs. empty spots.

## Features

- Custom YOLOv5 model trained on a local parking dataset (`empty` / `occupied`)
- Background inference thread processes images every 10 seconds
- Live web dashboard showing annotated parking lot image
- REST API returns detection results as JSON
- Auto GPU/CPU selection (CUDA when available)

## Tech Stack

- **Python 3.x**
- **YOLOv5** (PyTorch) — custom-trained model
- **Flask** — web server & dashboard
- **OpenCV** — image processing
- **Threading** — background inference

## Installation

```bash
cd Final_Python/Code
python -m venv venv
source venv/bin/activate
pip install -r ../../requirements.txt
```

## Usage

```bash
cd Final_Python/Code
python main.py
```

Open browser at `http://localhost:5000`

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web dashboard |
| `GET /latest_image` | JSON with image URL, occupied & empty counts |
| `GET /latest_images/<filename>` | Annotated image file |

## Project Structure

```
Parking_Spot_Detector/
├── Final_Python/
│   └── Code/
│       ├── main.py          # Flask app + detector thread
│       ├── templates/       # HTML frontend
│       ├── Dataset/         # Training/test images
│       │   └── test/        # Test images for inference
│       ├── yolov5/          # YOLOv5 framework
│       └── Latest/          # Output images & detection data
├── requirements.txt
└── README.md
```

## Academic Context

Python for Engineers — Course Project, Semester 2, 2024-25 | Group 2
