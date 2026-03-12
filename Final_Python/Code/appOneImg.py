import torch
import os

# Paths
MODEL_PATH = "/Users/ksanskar/Work/Course_Project_Python/Code/yolov5/runs/train/parking_detect_local7/weights/best.pt"
IMAGE_PATH = "Code/test/images/2013-04-14_10_40_05_jpg.rf.ba7764fcaa49a833e0e93b7aed9188e6.jpg "

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

# Correct class index mapping
model.names = ['empty', 'occupied']  # 0 = empty, 1 = occupied

# Run inference
if os.path.exists(IMAGE_PATH):
    results = model(IMAGE_PATH)
    detected_classes = results.pandas().xyxy[0]['name'].tolist()

    empty_count = detected_classes.count('empty')
    occupied_count = detected_classes.count('occupied')

    print(f"\nImage: {os.path.basename(IMAGE_PATH)}")
    print(f"🟥 Occupied parking spaces: {occupied_count}")
    print(f"🟩 Empty parking spaces: {empty_count}")

    results.show()
else:
    print(f"❌ Image not found at: {IMAGE_PATH}")
