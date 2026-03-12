import os
import time
import threading
import cv2
import torch
from flask import Flask, render_template, send_from_directory, jsonify, url_for

app = Flask(__name__)

# Paths — derived relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov5", "runs", "train", "parking_detect_local7", "weights", "best.pt")
TEST_DIR = os.path.join(BASE_DIR, "Dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "Latest")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv5 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False).to(device)
model.eval()
model.names = ['empty', 'occupied']   

# Detector thread
def run_detector():
    index = 0
    while True:
        test_images = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")])
        if not test_images:
            time.sleep(5)
            continue

        image_name = test_images[index % len(test_images)]
        image_path = os.path.join(TEST_DIR, image_name)

        # Run inference
        results = model(image_path)

        # Filter predictions by confidence threshold
        df_preds = results.pandas().xyxy[0]
        df_filtered = df_preds[df_preds['confidence'] >= 0.60]

        # Count classes
        detected_classes = df_filtered['name'].tolist()
        occupied = detected_classes.count('occupied')
        empty = detected_classes.count('empty')

        # Render image with detections
        results.pandas().xyxy[0] = df_filtered  # update for rendering
        results.render()
        annotated_img = results.ims[0]  # rendered image as numpy array

        # Save output image
        output_image = os.path.join(OUTPUT_DIR, "current.jpg")
        cv2.imwrite(output_image, annotated_img)

        # Save detection data
        with open(os.path.join(OUTPUT_DIR, "current.txt"), "w") as f:
            f.write(f"{occupied},{empty}")

        print(f"[{time.ctime()}] Image: {image_name} | 🟥 Occupied: {occupied} | 🟩 Empty: {empty}")
        index += 1
        time.sleep(10)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/latest_image")
def latest_image():
    data_file = os.path.join(OUTPUT_DIR, "current.txt")
    occupied = empty = "?"
    if os.path.exists(data_file):
        with open(data_file) as f:
            occupied, empty = f.read().strip().split(",")

    return jsonify({
        "image": url_for('latest_image_file', filename="current.jpg"),
        "occupied": occupied,
        "empty": empty
    })

@app.route("/latest_images/<filename>")
def latest_image_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# ✅ Start detector thread manually
if __name__ == "__main__":
    threading.Thread(target=run_detector, daemon=True).start()
    app.run(debug=True, use_reloader=False)
