import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import torch
import easyocr
import pytesseract
import os
import re

# 🔹 Load YOLO Models
HELMET_MODEL_PATH = "best.pt"  # Update path if needed
PLATE_MODEL_PATH = "last.pt"   # Update path if needed

# Path to Tesseract-OCR (Update if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
try:
    helmet_model = YOLO(HELMET_MODEL_PATH).to(device)
    plate_model = YOLO(PLATE_MODEL_PATH).to(device)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Initialize OCR Reader
reader = easyocr.Reader(['en'])

# CSV File Setup
CSV_FILE = "detection_results.csv"

if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["Person ID", "Helmet Status", "Number Plate"])
    df.to_csv(CSV_FILE, index=False)

st.title("🛵 Helmet & Number Plate Detection 🚀")
st.write("Upload an image to detect helmets and recognize number plates.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# 🔹 Function: Extract Clean Number Plate Text
def extract_valid_plate(text):
    """Extracts only valid alphanumeric characters & dots."""
    text = re.sub(r'[^A-Z0-9.]', '', text.upper())
    return text

if uploaded_file is not None:
    # 🔹 Load and Convert Image
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    person_id = 1
    detection_data = []

    # 🔹 Helmet Detection
    helmet_results = helmet_model(image)

    for result in helmet_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = box.conf[0].item()

            helmet_detected = cls == 0
            color = (0, 255, 0) if helmet_detected else (0, 0, 255)
            label = f"Helmet ({conf:.2f})" if helmet_detected else f"No Helmet ({conf:.2f})"

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 🔹 **Number Plate Detection**
            plate_text = "No plate detected"
            plate_results = plate_model(image)

            for plate_result in plate_results:
                for plate_box in plate_result.boxes:
                    plate_conf = plate_box.conf[0].item()  # Confidence Score
                    if plate_conf < 0.5:  # Ignore weak detections
                        continue

                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])

                    # Ensure cropping stays within bounds
                    h, w, _ = image.shape
                    px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)

                    plate_crop = image[py1:py2, px1:px2]

                    # 🛠 **OCR Preprocessing**
                    plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    plate_gray = cv2.GaussianBlur(plate_gray, (5, 5), 0)
                    plate_gray = cv2.equalizeHist(plate_gray)
                    _, plate_thresh = cv2.threshold(plate_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # 🔍 **Tesseract OCR with Strict Character Set**
                    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.'
                    detected_text_tesseract = pytesseract.image_to_string(plate_thresh, config=custom_config).strip()

                    # 🔍 **Filter Only Valid Number Plates**
                    plate_text_cleaned = extract_valid_plate(detected_text_tesseract)

                    # 🔍 **Fallback to EasyOCR if Tesseract Fails**
                    if not plate_text_cleaned:
                        detected_text_easyocr = reader.readtext(plate_thresh, detail=0)
                        if detected_text_easyocr:
                            plate_text_cleaned = extract_valid_plate(detected_text_easyocr[0])

                    # Final valid plate text
                    plate_text = plate_text_cleaned if plate_text_cleaned else "Not Recognized"

                    # Draw bounding box around plate
                    cv2.rectangle(image, (px1, py1), (px2, py2), (255, 255, 0), 3)
                    cv2.putText(image, plate_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            detection_data.append([f"Person {person_id}", "✅ Helmet" if helmet_detected else "❌ No Helmet", plate_text])
            person_id += 1

    # 🔹 **Display Image First**
    st.image(image, channels="BGR")

    # 🔹 **Save to CSV Immediately (Auto-Saving)**
    new_data = pd.DataFrame(detection_data, columns=["Person ID", "Helmet Status", "Number Plate"])
    new_data.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)

    # 🔹 **Then Show CSV Data**
    st.write("### 📋 Detection Results:")
    st.dataframe(new_data)

    st.success("✅ Data automatically saved to `detection_results.csv`")
