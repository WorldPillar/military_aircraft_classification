import io
import streamlit as st
from PIL import Image

import cv2
import os
import time
from ultralytics import YOLO
import shutil

video_path = f'samples/videos/'

model = YOLO('yolov8-m-best.pt')
model.to('cuda')
st.title('Распознавание военных самолетов в облаке Streamlit')
file = st.empty()
button = st.empty()
video = st.empty()
if os.path.isdir("runs/detect/predict"):
    shutil.rmtree("runs/detect/predict")


@st.cache(allow_output_mutation=True)
def preprocess_image(img):
    return img.resize((img.height, img.width), Image.LANCZOS)


def load_image():
    uploaded_file = file.file_uploader(label='Выберите файл для распознавания')
    if uploaded_file is not None:
        print(uploaded_file)
        if uploaded_file.name[-3:] != 'mp4':
            image_data = uploaded_file.getvalue()
            video.image(image_data)
            return Image.open(io.BytesIO(image_data))
        else:
            cap = cv2.VideoCapture(video_path + uploaded_file.name)
            update(cap)
    else:
        return None


def update(cap=None, prev_frame_time=0):
    print(cap)
    ret, frame = cap.read()
    if not ret:
        return
    new_frame_time = time.time()
    fps = round(1 / (new_frame_time - prev_frame_time), 1)
    prev_frame_time = new_frame_time
    draw_fps(frame, fps)
    model.predict(frame, save=True, conf=0.5)
    image = Image.open("runs/detect/predict/image0.jpg")
    image = image.resize((600, 600), Image.LANCZOS)
    video.image(image)
    update(cap, prev_frame_time)


def draw_fps(frame, fps):
    cv2.putText(frame, str(fps), (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 255, 255))


result = button.button('Распознать')
img = load_image()
if result:
    x = preprocess_image(img)
    preds = model.predict(x, save=True)
    image = Image.open("runs/detect/predict/image0.jpg")
    image = image.resize((image.height, image.width), Image.LANCZOS)
    video.image(image)
