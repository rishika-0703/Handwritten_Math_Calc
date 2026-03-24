import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your trained symbol model
model = load_model("D:/Raghu/Downloads/symbol_model.keras")

# Map indices to symbols (adjust based on training order)
symbols = {0: '+', 1: '-', 2: '*', 3: '/', 4: '=', 5: 'x'}

def preprocess_image(img):
    gray = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Thicken strokes
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Pad to square before resizing
    h, w = thresh.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = thresh

    resized = cv2.resize(square, (28, 28))
    norm = resized / 255.0
    norm = norm.reshape(1, 28, 28, 1)
    return norm

# Streamlit UI
st.title("Live Handwritten Symbol Recognition")

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = preprocess_image(canvas_result.image_data)
    prediction = model.predict(img)
    symbol_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    symbol = symbols[symbol_index]

    st.image(canvas_result.image_data, caption="Your Drawing", use_column_width=True)
    st.write("Predicted Symbol:", symbol)
    st.write(f"Confidence: {confidence:.2f}%")