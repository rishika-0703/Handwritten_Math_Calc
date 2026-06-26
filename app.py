import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from tensorflow.keras.models import load_model


# -----------------------------
# Load Both Specialized Models
# -----------------------------
@st.cache_resource
def load_project_models():
    digit_net = load_model("D:/rishika/downloads/mnist_digit_model.keras")
    symbol_net = load_model("D:/rishika/downloads/symbol_model.keras")
    return digit_net, symbol_net


digits_model, symbols_model = load_project_models()

# Label maps matching your 2 separate model targets
digit_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
symbol_classes = ['+', '-', '*', '/']


# -----------------------------
# Image Core Processing
# -----------------------------
def preprocess_char(img):
    """
    Pads the drawing into a perfect square box to keep symbols like
    '-' and '/' from stretching into blocks when resized to 28x28.
    """
    h, w = img.shape[:2]
    max_side = max(h, w)

    # 1. Pad matrix securely to preserve the original structural weights
    pad_size = int(max_side * 1.4) if max_side > 0 else 28
    padded_img = np.zeros((pad_size, pad_size), dtype=np.uint8)

    # 2. Center the cropped stroke
    x_offset = (pad_size - w) // 2
    y_offset = (pad_size - h) // 2
    padded_img[y_offset:y_offset + h, x_offset:x_offset + w] = img

    # 3. Downscale and normalize to 0.0 - 1.0 bounds
    img_resized = cv2.resize(padded_img, (28, 28), interpolation=cv2.INTER_AREA)
    img_normalized = img_resized.astype("float32") / 255.0
    return img_normalized.reshape(1, 28, 28, 1)


def segment_characters(canvas_img):
    """
    Processes RGB channels with binary inversion to cleanly isolate black strokes
    on a white background, without alpha channel interference.
    """
    # Look at RGB channels only
    rgb_img = canvas_img[:, :, :3].astype("uint8")
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # Invert binary: White background becomes Black (0), Black stroke becomes White (255)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Bridge loose lines or brush cross paths (crucial for '+' or '*')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:  # Ignore tiny stray canvas clicks
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        char_img = thresh[y:y + h, x:x + w]
        chars.append((x, char_img))

    # Sort layouts from left to right sequentially
    chars = sorted(chars, key=lambda item: item[0])
    return [c[1] for c in chars]


def router_classify(img):
    """
    Routes characters using strict geometric and structural rules to prevent
    common symbol vs digit confusions (+ vs 4, / vs 1/7, and * vs digits).
    """
    h, w = img.shape[:2]
    aspect_ratio = w / float(h) if h > 0 else 1.0

    # ---------------------------------------------------------
    # Rule 1: Strict Check for MINUS (-)
    # ---------------------------------------------------------
    if aspect_ratio > 2.0 and h < 30:
        return "-"

    # ---------------------------------------------------------
    # Rule 2: Strict Check for SLASH / DIVISION (/)
    # ---------------------------------------------------------
    if 0.25 < aspect_ratio < 0.80 and h > 20:
        top_half = img[0:h // 2, :]
        bottom_half = img[h // 2:, :]

        if np.sum(top_half > 0) > 0 and np.sum(bottom_half > 0) > 0:
            # Fixed index reference to strictly parse the column components
            top_x = np.mean(np.where(top_half > 0)[1])
            bottom_x = np.mean(np.where(bottom_half > 0)[1])

            if top_x > bottom_x + (w * 0.22):
                top_row_density = np.sum(img[0:max(1, h // 6), :] > 0) / (max(1, h // 6) * w)
                if top_row_density < 0.55:
                    return "/"

    # ---------------------------------------------------------
    # Rule 3: Strict Check for PLUS (+) vs FOUR (4)
    # ---------------------------------------------------------
    if 0.75 < aspect_ratio < 1.3:
        mid_y = h // 2
        mid_x = w // 2

        horiz_center_line = np.sum(img[max(0, mid_y - 1):mid_y + 2, :] > 0) / (3 * w)
        vert_center_line = np.sum(img[:, max(0, mid_x - 1):mid_x + 2] > 0) / (3 * h)

        if horiz_center_line > 0.45 and vert_center_line > 0.45:
            return "+"

    # ---------------------------------------------------------
    # Rule 4: Strict Check for MULTIPLICATION (*) / (x)
    # ---------------------------------------------------------
    # Multiplication marks are square-like, have a dense center crossing point,
    # and strokes branching directly into all 4 corner quadrants.
    if 0.70 < aspect_ratio < 1.35 and h > 15:
        mid_y = h // 2
        mid_x = w // 2

        # Count the presence of pixels in all four corner quadrants
        q1 = np.sum(img[0:mid_y, 0:mid_x] > 0)  # Top-Left
        q2 = np.sum(img[0:mid_y, mid_x:] > 0)  # Top-Right
        q3 = np.sum(img[mid_y:, 0:mid_x] > 0)  # Bottom-Left
        q4 = np.sum(img[mid_y:, mid_x:] > 0)  # Bottom-Right

        # Look at the pixel intersection exactly at the center point (5x5 box matrix)
        center_box = img[max(0, mid_y - 2):mid_y + 3, max(0, mid_x - 2):mid_x + 3]
        center_density = np.sum(center_box > 0) / max(1, center_box.size)

        # A multiplication symbol has drawn parts in all 4 corners and intersects heavily in the middle
        if q1 > 0 and q2 > 0 and q3 > 0 and q4 > 0 and center_density > 0.45:
            # To differentiate from '8', '0', or 'x' vs 'plus':
            # Multiplication signs are hollow at the top-middle and bottom-middle outer borders.
            top_mid_empty = np.sum(img[0:max(1, h // 5), max(0, mid_x - 2):mid_x + 3] > 0) == 0
            bot_mid_empty = np.sum(img[int(h * 0.8):, max(0, mid_x - 2):mid_x + 3] > 0) == 0

            if top_mid_empty or bot_mid_empty:
                return "*"

    # ---------------------------------------------------------
    # Neural Network Predictions (Fallback for remaining digits)
    # ---------------------------------------------------------
    processed = preprocess_char(img)
    digit_pred = digits_model.predict(processed, verbose=0)
    symbol_pred = symbols_model.predict(processed, verbose=0)

    digit_conf = np.max(digit_pred)
    digit_idx = np.argmax(digit_pred)
    symbol_conf = np.max(symbol_pred)
    symbol_idx = np.argmax(symbol_pred)

    # Determine if symbol model output array has 4 or 14 elements
    if symbol_pred.shape[-1] == 14:
        symbol_map_14 = {10: '+', 11: '-', 12: '*', 13: '/'}
        pred_symbol = symbol_map_14.get(symbol_idx, '+')
    else:
        symbol_classes_4 = ['+', '-', '*', '/']
        pred_symbol = symbol_classes_4[symbol_idx] if symbol_idx < 4 else '+'

    if symbol_conf > digit_conf and symbol_conf > 0.45:
        return pred_symbol
    else:
        return str(digit_idx)


def evaluate_equation(equation):
    allowed_chars = "0123456789+-*/."
    cleaned_expr = "".join([c for c in equation if c in allowed_chars])

    if not cleaned_expr:
        return "No valid expression found"
    try:
        return eval(cleaned_expr)
    except Exception as e:
        return f"Syntax Error"


# -----------------------------
# Streamlit UI Configuration (Maintained Original Style)
# -----------------------------
st.set_page_config(page_title="Handwritten Calculator", layout="centered")
st.title("Handwritten Math Calculator")
st.write("Draw a clear horizontal equation (e.g., `5+3*2`) below:")

canvas_result = st_canvas(
    stroke_width=12,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=650,
    height=250,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True  # Triggers automatic rerun on every completed line stroke
)

if canvas_result.image_data is not None:
    chars = segment_characters(canvas_result.image_data)

    if chars:
        with st.expander("See isolated characters sent to model"):
            cols = st.columns(len(chars))
            for idx, c in enumerate(chars):
                cols[idx].image(255 - c, width=40, caption=f"Item {idx + 1}")

        # Compute predictions in real time
        equation = "".join([router_classify(c) for c in chars])

        st.subheader("Results")
        st.info(f"**Recognized Equation:** `{equation}`")

        result = evaluate_equation(equation)
        st.success(f"**Calculated Result:** `{result}`")
else:
    st.warning("Please draw an equation on the canvas canvas to begin.")
