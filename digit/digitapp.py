import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# OpenCV is used for image preprocessing. On some deployment platforms (e.g. Streamlit Cloud)
# the standard opencv-python package can cause import errors, so we prefer the headless
# variant and provide a fallback in case cv2 isn't available.
try:
    import cv2
except ImportError:
    cv2 = None
    st.warning("OpenCV (cv2) not installed; falling back to Pillow for preprocessing.")

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

st.title("🧠 MNIST Handwritten Digit Recognizer")

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.h5")

model = load_model()

st.write("Draw a digit (0–9) below and click Predict.")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_image(img):
    """Take the raw RGBA numpy array from the canvas and return a 1x28x28x1 tensor.

    Uses OpenCV when available for conversion and resizing; otherwise falls back to
    Pillow and numpy operations so the app can still run without cv2.
    """

    # cast to uint8 in case it is float
    arr = img.astype("uint8")

    if cv2:
        # Convert RGBA -> grayscale
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        # Pillow fallback path
        pil_img = Image.fromarray(arr)
        pil_gray = pil_img.convert("L")
        pil_resized = pil_gray.resize((28, 28), Image.ANTIALIAS)
        resized = np.array(pil_resized)

    # Normalize
    normalized = resized / 255.0
    # Reshape for CNN
    tensor = normalized.reshape(1, 28, 28, 1)
    return tensor


# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Digit"):

    if canvas_result.image_data is not None:

        processed_img = preprocess_image(canvas_result.image_data)

        prediction = model.predict(processed_img)
        predicted_digit = np.argmax(prediction)

        st.success(f"Predicted Digit: {predicted_digit}")

        st.subheader("Prediction Confidence")
        st.bar_chart(prediction[0])

        # Show processed image
        st.image(processed_img.reshape(28, 28), width=150)