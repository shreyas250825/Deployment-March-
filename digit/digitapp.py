import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import cv2

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

    # Convert to grayscale
    img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize
    img = img / 255.0

    # Reshape for CNN
    img = img.reshape(1, 28, 28, 1)

    return img


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