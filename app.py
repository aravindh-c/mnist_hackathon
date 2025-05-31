import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Load model
model = load_model("cnn_hackathon_mnist.keras")

# UI setup
st.title("🧠 Handwritten Digit Recognizer")
st.markdown("Draw a digit (0–9) below:")

# Canvas setup
canvas_result = st_canvas(
    fill_color="#000000",  # black ink
    stroke_width=10,
    stroke_color="#FFFFFF",  # white stroke
    background_color="#000000",  # black bg (like MNIST)
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Preprocess canvas drawing
    img = canvas_result.image_data[:, :, 0]  # use red channel (grayscale)
    img = Image.fromarray(img).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    pred_digit = np.argmax(prediction)

    st.write(f"### 🧾 Prediction: **{pred_digit}**")
    st.bar_chart(prediction[0])
