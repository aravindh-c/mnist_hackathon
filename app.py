# --- MNIST Digit Recognizer with Enhanced Preprocessing for Canvas Input ---
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ImageOps

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

st.set_page_config(page_title="MNIST Digit Recognizer")

st.title("🖌️ Draw a Digit")
st.markdown("""
Draw one or more digits (0–9) in the canvas below and click **Predict** to see the model's prediction.
""")

# --- Load Model ---
model = load_model("cnn_hackathon_mnist_2.keras")

# --- Drawing Canvas ---
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=150,
    height=150,
    drawing_mode="freedraw",
    key="canvas",
)

# --- Preprocess Canvas Image ---
def preprocess_image(img_data):
    # Convert to grayscale
    gray = cv2.cvtColor(img_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to highlight digit
    th = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours of digits
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predictions = []
    digit_images = []
    output_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)

    # Sort contours left to right (for consistent digit order)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter out small noise
        if w < 5 or h < 5:
            continue

        # Crop and center digit
        digit = th[y:y+h, x:x+w]
        digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(digit, ((5,5),(5,5)), mode='constant', constant_values=0)

        # Normalize and reshape
        padded_digit = padded_digit / 255.0
        padded_digit = padded_digit.reshape(1, 28, 28, 1)

        pred = model.predict(padded_digit)[0]
        pred_label = np.argmax(pred)
        confidence = int(np.max(pred) * 100)
        # Inside for loop after prediction
        true_label = st.text_input(f"Label for Digit at ({x},{y})", key=f"label_{x}_{y}")
        if true_label.isdigit():
            np.save(f"data/digit_{true_label}_{datetime.datetime.now().timestamp()}.npy", padded_digit)
        

        predictions.append((pred_label, confidence, (x, y)))
        digit_images.append(padded_digit.reshape(28, 28))

        cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        cv2.putText(output_img, f"{pred_label} ({confidence}%)", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    X = []
    y = []
    for file in os.listdir("data"):
        if file.endswith(".npy"):
            label = int(file.split("_")[1])
            img = np.load(os.path.join("data", file))
            X.append(img)
            y.append(label)
    
    X = np.array(X).reshape(-1, 28, 28, 1)
    X = X.astype("float32") / 255.0
    y = to_categorical(y, 10)
    
    # Model (can match original MNIST structure)
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax')
    ])
    st.write(X)
    st.write(y)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32)
    model.save("cnn_hackathon_mnist_updated.keras")
    return predictions, output_img, digit_images

# --- Handle Prediction ---
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data
        predictions, annotated_img, digit_images = preprocess_image(img_data)

        st.image(annotated_img, caption="Processed Canvas")

        if predictions:
            pred_str = ", ".join([f"{p[0]} ({p[1]}%)" for p in predictions])
            st.success(f"Predicted digits: {pred_str}")

            # Show each extracted digit
            st.subheader("Extracted Digits")
            cols = st.columns(len(digit_images))
            for i, img in enumerate(digit_images):
                with cols[i]:
                    st.image(img, width=64, caption=f"Digit {predictions[i][0]}")
        else:
            st.warning("No digit detected. Try writing clearly inside the canvas.")

