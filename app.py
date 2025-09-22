import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ---------------------------
# Define focal loss again (same as training)
# ---------------------------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bce_exp = tf.exp(-bce)
        return alpha * (1 - bce_exp) ** gamma * bce
    return focal_loss_fixed

# ---------------------------
# Load trained model
# ---------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("tb_model.keras", compile=False)
    return model

model = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🩺 TB Detection from Chest X-rays")
st.write("Upload a chest X-ray image, and the model will predict whether it shows **TB** or **Normal**.")

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

    # Prediction
    prediction = model.predict(img_array)[0][0]
    label = "TB" if prediction > 0.5 else "Normal"

    # Show result
    st.image(uploaded_file, caption=f"Prediction: **{label}**", use_column_width=True)
    st.write(f"🔹 Model confidence: {prediction:.4f} (closer to 1 → TB, closer to 0 → Normal)")
