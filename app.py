import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("asl_model (1).keras")

st.title("🤟 ASL Sign Language Classifier")
st.write("Upload a image of a hand sign to predict the corresponding ASL letter.")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

def preprocess_image(img):
    # Load image and convert to grayscale
    img = Image.open(r"C:\Users\HP\Downloads\images (2).jpg").convert('L')  # 'L' mode = grayscale
    img = img.resize((28, 28))  # Resize to match model input

    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_array = preprocess_image(image)

    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
                 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
                 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
                 }

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_letter = label_map.get(predicted_class, "Unknown")
    confidence = 100 * np.max(prediction)

    st.markdown(f"### Prediction: **{predicted_letter}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")



