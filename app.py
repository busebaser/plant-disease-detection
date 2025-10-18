import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Disease Diagnosis",
    page_icon="🌿",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Loads the trained Keras model (tries .keras then .h5)."""
    candidates = ['plant_disease_model.keras', 'plant_disease_model.h5']
    for fn in candidates:
        if os.path.exists(fn):
            try:
                return tf.keras.models.load_model(fn, compile=False)
            except Exception as e:
                st.warning(f"Failed loading {fn}: {e}")
    raise FileNotFoundError(f"None of the model files found: {candidates}")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

try:
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
except Exception as e:
    st.error(f"Failed to load class_names.json: {e}")
    st.stop()

# --- Image Preprocessing ---
def preprocess_image(image, target_size=(224, 224)):
    """Preprocesses the uploaded image to match model input.
    Ensures RGB, resizes, converts to float32 and keeps pixel values as-is.
    """
    img = image.convert('RGB')                        
    img = img.resize(target_size, resample=Image.BILINEAR)
    img_array = np.array(img).astype('float32')
    img_array = np.expand_dims(img_array, 0)        
    return img_array

# --- UI Layout ---
st.title("🌿 Plant Disease Diagnosis")
st.write("Upload an image of a plant leaf and the AI will diagnose the disease.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=400)

    # Show a spinner while diagnosing
    with st.spinner('Diagnosing...'):
        processed_image = preprocess_image(image)
        raw_pred = model.predict(processed_image)[0]   # get 1D array

        # If model already outputs probabilities (sum ~= 1) use as-is, else apply softmax
        if np.isclose(raw_pred.sum(), 1.0, atol=1e-3):
            probs = raw_pred
        else:
            probs = tf.nn.softmax(raw_pred).numpy()

        top_k = 5

        # Sanity check: lengths match
        if len(probs) != len(class_names):
            st.error(f"Class count mismatch: model output={len(probs)} vs class_names={len(class_names)}")
            st.write("First 10 model outputs:", probs[:10])
        else:
            idx = int(np.argmax(probs))
            predicted_class = class_names[idx]
            confidence = 100 * float(np.max(probs))

            st.divider()
            st.success(f"**Diagnosis:** {predicted_class.replace('___', ' - ').replace('_', ' ')}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            # Optional: show top-5
            top_idx = probs.argsort()[-top_k:][::-1]
            with st.expander("Top predictions"):
                for i in top_idx:
                    label = class_names[i].replace('___', ' - ').replace('_', ' ')
                    st.write(f"{label}: {probs[i]:.4f}")

    st.divider()
