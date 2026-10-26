import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import os

# Page Config
st.set_page_config(page_title="PixelPoet AI", page_icon="🌟", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0px 4px 15px rgba(75, 108, 183, 0.4);
    }
    .hero-header {
        font-family: 'Inter', sans-serif;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 900;
        margin-bottom: 0px;
    }
    .hero-subtitle {
        text-align: center;
        color: #A0AEC0;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    .caption-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        color: #E2E8F0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_models(model_path, tokenizer_path, feature_extractor_path):
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer

def generate_caption(image_path, caption_model, feature_extractor, tokenizer, max_length=34, img_size=224):
    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    image_features = feature_extractor.predict(img, verbose=0)
    
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    return caption

def main():
    st.markdown('<h1 class="hero-header">PixelPoet AI ✨</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">An AI model that instantly creates engaging captions for your images, making every post social-ready.</p>', unsafe_allow_html=True)
    
    model_path = "models/model.keras"
    tokenizer_path = "models/tokenizer.pkl"
    feature_extractor_path = "models/feature_extractor.keras"
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path) or not os.path.exists(feature_extractor_path):
        st.error("Model files are missing. Please ensure the 'models' directory contains 'model.keras', 'tokenizer.pkl', and 'feature_extractor.keras'.")
        return

    with st.spinner("Loading Deep Learning Models into Memory..."):
        try:
            caption_model, feature_extractor, tokenizer = load_all_models(model_path, tokenizer_path, feature_extractor_path)
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return

    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown("### 🖼️ 1. Upload Image")
        uploaded_image = st.file_uploader("Drop an image here...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_image is not None:
            # Save the uploaded image temporarily
            with open("uploaded_image.jpg", "wb") as f:
                f.write(uploaded_image.getbuffer())
                
            # Show a cool preview
            img_pil = Image.open(uploaded_image)
            st.image(img_pil, caption="Source Image", use_container_width=True)

    with col2:
        st.markdown("### 🧠 2. Generate Insight")
        if uploaded_image is not None:
            if st.button("Generate Caption 🚀", use_container_width=True):
                with st.spinner("Analyzing pixels and generating semantics..."):
                    caption = generate_caption("uploaded_image.jpg", caption_model, feature_extractor, tokenizer)
                
                st.success("Analysis Complete!")
                st.markdown(f'<div class="caption-box">"{caption.capitalize()}"</div>', unsafe_allow_html=True)
                
                # Plot the image using matplotlib as requested by the user, with a sleek dark theme
                with st.expander("View Matplotlib Processed Output"):
                    img = load_img("uploaded_image.jpg", target_size=(224, 224))
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title("Processed Tensor Visualization", color='#4ECDC4', fontsize=14, pad=15)
                    st.pyplot(fig)
        else:
            st.info("👈 Upload an image on the left to unlock insights!")

if __name__ == "__main__":
    main()
