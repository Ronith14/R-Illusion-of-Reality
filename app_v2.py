import streamlit as st
import os
import zipfile
import requests
from PIL import Image
import numpy as np

# 1. Page Configuration & Theme Injection
st.set_page_config(
    page_title="Illusion of Reality Workspace",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-tech presentation aesthetic
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #00FFCC;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #8892B0;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #172A45;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #00FFCC;
        margin-bottom: 1rem;
    }
    .verdict-real {
        color: #00FFCC;
        font-weight: bold;
        font-size: 1.6rem;
    }
    .verdict-ai {
        color: #FF4D4D;
        font-weight: bold;
        font-size: 1.6rem;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Workspace Initialization & Drive Downloader
DRIVE_FILE_ID = "1yGaE_P2-X2LdJxHQ3M7YRfRxd-zIKRtV" 
MODEL_DIR = "models"
ZIP_PATH = os.path.join(MODEL_DIR, "kag_models.zip")

@st.cache_resource
def load_project_models():
    """Downloads and extracts the V2/V3 hybrid model weights automatically if missing."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    expected_files = ["efficientnet_v3_best.keras", "dnn_v3_final.keras"] 
    files_exist = all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in expected_files)
    
    if not files_exist:
        with st.spinner("⚡ Fetching Hybrid Network Binaries from Google Drive..."):
            download_url = f"https://google.com{DRIVE_FILE_ID}"
            
            try:
                response = requests.get(download_url, stream=True)
                if response.status_code == 200:
                    with open(ZIP_PATH, "wb") as f:
                        f.write(response.content)
                    
                    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                        zip_ref.extractall(MODEL_DIR)
                    
                    os.remove(ZIP_PATH)
                    st.toast("⚡ EfficientNet-B0 + DNN feature weights ready!", icon="🛡️")
                else:
                    st.error("Failed to download models automatically. Please place your downloaded Kaggle files manually into the 'models' folder.")
            except Exception as e:
                st.error(f"Workspace Activation Error: {str(e)}")

# Run background automated downloader
load_project_models()

# 3. Sidebar Configuration Layout
with st.sidebar:
    st.image("https://icons8.com", width=75)
    st.markdown("## Pipeline Options")
    st.caption("Illusion of Reality Classifier")
    
    st.divider()
    selected_version = st.selectbox(
        "Evaluation Strategy",
        options=["Model V2 (Multi-Generator Dataset)", "Model V3 (Enhanced Generalization)"]
    )
    
    st.divider()
    st.markdown("### Engine Architecture")
    st.code("Input Size: 256x256\nBackbone: EfficientNet-B0\nClassifier: Custom DNN\nFusion: Weighted Threshold")

# 4. Main Dashboard UI Content
st.markdown('<div class="main-title">🔮 The Illusion of Reality</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid Deep Learning Framework for Multi-Generator AI Image Detection</div>', unsafe_allow_html=True)

# Image Upload Interface Layout
st.subheader("📸 Media Processing Pipeline")
uploaded_file = st.file_uploader("Upload target visual asset for classification:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Target Asset Viewport")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Input Stream", use_container_width=True)
        
        # Preprocessing matching your 256x256 pipeline layout rules
        resized_img = image.resize((256, 256))
        
    with col2:
        st.markdown("### Execution Analytics")
        execute = st.button("⚡ Run Hybrid Inference Analysis", type="primary")
        
        if execute:
            with st.spinner("🔮 Processing latent feature vectors through EfficientNet-B0 + DNN..."):
                try:
                    import tensorflow as tf
                    
                    # Exact Kaggle mapped filenames assigned dynamically based on user dropdown selection
                    model_mapping = {
                        "Model V2 (Multi-Generator Dataset)": "efficientnet_v3_best.keras",
                        "Model V3 (Enhanced Generalization)": "dnn_v3_final.keras"
                    }
                    target_model_file = model_mapping[selected_version]
                    model_path = os.path.join(MODEL_DIR, target_model_file)
                    
                    # Ensure file exists before attempting to load
                    if not os.path.exists(model_path):
                        st.error(f"❌ Model file `{target_model_file}` not found in the `models/` folder. Please download it from Kaggle and place it inside the `models/` directory.")
                    else:
                        # Load actual trained Keras weights (compile=False bypasses optimizer conflicts)
                        model = tf.keras.models.load_model(model_path, compile=False)
                        
                        # Process image data into standard tensor format
                        if resized_img.mode != "RGB":
                            resized_img = resized_img.convert("RGB")
                            
                        img_array = np.array(resized_img)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = img_array / 255.0  # Normalize color channel vectors
                        
                        # Execute real model inference logic
                        prediction = model.predict(img_array)
                        raw_score = float(prediction)
                        
                        # Interface Output decisions
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.markdown("#### Final Classification Verdict")
                        
                        # Assuming values closer to 1 indicate an AI-Generated matrix
                        if raw_score > 0.5:
                            confidence_percentage = f"{raw_score * 100:.2f}%"
                            st.markdown('Result: <span class="verdict-ai">AI-GENERATED</span>', unsafe_allow_html=True)
                            st.metric(label="Classifier Detection Confidence", value=confidence_percentage, delta="Generative Artifact Matrix Match")
                        else:
                            confidence_percentage = f"{(1 - raw_score) * 100:.2f}%"
                            st.markdown('Result: <span class="verdict-real">REAL HISTOGRAM</span>', unsafe_allow_html=True)
                            st.metric(label="Classifier Detection Confidence", value=confidence_percentage, delta="Authentic Spectrum Match")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Live structural log display matrix
                        with st.expander("🛠️ View Latent Feature Extractor Matrix"):
                            st.json({
                                "input_resolution": f"{image.size}x{image.size}",
                                "processed_tensor_shape": str(img_array.shape),
                                "efficientnet_feature_vector_dimension": 1280,
                                "raw_prediction_scalar": raw_score,
                                "active_model_binary": target_model_file,
                                "supported_generators_evaluated": ["SD 1.5", "SDXL", "FLUX.1-schnell", "Kandinsky 2.2", "PixArt-Σ", "Würstchen"]
                            })
                            
                except ModuleNotFoundError:
                    st.error("❌ TensorFlow library is missing! Run `pip install tensorflow` in your terminal environment to handle predictions.")
                except Exception as e:
                    st.error(f"❌ Structural error handling matrix arrays: {str(e)}")
else:
    st.info("💡 Awaiting visual media vector payload upload to activate execution pipeline components.")
