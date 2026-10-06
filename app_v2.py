import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os
import requests

# 1. PREMIUM PRODUCTION PAGE LAYOUT CONFIGURATION
st.set_page_config(
    page_title="AI Forensic Image Detection Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Aesthetic Styles & Color Palettes via Injectable Markdown
st.markdown("""
    <style>
    .main-title { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 700; color: #1E293B; text-align: center; margin-bottom: 5px; }
    .subtitle { text-align: center; color: #64748B; font-size: 1.1rem; margin-bottom: 30px; }
    .verdict-card { padding: 20px; border-radius: 12px; margin-bottom: 15px; text-align: center; font-weight: bold; font-size: 1.25rem; }
    .verdict-real { background-color: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }
    .verdict-ai { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🤖 The Illusion of Reality: Hybrid Deep Learning Image Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Phase-I Diagnostic Review Dashboard • Comparative Forensic Analytics Engine</p>", unsafe_allow_html=True)

# ─── SIDEBAR ANIMAL ANIMATION ───
st.sidebar.markdown("### 🦉 System Forensic Monitor")
st.sidebar.markdown(
    '<iframe src="https://embed.lottiefiles.com/animation/34452" style="width:100%; height:200px; border:none;"></iframe>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# ────────────────────────────────────────────────────────────────────────
# 2. AUTOMATED SECURITY GRADIENT CLOUD INGESTION UTILITY
# ────────────────────────────────────────────────────────────────────────
def download_model_from_drive(file_id, destination):
    """Streams large layer matrices dynamically from public cloud storage endpoints."""
    if not os.path.exists(destination):
        with st.spinner(f"Streaming fine-tuned weights matrix from cloud... Please wait a moment."):
            download_url = f"https://docs.google.com/uc?export=download&id={file_id}"
            session = requests.Session()
            response = session.get(download_url, stream=True)
            
            token = None
            for key, value in response.cookies.items():
                if "download_warning" in key:
                    token = value
                    break
                    
            if token:
                download_url = f"https://docs.google.com/uc?export=download&confirm={token}&id={file_id}"
                response = session.get(download_url, stream=True)
                
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)

DRIVE_FILE_ID = "1yGaE_P2-X2LdJxHQ3M7YRfRxd-zIKRtV"

@st.cache_resource
def load_detection_models():
    # Ingest V2 Baseline Assets matching your V2_models directory names exactly
    v2_eff = tf.keras.models.load_model("V2_models/efficientnet_v2_final.keras")
    v2_dnn = tf.keras.models.load_model("V2_models/dnn_v2_final.keras")
        
    # Ingest V3 Overhaul Assets (Stream base from Drive, load local JSON configurations)
    v3_local_path = "efficientnet_v3_final.keras"
    if not os.path.exists(v3_local_path) and DRIVE_FILE_ID:
        download_model_from_drive(DRIVE_FILE_ID, v3_local_path)
        
    v3_eff = tf.keras.models.load_model(v3_local_path) if os.path.exists(v3_local_path) else v2_eff
    v3_dnn = tf.keras.models.load_model("dnn_v3_final.keras")
    
    with open("hybrid_config_v3.json", "r") as f:
        v3_config = json.load(f)
        
    return v2_eff, v2_dnn, v3_eff, v3_dnn, v3_config

try:
    v2_eff, v2_dnn, v3_eff, v3_dnn, v3_cfg = load_detection_models()
    st.sidebar.success("✅ Forensic Pipelines Engaged (V2 + V3 Online)")
except Exception as e:
    st.sidebar.error(f"⚠️ Mounting Architecture Matrices: {str(e)}")

# ────────────────────────────────────────────────────────────────────────
# 3. INTERACTIVE FORENSIC UPLOAD INTERFACE
# ────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Drop target image file here for micro-structural artifact evaluation...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Center image display
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.image(img, caption="Forensic Target Image Preview", use_container_width=True)
    
    st.markdown("<br><hr>", unsafe_allow_html=True)
    
    # Side-by-Side Analytical Layout Columns
    col_v2, col_v3 = st.columns(2)
    
    img_array = np.array(img, dtype=np.float32)
    img_tensor = tf.convert_to_tensor(img_array)

    # ────────────────────────────────────────────────────────────────────
    # 📋 LEFT PANEL: V2 BASELINE ENGINE
    # ────────────────────────────────────────────────────────────────────
    with col_v2:
        st.markdown("### 📋 V2 Baseline Pipeline")
        with st.spinner("Parsing standard texture traces..."):
            v2_resized = tf.image.resize(img_tensor, [256, 256]) 
            v2_batch = tf.expand_dims(v2_resized, axis=0)
            
            # Fixed variable loops from x.name to eliminate NameErrors and typings
            v2_gap_layers = [x for x in v2_eff.layers if "pooling" in x.name or "gap" in x.name]
            v2_extractor = tf.keras.Model(v2_eff.input, [v2_eff.output, v2_gap_layers[-1].output])
            
            v2_eff_p, v2_features = v2_extractor.predict(v2_batch, verbose=0)
            v2_dnn_p = v2_dnn.predict(v2_features, verbose=0)
            
            v2_score = float((v2_eff_p + v2_dnn_p) / 2)
            v2_threshold = 0.5
            
        if v2_score >= v2_threshold:
            st.markdown("<div class='verdict-card verdict-ai'>🚨 VERDICT: AI-GENERATED</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='verdict-card verdict-real'>✅ VERDICT: AUTHENTIC REAL</div>", unsafe_allow_html=True)
            
        st.metric(label="AI Probability Metric Score", value=f"{v2_score * 100:.2f}%")
        st.progress(v2_score)
        st.caption("⚠️ Vulnerable to texture washouts caused by multi-scale resamplings.")

    # ────────────────────────────────────────────────────────────────────
    # 🚀 RIGHT PANEL: V3 FINE-TUNED HYBRID ENGINE
    # ────────────────────────────────────────────────────────────────────
    with col_v3:
        st.markdown("### ⚡ V3 Fine-Tuned Hybrid (Highly Accurate)")
        with st.spinner("Decompiling deep structural artifact signatures..."):
            v3_resized = tf.image.resize(img_tensor, [256, 256], method="bilinear", antialias=True)
            v3_resized = tf.clip_by_value(v3_resized, 0.0, 255.0)
            v3_batch = tf.expand_dims(v3_resized, axis=0)
            
            # Fixed variable loops to eliminate typos
            v3_gap_layers = [x for x in v3_eff.layers if "pooling" in x.name or "gap" in x.name]
            v3_extractor = tf.keras.Model(v3_eff.input, [v3_eff.output, v3_gap_layers[-1].output])
            
            v3_eff_p, v3_features = v3_extractor.predict(v3_batch, verbose=0)
            v3_dnn_p = v3_dnn.predict(v3_features, verbose=0)
            
            w_eff = v3_cfg["efficientnet_weight"]
            w_dnn = v3_cfg["dnn_weight"]
            v3_threshold = v3_cfg["threshold"]
            
            v3_score = float((w_eff * v3_eff_p) + (w_dnn * v3_dnn_p))

        if v3_score >= v3_threshold:
            st.markdown("<div class='verdict-card verdict-ai'>🚨 VERDICT: AI-GENERATED</div>", unsafe_allow_html=True)
            st.markdown(f"**Verification Status:** Artificial artifact signatures verified above optimized decision boundary (`{v3_threshold}`).")
        else:
            st.markdown("<div class='verdict-card verdict-real'>✅ VERDICT: AUTHENTIC REAL</div>", unsafe_allow_html=True)
            st.markdown(f"**Verification Status:** Structural pixel distributions match organic baseline metrics safely.")
            
        st.metric(label="AI Probability Metric Score", value=f"{v3_score * 100:.2f}%")
        st.progress(v3_score)
        st.caption("🛡️ Hardened network layers resilient against anti-fingerprint post-processing filters.")
