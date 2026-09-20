import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="The Illusion of Reality | AI Image Detection",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "V2_models"
EFF_PATH = os.path.join(MODEL_DIR, "efficientnet_v2_final.keras")
DNN_PATH = os.path.join(MODEL_DIR, "dnn_v2_final.keras")
CONFIG_PATH = os.path.join(MODEL_DIR, "hybrid_config_v2.json")
ROBUSTNESS_PATH = "robustness_summary.csv"

# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "user_name": "",
    "scan_count": 0,
    "scan_history": [],
    "last_result": None,
    "session_started": datetime.now().strftime("%d %b %Y, %I:%M %p"),
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>
:root {
    --bg: #050811;
    --panel: rgba(12, 18, 31, 0.88);
    --panel2: rgba(15, 22, 40, 0.78);
    --cyan: #00e5ff;
    --violet: #8b5cf6;
    --muted: #91a0b7;
    --line: rgba(255,255,255,0.08);
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(0,229,255,0.08), transparent 28%),
        radial-gradient(circle at 90% 12%, rgba(139,92,246,0.10), transparent 30%),
        linear-gradient(135deg, #03050b 0%, #07101d 48%, #04060c 100%);
    color: #ffffff;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Hide Streamlit branding clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

@keyframes riseIn {
    0% { opacity: 0; transform: translateY(24px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
    0% { opacity: 0; transform: translateX(80px); }
    100% { opacity: 1; transform: translateX(0); }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 18px rgba(0,229,255,0.12); }
    50% { box-shadow: 0 0 38px rgba(0,229,255,0.30); }
}

@keyframes scanGlow {
    0%, 100% { border-color: rgba(0,229,255,0.15); }
    50% { border-color: rgba(0,229,255,0.42); }
}

.fade-in {
    animation: riseIn 0.75s ease-out;
}

.slide-in {
    animation: slideIn 0.75s ease-out;
}

/* ========================================================
   ENTRY SCREEN
   ======================================================== */
.entry-shell {
    min-height: 82vh;
    display: flex;
    align-items: center;
    animation: slideIn 0.8s ease-out;
}

.entry-card {
    background:
        linear-gradient(135deg, rgba(12,18,31,0.97), rgba(5,8,17,0.94));
    border: 1px solid rgba(0,229,255,0.18);
    border-radius: 30px;
    padding: 48px;
    animation: pulseGlow 4s infinite;
}

.r-logo {
    width: 72px;
    height: 72px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
    font-weight: 950;
    color: #fff;
    background: linear-gradient(135deg, #00e5ff, #7c4dff);
    box-shadow: 0 0 28px rgba(0,229,255,0.32);
    margin-bottom: 26px;
}

.entry-title {
    font-size: clamp(38px, 5vw, 64px);
    line-height: 0.98;
    font-weight: 950;
    letter-spacing: -2px;
    background: linear-gradient(90deg, #fff 0%, #00e5ff 52%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.entry-kicker {
    color: #00e5ff;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: 3px;
    margin-bottom: 12px;
}

.entry-subtitle {
    color: #9aa9bf;
    line-height: 1.8;
    font-size: 16px;
    margin-top: 20px;
    max-width: 700px;
}

.entry-side {
    padding: 34px 16px;
    animation: slideIn 1s ease-out;
}

.side-title {
    color: #fff;
    font-weight: 850;
    font-size: 25px;
    margin-bottom: 10px;
}

.side-text {
    color: #8898b0;
    font-size: 15px;
    line-height: 1.75;
}

.feature-chip {
    display: inline-block;
    padding: 8px 12px;
    margin: 4px 4px 4px 0;
    border-radius: 999px;
    border: 1px solid rgba(0,229,255,0.17);
    background: rgba(0,229,255,0.045);
    color: #cad6e7;
    font-size: 12px;
}

/* ========================================================
   NAVBAR
   ======================================================== */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    padding: 14px 18px;
    margin-bottom: 20px;
    border-radius: 18px;
    border: 1px solid var(--line);
    background: rgba(6,10,20,0.78);
    backdrop-filter: blur(14px);
}

.brand-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nav-logo {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 950;
    background: linear-gradient(135deg, #00e5ff, #7c4dff);
    box-shadow: 0 0 20px rgba(0,229,255,0.24);
}

.brand-title {
    font-weight: 900;
    color: #fff;
    letter-spacing: 0.8px;
    font-size: 15px;
}

.brand-sub {
    color: #71819a;
    font-size: 11px;
    letter-spacing: 1.7px;
}

.live-pill {
    border: 1px solid rgba(0,229,255,0.22);
    background: rgba(0,229,255,0.07);
    color: #8df4ff;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
}

.user-pill {
    border: 1px solid rgba(139,92,246,0.22);
    background: rgba(139,92,246,0.07);
    color: #d5c7ff;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 11px;
}

/* ========================================================
   HERO / CARDS
   ======================================================== */
.hero {
    padding: 28px 30px;
    border-radius: 26px;
    border: 1px solid rgba(255,255,255,0.08);
    background:
        radial-gradient(circle at 100% 0%, rgba(139,92,246,0.13), transparent 34%),
        linear-gradient(135deg, rgba(9,16,30,0.96), rgba(6,10,18,0.88));
    margin-bottom: 20px;
}

.hero-kicker {
    color: #00e5ff;
    font-size: 12px;
    letter-spacing: 2.5px;
    font-weight: 900;
}

.hero-title {
    font-size: clamp(30px, 4vw, 52px);
    font-weight: 950;
    line-height: 1.02;
    margin-top: 8px;
    letter-spacing: -1.5px;
}

.hero-copy {
    color: #8e9db3;
    max-width: 900px;
    line-height: 1.7;
    margin-top: 12px;
}

.glass-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 22px;
    animation: riseIn 0.55s ease-out;
}

.metric-card {
    background: linear-gradient(135deg, rgba(0,229,255,0.055), rgba(139,92,246,0.055));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    min-height: 116px;
}

.metric-label {
    color: #7c8da7;
    font-size: 11px;
    letter-spacing: 1.2px;
    font-weight: 800;
}

.metric-value {
    color: #fff;
    font-size: 30px;
    font-weight: 950;
    margin-top: 7px;
}

.metric-note {
    color: #697993;
    font-size: 11px;
    margin-top: 2px;
}

.result-card {
    border-radius: 24px;
    padding: 28px;
    border: 1px solid rgba(0,229,255,0.20);
    background:
        radial-gradient(circle at 90% 0%, rgba(0,229,255,0.10), transparent 30%),
        linear-gradient(135deg, rgba(10,18,32,0.97), rgba(6,9,16,0.92));
    animation: riseIn 0.65s ease-out;
}

.result-label {
    color: #73839c;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 900;
}

.result-main {
    font-size: 40px;
    line-height: 1;
    font-weight: 950;
    margin-top: 7px;
}

.ai-text { color: #a78bfa; }
.real-text { color: #67e8f9; }

.small-note {
    color: #7889a2;
    font-size: 12px;
    line-height: 1.65;
}

.chat-bubble {
    border: 1px solid rgba(0,229,255,0.12);
    background: rgba(0,229,255,0.035);
    border-radius: 18px;
    padding: 15px 17px;
    color: #b9c8db;
    margin: 10px 0;
}

.chat-bubble strong { color: #fff; }

.footer-line {
    text-align: center;
    color: #506078;
    font-size: 11px;
    letter-spacing: 1.5px;
    margin-top: 30px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================


def get_config_value(config, keys, default):
    for key in keys:
        if key in config:
            return config[key]
    return default


@st.cache_resource(show_spinner="Loading the V2 detection engine...")
def load_models():
    if not os.path.exists(EFF_PATH):
        raise FileNotFoundError(f"Missing EfficientNet model: {EFF_PATH}")
    if not os.path.exists(DNN_PATH):
        raise FileNotFoundError(f"Missing DNN model: {DNN_PATH}")

    eff_model = tf.keras.models.load_model(EFF_PATH, compile=False)
    dnn_model = tf.keras.models.load_model(DNN_PATH, compile=False)

    # The DNN was trained on the 1280-D EfficientNet representation.
    feature_model = None
    for layer in reversed(eff_model.layers[:-1]):
        try:
            shape = tuple(layer.output.shape)
            if len(shape) == 2 and shape[-1] == 1280:
                feature_model = tf.keras.Model(eff_model.input, layer.output)
                break
        except Exception:
            continue

    if feature_model is None:
        raise RuntimeError(
            "Could not locate the 1280-D EfficientNet feature layer. "
            "Check the saved V2 EfficientNet architecture."
        )

    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

    threshold = float(
        get_config_value(
            config,
            ["best_threshold", "BEST_THRESHOLD", "threshold"],
            0.36,
        )
    )
    eff_weight = float(
        get_config_value(
            config,
            ["BEST_EFF_WEIGHT", "best_eff_weight", "eff_weight"],
            0.5,
        )
    )
    dnn_weight = float(
        get_config_value(
            config,
            ["BEST_DNN_WEIGHT", "best_dnn_weight", "dnn_weight"],
            0.5,
        )
    )

    return eff_model, dnn_model, feature_model, config, threshold, eff_weight, dnn_weight


def preprocess_image(image: Image.Image):
    """
    Resize to the V2 input size.

    Keras EfficientNet models normally contain their own input Rescaling layer,
    so raw RGB pixels in the 0-255 range are supplied here. This avoids accidental
    double-normalization in the app.
    """
    image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    tensor = tf.image.resize(tensor, [256, 256], antialias=True)
    tensor = tf.expand_dims(tensor, axis=0)
    return tensor


def predict_image(image: Image.Image, eff_model, dnn_model, feature_model, threshold, eff_weight, dnn_weight):
    batch = preprocess_image(image)

    eff_raw = eff_model(batch, training=False).numpy().reshape(-1)[0]
    features = feature_model(batch, training=False)
    dnn_raw = dnn_model(features, training=False).numpy().reshape(-1)[0]

    hybrid = float(eff_weight * eff_raw + dnn_weight * dnn_raw)
    label = "AI-GENERATED" if hybrid >= threshold else "REAL"
    model_score = hybrid if label == "AI-GENERATED" else (1.0 - hybrid)

    return {
        "label": label,
        "model_score": float(model_score),
        "hybrid_ai_score": float(hybrid),
        "efficientnet_score": float(eff_raw),
        "dnn_score": float(dnn_raw),
        "threshold": float(threshold),
        "margin": float(abs(hybrid - threshold)),
    }


def reset_session():
    st.session_state.user_name = ""
    st.session_state.scan_count = 0
    st.session_state.scan_history = []
    st.session_state.last_result = None
    st.session_state.session_started = datetime.now().strftime("%d %b %Y, %I:%M %p")
    st.rerun()


# ============================================================
# MODEL CHECK
# ============================================================

try:
    (
        EFF_MODEL,
        DNN_MODEL,
        FEATURE_MODEL,
        MODEL_CONFIG,
        THRESHOLD,
        EFF_WEIGHT,
        DNN_WEIGHT,
    ) = load_models()
    MODEL_READY = True
    MODEL_ERROR = ""
except Exception as exc:
    MODEL_READY = False
    MODEL_ERROR = str(exc)


# ============================================================
# NAME ENTRY SCREEN
# ============================================================

if not st.session_state.user_name:
    st.markdown('<div class="entry-shell">', unsafe_allow_html=True)

    left, right = st.columns([1.3, 0.7], gap="large")

    with left:
        st.markdown(
            """
            <div class="entry-card">
                <div class="entry-kicker">HYBRID DEEP LEARNING • V2 ENGINE</div>
                <div class="r-logo">R</div>
                <div class="entry-title">THE ILLUSION<br>OF REALITY</div>
                <div class="entry-subtitle">
                    Before we begin the investigation, tell me what I should call you, mate.
                    <br><br>
                    Your image is waiting. The model is ready. Reality is not always what it looks like.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        name = st.text_input(
            "What's your name, mate?",
            placeholder="Enter your name...",
            key="entry_name",
            label_visibility="visible",
        )

        if st.button("ENTER THE EXPERIENCE  →", width="stretch", type="primary"):
            clean_name = " ".join(name.split()).strip()
            if clean_name:
                st.session_state.user_name = clean_name
                st.session_state.session_started = datetime.now().strftime("%d %b %Y, %I:%M %p")
                st.toast(f"Welcome, {clean_name}! Let's investigate some images. 🦊", icon="🦊")
                st.rerun()
            else:
                st.warning("I need a name first, mate. Even detectives introduce themselves.")

    with right:
        st.markdown(
            """
            <div class="entry-side">
                <div class="side-title">What happens next?</div>
                <div class="side-text">
                    You'll enter the AI image detection lab, upload an image, and let the hybrid model examine it.
                    <br><br>
                    No unnecessary drama. Just pixels, models, probabilities, and the occasional betrayal by reality.
                </div>
                <br>
                <div class="feature-chip">⚡ EfficientNet-B0</div>
                <div class="feature-chip">🧠 DNN Hybrid</div>
                <div class="feature-chip">🔬 Image Analysis</div>
                <div class="feature-chip">📊 Model Insights</div>
                <div class="feature-chip">🛡 Robustness Lab</div>
                <br><br>
                <div class="side-title">Built for investigation.</div>
                <div class="side-text">Upload. Analyze. Question what you see.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ============================================================
# MAIN NAVIGATION
# ============================================================

st.markdown('<div class="fade-in">', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="navbar">
        <div class="brand-wrap">
            <div class="nav-logo">R</div>
            <div>
                <div class="brand-title">THE ILLUSION OF REALITY</div>
                <div class="brand-sub">HYBRID AI-IMAGE DETECTION</div>
            </div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end;">
            <div class="live-pill">● V2 ENGINE ONLINE</div>
            <div class="user-pill">👤 {st.session_state.user_name}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR SESSION CONTROLS
# ============================================================

with st.sidebar:
    st.markdown("### 🦊 Investigation Session")
    st.write(f"**Investigator:** {st.session_state.user_name}")
    st.write(f"**Images analyzed:** {st.session_state.scan_count}")
    st.write(f"**Session started:** {st.session_state.session_started}")
    st.divider()
    if st.button("Start a New Session", width="stretch"):
        reset_session()

# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero slide-in">
        <div class="hero-kicker">WELCOME BACK, {st.session_state.user_name.upper()} • INVESTIGATION CONSOLE</div>
        <div class="hero-title">Can you tell what's real?</div>
        <div class="hero-copy">
            Upload an image and let the V2 hybrid detector inspect it using EfficientNet-B0 features and a DNN fusion layer.
            The score is a model output, not proof of absolute reality.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_READY:
    st.error(
        "The V2 model files could not be loaded. Check the paths below before launching the app:\n\n"
        f"`{EFF_PATH}`\n\n`{DNN_PATH}`\n\n`{CONFIG_PATH}`\n\n"
        f"Technical detail: {MODEL_ERROR}"
    )
    st.stop()

# ============================================================
# TABS
# ============================================================

tab_scan, tab_model, tab_robust, tab_batch, tab_system = st.tabs(
    [
        "⚡ SCAN",
        "◈ MODEL INTELLIGENCE",
        "◉ ROBUSTNESS LAB",
        "▦ BATCH",
        "◆ SYSTEM",
    ]
)

# ============================================================
# SCAN TAB
# ============================================================

with tab_scan:
    if not st.session_state.last_result:
        st.markdown(
            f"""
            <div class="chat-bubble">
                <strong>Hey {st.session_state.user_name}! 👋</strong><br>
                You can upload your image right here, mate. I'll take a look and tell you what the hybrid model detects.
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploaded = st.file_uploader(
        "Drop an image into the investigation zone",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        help="Supported formats: JPG, JPEG, PNG and WEBP.",
    )

    if uploaded is None:
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:38px;">🦊</div>
                <div style="font-size:21px;font-weight:850;color:#fff;margin-top:5px;">Your investigation starts here.</div>
                <div class="small-note" style="margin-top:8px;">
                    Upload a single image. I'll show you the hybrid result, model score, threshold,
                    component scores and the difference between the fused output and the decision boundary.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception as exc:
            st.error(f"I couldn't open that image: {exc}")
            st.stop()

        st.markdown(
            f"""
            <div class="chat-bubble">
                <strong>Thanks for uploading that, {st.session_state.user_name}! 📸</strong><br>
                I've loaded <strong>{uploaded.name}</strong>. It's ready for analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )

        preview_col, control_col = st.columns([1.2, 0.8], gap="large")

        with preview_col:
            st.image(image, caption="Uploaded image", width="stretch")

        with control_col:
            st.markdown(
                """
                <div class="glass-card">
                    <div style="color:#00e5ff;font-size:11px;letter-spacing:2px;font-weight:900;">READY TO SCAN</div>
                    <div style="font-size:23px;font-weight:900;color:#fff;margin-top:8px;">Let's investigate it.</div>
                    <div class="small-note" style="margin-top:8px;">
                        One EfficientNet pass feeds both the direct classifier and the hybrid feature pipeline.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            analyze = st.button(
                "🔎 ANALYZE THIS IMAGE",
                width="stretch",
                type="primary",
            )

            if analyze:
                with st.spinner(f"I'm examining the image now, {st.session_state.user_name}... 🔬"):
                    result = predict_image(
                        image,
                        EFF_MODEL,
                        DNN_MODEL,
                        FEATURE_MODEL,
                        THRESHOLD,
                        EFF_WEIGHT,
                        DNN_WEIGHT,
                    )

                result["filename"] = uploaded.name
                result["time"] = datetime.now().strftime("%H:%M:%S")
                st.session_state.last_result = result
                st.session_state.scan_count += 1
                st.session_state.scan_history.insert(0, result.copy())
                st.session_state.scan_history = st.session_state.scan_history[:10]
                st.toast(
                    f"Analysis complete, {st.session_state.user_name}. 🦊",
                    icon="✅",
                )

        # Show latest result
        if st.session_state.last_result:
            result = st.session_state.last_result
            ai = result["label"] == "AI-GENERATED"
            result_class = "ai-text" if ai else "real-text"
            score_pct = result["model_score"] * 100
            margin_pct = result["margin"] * 100

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">MODEL VERDICT</div>
                    <div class="result-main {result_class}">{result['label']}</div>
                    <div class="small-note" style="margin-top:10px;">
                        The model score for this verdict is <strong style="color:#fff;">{score_pct:.2f}%</strong>.
                        This is a model output, not absolute proof.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">HYBRID AI SCORE</div><div class="metric-value">{result["hybrid_ai_score"]*100:.2f}%</div><div class="metric-note">fused detector output</div></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">THRESHOLD</div><div class="metric-value">{result["threshold"]:.2f}</div><div class="metric-note">validation-selected boundary</div></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">SCORE MARGIN</div><div class="metric-value">{margin_pct:.2f}%</div><div class="metric-note">distance from boundary</div></div>',
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">SESSION SCANS</div><div class="metric-value">{st.session_state.scan_count}</div><div class="metric-note">images analyzed</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### 🔬 Inside the decision")
            c1, c2 = st.columns(2, gap="large")

            with c1:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="color:#00e5ff;font-weight:900;letter-spacing:1.2px;font-size:11px;">COMPONENT SCORES</div>
                        <div style="margin-top:13px;color:#d8e2f0;">EfficientNet-B0: <strong>{result['efficientnet_score']*100:.2f}% AI</strong></div>
                        <div style="margin-top:9px;color:#d8e2f0;">DNN branch: <strong>{result['dnn_score']*100:.2f}% AI</strong></div>
                        <div style="margin-top:9px;color:#d8e2f0;">Hybrid fusion: <strong>{result['hybrid_ai_score']*100:.2f}% AI</strong></div>
                        <div class="small-note" style="margin-top:12px;">Fusion weights: EfficientNet {EFF_WEIGHT:.2f} • DNN {DNN_WEIGHT:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                if result["margin"] < 0.08:
                    st.warning(
                        "This result is close to the decision boundary. Treat the verdict cautiously; the model is less separated from its threshold here."
                    )
                elif ai:
                    st.info(
                        "The hybrid score is above the configured threshold, so the system labels this image AI-generated."
                    )
                else:
                    st.info(
                        "The hybrid score is below the configured threshold, so the system labels this image REAL."
                    )

            if st.button("↻ Analyze Another Image", width="stretch"):
                st.session_state.last_result = None
                st.rerun()

            # Session history
            if st.session_state.scan_history:
                st.markdown("### 🕘 Recent investigations")
                history_df = pd.DataFrame(
                    [
                        {
                            "Time": r["time"],
                            "Image": r["filename"],
                            "Verdict": r["label"],
                            "Model Score": f"{r['model_score']*100:.2f}%",
                            "Hybrid AI Score": f"{r['hybrid_ai_score']*100:.2f}%",
                        }
                        for r in st.session_state.scan_history
                    ]
                )
                st.dataframe(history_df, width="stretch", hide_index=True)

# ============================================================
# MODEL INTELLIGENCE TAB
# ============================================================

with tab_model:
    st.markdown("### ◈ Model Intelligence")
    st.markdown(
        """
        <div class="chat-bubble">
            <strong>Here's what the lab numbers actually say.</strong><br>
            These are held-out V2 benchmark results, not marketing confetti.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_values = [
        ("TEST ACCURACY", "94.79%", "held-out V2 test set"),
        ("MACRO F1", "88.91%", "class-balanced summary"),
        ("BALANCED ACC", "87.34%", "real + AI recall balance"),
        ("ROC-AUC", "96.69%", "ranking performance"),
        ("PR-AUC", "99.32%", "precision-recall area"),
    ]

    cols = st.columns(len(metric_values))
    for col, (label, value, note) in zip(cols, metric_values):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Generator-wise AI detection")
    gen_data = pd.DataFrame(
        {
            "Generator": ["FLUX.1-schnell", "Kandinsky 2.2", "PixArt-Σ", "SD 1.5", "SDXL", "Würstchen"],
            "AI Detection Recall (%)": [97.33, 99.60, 99.07, 94.59, 97.66, 98.40],
        }
    ).set_index("Generator")
    st.bar_chart(gen_data, width="stretch")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="color:#00e5ff;font-size:11px;font-weight:900;letter-spacing:1.5px;">REAL IMAGE BEHAVIOR</div>
                <div style="font-size:32px;font-weight:950;color:#fff;margin-top:8px;">76.90%</div>
                <div style="color:#8e9db3;margin-top:5px;">Real-image recall on the V2 test set.</div>
                <div class="small-note" style="margin-top:10px;">That means false positives remain a meaningful limitation, especially on some human-centric real images.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <div style="color:#a78bfa;font-size:11px;font-weight:900;letter-spacing:1.5px;">CONFUSION MATRIX</div>
                <div style="margin-top:12px;color:#c7d3e5;line-height:1.8;">
                    Actual REAL → Predicted REAL: <strong>1152</strong><br>
                    Actual REAL → Predicted AI: <strong>346</strong><br>
                    Actual AI → Predicted REAL: <strong>200</strong><br>
                    Actual AI → Predicted AI: <strong>8788</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# ROBUSTNESS LAB TAB
# ============================================================

with tab_robust:
    st.markdown("### ◉ Robustness Lab")
    st.markdown(
        """
        <div class="chat-bubble">
            <strong>Reality gets messy outside the clean benchmark.</strong><br>
            JPEG compression and resizing can change pixels without changing the underlying image.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists(ROBUSTNESS_PATH):
        try:
            robust_df = pd.read_csv(ROBUSTNESS_PATH)
            st.dataframe(robust_df, width="stretch", hide_index=True)

            numeric_cols = [c for c in robust_df.columns if c.lower() not in {"variant", "name", "condition"}]
            score_col = None
            for candidate in ["accuracy", "macro_f1", "balanced_accuracy", "ai_recall"]:
                found = [c for c in robust_df.columns if c.lower() == candidate]
                if found:
                    score_col = found[0]
                    break

            if score_col:
                chart_df = robust_df.copy()
                index_col = chart_df.columns[0]
                chart_df = chart_df.set_index(index_col)[[score_col]]
                st.line_chart(chart_df, width="stretch")

        except Exception as exc:
            st.error(f"I found the robustness file, but couldn't read it: {exc}")
    else:
        st.info(
            "No `/kaggle/working/robustness_summary.csv` file was found yet. "
            "Run the robustness experiment first and this panel will populate automatically."
        )

# ============================================================
# BATCH TAB
# ============================================================

with tab_batch:
    st.markdown("### ▦ Batch Analysis")
    st.markdown(
        f"""
        <div class="chat-bubble">
            <strong>Got a whole pile of images, {st.session_state.user_name}?</strong><br>
            Upload them together and I'll run the same hybrid detector over the batch.
        </div>
        """,
        unsafe_allow_html=True,
    )

    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        if st.button("⚡ ANALYZE FULL BATCH", width="stretch", type="primary"):
            rows = []
            progress = st.progress(0)

            for idx, file in enumerate(batch_files, start=1):
                try:
                    img = Image.open(file).convert("RGB")
                    result = predict_image(
                        img,
                        EFF_MODEL,
                        DNN_MODEL,
                        FEATURE_MODEL,
                        THRESHOLD,
                        EFF_WEIGHT,
                        DNN_WEIGHT,
                    )
                    rows.append(
                        {
                            "Image": file.name,
                            "Verdict": result["label"],
                            "Model Score %": round(result["model_score"] * 100, 2),
                            "Hybrid AI %": round(result["hybrid_ai_score"] * 100, 2),
                            "EfficientNet AI %": round(result["efficientnet_score"] * 100, 2),
                            "DNN AI %": round(result["dnn_score"] * 100, 2),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "Image": file.name,
                            "Verdict": "ERROR",
                            "Model Score %": None,
                            "Hybrid AI %": None,
                            "EfficientNet AI %": None,
                            "DNN AI %": None,
                            "Error": str(exc),
                        }
                    )
                progress.progress(idx / len(batch_files))

            batch_df = pd.DataFrame(rows)
            st.success(f"Batch analysis complete. {len(batch_files)} image(s) checked.")
            st.dataframe(batch_df, width="stretch", hide_index=True)

            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Batch Results CSV",
                data=csv_bytes,
                file_name="illusion_of_reality_batch_results.csv",
                mime="text/csv",
                width="stretch",
            )

# ============================================================
# SYSTEM TAB
# ============================================================

with tab_system:
    st.markdown("### ◆ System")

    st.markdown(
        """
        <div class="glass-card">
            <div style="color:#00e5ff;font-size:11px;letter-spacing:1.7px;font-weight:900;">ARCHITECTURE</div>
            <div style="font-size:23px;font-weight:900;color:#fff;margin-top:8px;">Image → EfficientNet-B0 → DNN → Hybrid Fusion</div>
            <div class="small-note" style="margin-top:10px;">
                The application uses the saved V2 EfficientNet-B0 classifier, extracts the 1280-dimensional representation,
                feeds that representation into the saved DNN branch, and combines the two AI scores using validation-selected fusion weights.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Runtime configuration")
    config_df = pd.DataFrame(
        {
            "Parameter": [
                "Input size",
                "EfficientNet weight",
                "DNN weight",
                "Decision threshold",
                "EfficientNet model",
                "DNN model",
                "Config file",
            ],
            "Value": [
                "256 × 256",
                f"{EFF_WEIGHT:.4f}",
                f"{DNN_WEIGHT:.4f}",
                f"{THRESHOLD:.4f}",
                os.path.basename(EFF_PATH),
                os.path.basename(DNN_PATH),
                os.path.basename(CONFIG_PATH),
            ],
        }
    )
    st.dataframe(config_df, width="stretch", hide_index=True)

    st.markdown("### Research notes")
    st.info(
        "The final V2 test performance was 94.79% accuracy, 88.91% Macro F1, 87.34% balanced accuracy, "
        "96.69% ROC-AUC and 99.32% PR-AUC. The model also showed meaningful false positives on real images, "
        "so the application deliberately avoids claiming that every prediction is certain or universally correct."
    )

    st.markdown("### Session")
    st.write(f"Current investigator: **{st.session_state.user_name}**")
    st.write(f"Images analyzed this session: **{st.session_state.scan_count}**")

    if st.button("↪ Change Investigator / Reset Session", width="stretch"):
        reset_session()

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-line">
        THE ILLUSION OF REALITY • UPLOAD • ANALYZE • QUESTION WHAT YOU SEE
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('</div>', unsafe_allow_html=True)
