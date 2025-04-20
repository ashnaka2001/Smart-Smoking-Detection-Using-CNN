import streamlit as st
import cv2
import numpy as np
import winsound
from keras.models import load_model
from PIL import Image
import time

st.set_page_config(page_title="🚭 Smart Smoking Detection", page_icon="🚬", layout="wide")

st.markdown("""
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #ddeeff);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Headings */
    .smoke-title {
        font-size: 42px;
        font-weight: 800;
        color: #d90429;
        text-shadow: 1px 1px 2px #999;
    }

    .big-font {
        font-size: 22px !important;
        font-weight: 600;
        color: #2b2d42;
    }

    /* Rounded image */
    .rounded-img {
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #edf2f4;
    }

    /* Button enhancements */
    button[kind="primary"] {
        background-color: #ef233c;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }

    button[kind="secondary"] {
        border-radius: 8px;
    }

    /* Webcam info message */
    .stAlert > div {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_smoking_model():
    try:
        return load_model("smoking_detection_model.h5", compile=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_smoking_model()
categories = ['Not_Smoking', 'Smoking']


def predict_smoking(img):
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, (150, 150))
        reshaped_img = np.stack((resized_img,) * 3, axis=-1).reshape(1, 150, 150, 3)
        reshaped_img = reshaped_img.astype('float32') / 255.0
        prediction = model.predict(reshaped_img)
        if prediction.size > 0:
            return categories[np.argmax(prediction)]
        else:
            return "Not_Smoking"  # Default fallback

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return "Not_Smoking"

def play_alert_sound():
    winsound.Beep(1000, 300)


st.sidebar.title("🚦 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧠 Smoking Detection"])
sound_alert = st.sidebar.checkbox("🔊 Enable Sound Alert", value=True)

if page == "🏠 Home":
    st.markdown('<div class="smoke-title"> Smart Smoking Detection 🚭</div><br>', unsafe_allow_html=True)
    st.markdown("### 🔍 Detect smoking in real-time using deep learning.", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        #st.image("Smoking-2 (1).png", use_container_width=True, caption="🚬 Stop Smoking. Start Living.")
        st.image("Smoking-2 (1).png", use_container_width=True)
        st.markdown('<p style="text-align:center; color:black; font-weight:500;">🚭 Stop Smoking. Start Living.</p>',
                    unsafe_allow_html=True)

    with col2:
        st.markdown("""
        #### 🚀 Features
        - 💡 Deep learning-powered predictions
        - 🖼️ Image upload & webcam support
        - 🚨 Real-time alerts with sound
        - 📊 Fast, lightweight, and accurate
        """, unsafe_allow_html=True)
        st.info("👈 Use the sidebar to start detection!")


elif page == "🧠 Smoking Detection":
    st.markdown('<div class="smoke-title">🧪 Real-Time Detection Zone</div>', unsafe_allow_html=True)
    method = st.radio("Choose Detection Method:", ["📷 Upload an Image", "📹 Use Webcam"])

    if method == "📷 Upload an Image":
        uploaded_file = st.file_uploader("📁 Upload a photo", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, channels="BGR", caption="📸 Uploaded Image", use_container_width=True)

            with st.spinner("🔍 Analyzing image..."):
                time.sleep(1)
                category = predict_smoking(img)

            st.success(f"🎯 Prediction: **{category}**")

            if category == "Smoking":
                st.warning("🚨 Smoking Detected!")
                if sound_alert:
                    play_alert_sound()

    elif method == "📹 Use Webcam":
        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start Webcam"):
                st.session_state.webcam_running = True
        with col2:
            if st.button("⏹ Stop Webcam"):
                st.session_state.webcam_running = False

        frame_display = st.empty()

        if st.session_state.webcam_running:
            video = cv2.VideoCapture(0)
            st.info("🔴 Webcam running...")

            while st.session_state.webcam_running:
                success, frame = video.read()
                if not success:
                    st.error("⚠️ Could not read from webcam.")
                    break

                category = predict_smoking(frame)
                cv2.putText(frame, category, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255) if category == "Smoking" else (0, 200, 0), 3)
                frame_display.image(frame, channels="BGR", use_container_width=True)

                if category == "Smoking":
                    st.warning("🚨 Smoking Detected!")
                    if sound_alert:
                        play_alert_sound()

            video.release()
            cv2.destroyAllWindows()
