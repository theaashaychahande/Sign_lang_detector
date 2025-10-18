import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import time

try:
    from utils import ASLDetector
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

st.set_page_config(
    page_title="ASL Sign Language Detector",
    page_icon="✋",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .confidence-bar {
        background-color: #1f77b4;
        height: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">✋ ASL Sign Language Detector</h1>', unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Home", "Real-time Detection", "Image Upload", "Model Info"])
    
    if 'detector' not in st.session_state:
        st.session_state.detector = ASLDetector()
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "Real-time Detection":
        show_realtime_detection()
    elif app_mode == "Image Upload":
        show_image_upload()
    elif app_mode == "Model Info":
        show_model_info()

def show_home():
    st.markdown("""
    ## Welcome to the ASL Sign Language Detector! 🎉
    
    This app can recognize American Sign Language (ASL) letters a-z and numbers 0-9 in real-time using your webcam.
    
    ### How to use:
    1. **Real-time Detection**: Use your webcam to show ASL signs
    2. **Image Upload**: Upload an image containing ASL signs
    3. **Model Info**: Check what signs the model can recognize
    
    ### Features:
    - ✅ Real-time hand detection
    - ✅ ASL alphabet (a-z) recognition
    - ✅ ASL numbers (0-9) recognition
    - ✅ Confidence scoring
    - ✅ Free and open source
    
    ### Get Started:
    Select **"Real-time Detection"** from the sidebar to start using your webcam!
    """)
    
    st.subheader("Supported Signs")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Alphabet (a-z)**")
        st.write("a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z")
    
    with col2:
        st.write("**Numbers (0-9)**")
        st.write("0, 1, 2, 3, 4, 5, 6, 7, 8, 9")

def show_realtime_detection():
    st.header("Real-time ASL Detection")
    
    if st.session_state.detector.model is None:
        st.error("❌ Model not found! Please train the model first.")
        st.info("""
        To train the model:
        1. Run: `python train_model.py`
        2. Make sure your ASL dataset is in the `asl_dataset/` folder
        3. The model will be saved as `model/asl_model.pkl`
        """)
        return
    
    st.info("📷 Click 'Start Camera' to begin real-time detection. Make sure to show one hand at a time.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        run_camera = st.checkbox("Start Camera")
        FRAME_WINDOW = st.image([])
        
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        if run_camera:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Cannot access camera. Please check if it's being used by another application.")
                return
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                frame = cv2.flip(frame, 1)
                
                prediction, confidence, hand_landmarks = st.session_state.detector.predict(frame)
                
                if hand_landmarks:
                    frame = st.session_state.detector.draw_landmarks(frame, hand_landmarks)
                
                if prediction != "No hand detected" and prediction != "Model not loaded":
                    with prediction_placeholder.container():
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Prediction: <span style="color: #1f77b4;">{prediction}</span></h3>
                            <p>Confidence: {confidence:.2%}</p>
                            <div class="confidence-bar" style="width: {confidence * 100}%"></div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    prediction_placeholder.info("👋 Show your hand to the camera")
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)
                
                time.sleep(0.1)
            
            cap.release()
        else:
            st.info("👆 Check 'Start Camera' to begin")

def show_image_upload():
    st.header("Upload ASL Image")
    
    if st.session_state.detector.model is None:
        st.error("❌ Model not found! Please train the model first.")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Analysis")
            
            prediction, confidence, hand_landmarks = st.session_state.detector.predict(image_np)
            
            if hand_landmarks:
                annotated_image = st.session_state.detector.draw_landmarks(image_np, hand_landmarks)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                st.subheader("Annotated Image")
                st.image(annotated_image_rgb, use_column_width=True)
            
            if prediction != "No hand detected" and prediction != "Model not loaded":
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Prediction: <span style="color: #1f77b4; font-size: 2em;">{prediction}</span></h2>
                    <p style="font-size: 1.2em;">Confidence: {confidence:.2%}</p>
                    <div class="confidence-bar" style="width: {confidence * 100}%"></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ No hand detected in the image")
                st.info("Please upload an image that clearly shows a hand making an ASL sign")

def show_model_info():
    st.header("Model Information")
    
    if st.session_state.detector.model is None:
        st.error("❌ Model not found! Please train the model first.")
        st.info("""
        To train the model:
        1. Make sure you have the `asl_dataset` folder with 0-9 and a-z subfolders
        2. Run: `python train_model.py`
        3. Wait for training to complete
        """)
        return
    
    st.success(f"✅ Model loaded successfully!")
    st.write(f"**Number of classes:** {len(st.session_state.detector.classes)}")
    st.write(f"**Supported signs:** {', '.join(sorted(st.session_state.detector.classes))}")
    
    st.subheader("How to Train/Retrain the Model")
    
    if st.button("🚀 Train Model Now"):
        with st.spinner("Training model... This may take a few minutes."):
            import subprocess
            try:
                result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    st.success("✅ Model trained successfully!")
                    st.text_area("Training Output", result.stdout, height=200)
                    
                    st.session_state.detector = ASLDetector()
                    st.rerun()
                else:
                    st.error("❌ Training failed!")
                    st.text_area("Error Output", result.stderr, height=200)
            except subprocess.TimeoutExpired:
                st.error("❌ Training timed out! This might be normal for large datasets.")
            except Exception as e:
                st.error(f"❌ Training error: {e}")

if __name__ == "__main__":
    main()
