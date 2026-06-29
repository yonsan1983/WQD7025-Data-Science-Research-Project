import os
import time
import cv2
import torch
import timm
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# ==============================================================================
# 1. PAGE CONFIGURATION & ACADEMIC STYLE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Facial Emotion Recognition Demo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to provide a clean, professional, academic aesthetic
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #f8fafc;
    }
    
    /* Academic title styling */
    .project-header {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 20px;
        border-bottom: 2px solid #e2e8f0;
    }
    .project-title {
        color: #1e293b;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .project-subtitle {
        color: #475569;
        font-size: 1.1rem;
        font-style: italic;
        margin-bottom: 15px;
    }
    
    /* Box layouts for results */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    
    /* Highlights for predicted class */
    .prediction-highlight {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2563eb;
        text-align: center;
        margin: 10px 0;
    }
    
    .inference-meta {
        font-size: 0.95rem;
        color: #64748b;
        text-align: center;
        border-top: 1px solid #f1f5f9;
        padding-top: 8px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Render academic title and header
st.markdown("""
<div class="project-header">
    <div class="project-title">A Comparative Analysis of Deep Learning Models for Facial Emotion Recognition</div>
    <div class="project-subtitle">Master in Data Science Final Year Project — Deployment Demonstration</div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CONSTANTS & EMOTION LABELS
# ==============================================================================
# Order corresponds exactly to CLASS_MAPPING index order in training notebook
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

# ImageNet standardization values used in training pipeline
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Supported architectures dropdown map
ARCHITECTURES = {
    "ViT-Tiny": "vit_tiny_patch16_224",
    "ResNet18": "resnet18",
    "MambaVision-Tiny": "mamba_vision_T",
    "MambaVision-Base": "mamba_vision_B",
    "ViT-Small": "vit_small_patch16_224",
    "ViT-Base": "vit_base_patch16_224"
}

# ==============================================================================
# 3. SIDEBAR CONFIGURATION
# ==============================================================================
st.sidebar.header("⚙️ Model Configuration")

# Model Architecture selector
selected_arch_label = st.sidebar.selectbox(
    "Model Architecture",
    options=list(ARCHITECTURES.keys()),
    index=0,  # Default to ViT-Tiny
    help="Select the architecture template matching your trained checkpoint file."
)
selected_arch_name = ARCHITECTURES[selected_arch_label]

# Checkpoint path text entry
checkpoint_path = st.sidebar.text_input(
    "Model Checkpoint Path",
    value="vit_tiny_best.pth",
    help="Path to the trained .pth file relative to the project folder."
)

# Hardware execution device setup
device_option = st.sidebar.selectbox(
    "Execution Device",
    options=["Auto Detect (GPU/CPU)", "Force CPU"],
    index=0
)

# Determine the actual torch device to use
if device_option == "Force CPU":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Show current device in sidebar
st.sidebar.info(f"Using Device: **{str(device).upper()}**")

# Face Detection Settings
st.sidebar.header("👤 Face Detection Settings")
enable_face_crop = st.sidebar.checkbox(
    "Enable Face Detection & Cropping",
    value=True,
    help="If checked, crops detected faces before running emotion prediction. "
         "If unchecked or face detection fails, processes the whole image."
)
min_neighbors = st.sidebar.slider(
    "Haar Cascade Sensitivity (minNeighbors)",
    min_value=3, max_value=10, value=5,
    help="Higher values detect fewer but higher-quality faces."
)
scale_factor = st.sidebar.slider(
    "Haar Cascade Scale Factor",
    min_value=1.05, max_value=1.4, value=1.1, step=0.05,
    help="Image reduction size factor at each image scale pass."
)

# Show Project info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About This App
This interface demonstrates the deployment viability of deep learning models for
**Facial Emotion Recognition (FER)**. 

During presentation, use this side panel to load different models (e.g., ViT-Tiny, ResNet18) and compare their predictions, confidences, and inference latencies.
""")

# ==============================================================================
# 4. CORE PIPELINE FUNCTIONS
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_emotion_model(arch_name, checkpoint_path, target_device):
    """
    Instantiates the model architecture, loads the state dictionary weights from
    the checkpoint file, resolves any common classifier key mismatches, and loads
    the model onto the selected hardware device. Cached to prevent reloading on rerun.
    """
    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: '{checkpoint_path}'")

    # Instantiate model using timm or mambavision
    if "mamba" in arch_name.lower():
        try:
            from mambavision import create_model as create_mamba_model
            # Instantiate custom MambaVision structure with 6 output classes
            model = create_mamba_model(arch_name, pretrained=False, num_classes=6)
        except ImportError as e:
            raise ImportError(
                "MambaVision package is not installed. To evaluate MambaVision, "
                "please run `pip install mambavision` or load a standard timm model instead."
            ) from e
    else:
        # Instantiate standard timm model structure with 6 output classes
        model = timm.create_model(arch_name, pretrained=False, num_classes=6)

    # Load state dict onto CPU first, then transfer to target device
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Handle state dict class key mismatches common in fine-tuned ResNets
    # (e.g., fc.1.weight -> fc.weight)
    if "fc.1.weight" in state_dict:
        state_dict["fc.weight"] = state_dict.pop("fc.1.weight")
    if "fc.1.bias" in state_dict:
        state_dict["fc.bias"] = state_dict.pop("fc.1.bias")

    # Load the weights into the model
    model.load_state_dict(state_dict, strict=False)
    model.to(target_device)
    model.eval()
    
    return model


@st.cache_resource(show_spinner=False)
def load_face_cascade():
    """
    Loads OpenCV's Haar Cascade classifier for frontal face detection.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar Cascade classifier.")
    return face_cascade


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Applies the standardized data preprocessing pipeline matching the training phase:
    1. Convert PIL image to RGB.
    2. Resize to 224x224.
    3. Convert pixel values to Float Tensor [0.0, 1.0].
    4. Standardize with ImageNet mean and standard deviation.
    5. Add batch dimension (1, C, H, W).
    """
    # 1. Ensure RGB mode (removes alpha channel if present)
    img_rgb = image.convert("RGB")
    
    # 2. Resize to 224 x 224
    img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
    
    # 3. Convert to float numpy array normalized between [0, 1]
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Convert from HWC to CHW format
    img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1)))
    
    # 4. Normalize channel-wise using standard ImageNet mean and std
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img_normalized = (img_tensor - mean) / std
    
    # 5. Add batch dimension (Batch size = 1)
    img_batch = img_normalized.unsqueeze(0)
    
    return img_batch


def predict_emotion(preprocessed_tensor: torch.Tensor, model: torch.nn.Module, target_device: torch.device):
    """
    Runs model inference on a preprocessed face tensor.
    Tracks precise execution latency in milliseconds and outputs emotion probabilities.
    """
    # Transfer input to matching execution hardware
    x = preprocessed_tensor.to(target_device)
    
    # Synchronize CUDA operations if using GPU for accurate time profiling
    if target_device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    
    # Disable gradient tracking for inference speed and memory optimization
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        
    if target_device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    # Calculate inference latency in milliseconds
    inference_time_ms = (end_time - start_time) * 1000
    
    # Pull scores back to CPU numpy array
    probs_np = probabilities.cpu().numpy()
    
    # Get index of highest confidence class
    pred_idx = int(np.argmax(probs_np))
    pred_label = EMOTIONS[pred_idx]
    pred_confidence = float(probs_np[pred_idx])
    
    # Create dictionary mapping labels to confidence scores
    emotions_dict = {EMOTIONS[i]: float(probs_np[i]) for i in range(len(EMOTIONS))}
    
    return pred_label, pred_confidence, emotions_dict, inference_time_ms

# ==============================================================================
# 5. MODEL INITIALIZATION AND ERROR HANDLING
# ==============================================================================
model_loaded = False
model = None

try:
    with st.spinner(f"Loading {selected_arch_label} model from '{checkpoint_path}'..."):
        model = load_emotion_model(selected_arch_name, checkpoint_path, device)
        model_loaded = True
except FileNotFoundError as e:
    st.error(f"⚠️ **Model Loading Failed**: File not found.\n\n*Error details: {e}*")
    st.info("💡 **Instructions**: Please verify your model checkpoint path in the sidebar configuration. "
            "Make sure the `.pth` file is located in the current workspace or provide its absolute path.")
except Exception as e:
    st.error(f"⚠️ **Model Loading Failed**: Incompatible architecture or corrupted file.\n\n*Error details: {e}*")
    st.info("💡 **Instructions**: Check that your chosen Model Architecture dropdown corresponds with "
            "the architecture of the file weights you are trying to load.")

# Load face detector
try:
    face_cascade = load_face_cascade()
    detector_loaded = True
except Exception as e:
    detector_loaded = False
    st.warning(f"⚠️ **Face detector initialization failed**: {e}. "
               "Face cropping is disabled; full images will be sent directly to the model.")

# ==============================================================================
# 6. APPLICATION WORKFLOW
# ==============================================================================
if model_loaded:
    # Set up UI layouts: left col for inputs, right col for predictions
    col_input, col_pred = st.columns([1, 1])
    
    with col_input:
        st.subheader("📸 Input Interface")
        
        # Dual-input tabs with real-time stream option
        input_tab = st.radio(
            "Select Input Source:",
            options=["📤 Upload Image File", "📷 Take Webcam Photo", "🎥 Live Video Stream (Real-time)"],
            horizontal=True
        )
        
        input_image = None
        
        if input_tab == "📤 Upload Image File":
            uploaded_file = st.file_uploader(
                "Choose a face image...", 
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPEG, PNG"
            )
            if uploaded_file is not None:
                try:
                    input_image = Image.open(uploaded_file)
                except Exception as e:
                    st.error(f"Error opening image file: {e}")
            if input_image is not None:
                st.image(input_image, caption="Original Input Preview", use_column_width=True)
                
        elif input_tab == "📷 Take Webcam Photo":
            # Active camera input widget
            camera_file = st.camera_input("Capture a live face frame for classification")
            if camera_file is not None:
                try:
                    input_image = Image.open(camera_file)
                except Exception as e:
                    st.error(f"Error reading webcam frame: {e}")
            if input_image is not None:
                st.image(input_image, caption="Original Input Preview", use_column_width=True)
                
        else: # "🎥 Live Video Stream (Real-time)"
            start_stream = st.checkbox("Start Live Video Stream", value=False)
            frame_placeholder = st.empty()
            if not start_stream:
                st.info("Check 'Start Live Video Stream' above to launch your webcam.")
            
    with col_pred:
        st.subheader("📊 Emotion Analysis Results")
        
        if input_tab == "🎥 Live Video Stream (Real-time)":
            realtime_status = st.empty()
            emotion_metric = st.empty()
            fps_metric = st.empty()
            chart_placeholder = st.empty()
            crop_preview_placeholder = st.empty()
            
            if start_stream:
                # Open standard webcam video capture (index 0)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Error: Could not access the webcam. Please ensure it is connected and not in use by another app.")
                else:
                    prev_time = time.time()
                    frame_count = 0
                    try:
                        while start_stream:
                            ret, frame = cap.read()
                            if not ret:
                                realtime_status.error("Failed to grab webcam frame.")
                                break
                            
                            frame_count += 1
                            # Flip frame horizontally for natural mirror behavior
                            frame = cv2.flip(frame, 1)
                            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            detection_successful = False
                            face_cropped_image = None
                            
                            if enable_face_crop and detector_loaded:
                                faces = face_cascade.detectMultiScale(
                                    gray_img, 
                                    scaleFactor=scale_factor, 
                                    minNeighbors=min_neighbors, 
                                    minSize=(40, 40)
                                )
                                if len(faces) > 0:
                                    # Select the largest detected face
                                    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                                    x, y, w, h = faces_sorted[0]
                                    cropped_np = frame_rgb[y:y+h, x:x+w]
                                    face_cropped_image = Image.fromarray(cropped_np)
                                    detection_successful = True
                                    
                                    # Draw Royal Blue border around the face
                                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (37, 99, 235), 3)
                            
                            # Calculate FPS
                            curr_time = time.time()
                            elapsed = curr_time - prev_time
                            fps = 1.0 / elapsed if elapsed > 0 else 0.0
                            prev_time = curr_time
                            
                            if detection_successful and face_cropped_image is not None:
                                try:
                                    tensor_input = preprocess_image(face_cropped_image)
                                    pred_label, pred_confidence, probs_dict, latency = predict_emotion(
                                        tensor_input, model, device
                                    )
                                    
                                    # Overlay prediction details on the video feed frame
                                    text = f"{pred_label} ({pred_confidence * 100:.1f}%)"
                                    (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                                    cv2.rectangle(frame_rgb, (x, y - h_text - 15), (x + w_text, y), (37, 99, 235), cv2.FILLED)
                                    cv2.putText(frame_rgb, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                                    
                                    # Render metrics onto prediction column placeholders
                                    emotion_metric.markdown(f"""
                                    <div class="metric-card" style="text-align: center;">
                                        <div style="color: #475569; font-weight: 600; font-size: 1rem;">Detected Emotion</div>
                                        <div class="prediction-highlight">{pred_label}</div>
                                        <div class="confidence-text">Confidence: <b>{pred_confidence * 100:.2f}%</b></div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    fps_metric.markdown(f"⚡ **Inference Latency:** {latency:.1f} ms | 📷 **Stream Speed:** {fps:.1f} FPS")
                                    
                                    # Render charts at a throttled rate to maintain high framerate
                                    if frame_count % 3 == 0:
                                        df = pd.DataFrame({
                                            "Emotion": list(probs_dict.keys()),
                                            "Probability": list(probs_dict.values())
                                        }).set_index("Emotion")
                                        chart_placeholder.bar_chart(df)
                                        
                                        with crop_preview_placeholder.container():
                                            col_c1, col_c2 = st.columns([1, 2])
                                            with col_c1:
                                                st.image(face_cropped_image, caption="Face Crop", use_column_width=True)
                                                
                                except Exception as eval_err:
                                    realtime_status.warning(f"Inference error: {eval_err}")
                            else:
                                # Fallback view when no face is active
                                emotion_metric.markdown("""
                                <div class="metric-card" style="text-align: center; padding: 25px;">
                                    <h4 style="color: #64748b;">No Face Detected</h4>
                                    <span style="font-size: 0.9rem; color: #94a3b8;">Adjust lighting or face position</span>
                                </div>
                                """, unsafe_allow_html=True)
                                fps_metric.markdown(f"📷 **Stream Speed:** {fps:.1f} FPS (Inference idle)")
                            
                            # Draw FPS display banner in frame header
                            cv2.putText(frame_rgb, f"FPS: {fps:.1f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 197, 94), 2, cv2.LINE_AA)
                            
                            # Render image frame in place of original preview
                            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                            
                            # Control yield time
                            time.sleep(0.01)
                            
                    finally:
                        cap.release()
                        frame_placeholder.empty()
                        emotion_metric.empty()
                        fps_metric.empty()
                        chart_placeholder.empty()
                        crop_preview_placeholder.empty()
                        
        else: # Static Image prediction mode (Upload / Take Photo)
            if input_image is None:
                st.info("👈 Please upload an image or capture a webcam photo to run model inference.")
            else:
                # Check for face detection
                face_cropped_image = None
                detection_successful = False
                
                # Draw boundaries box to overlay on user preview
                preview_image_np = np.array(input_image.convert("RGB"))
                
                if enable_face_crop and detector_loaded:
                    # Haar cascade requires grayscale format
                    gray_img = cv2.cvtColor(preview_image_np, cv2.COLOR_RGB2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray_img, 
                        scaleFactor=scale_factor, 
                        minNeighbors=min_neighbors, 
                        minSize=(40, 40)
                    )
                    
                    if len(faces) > 0:
                        # Crop the largest face (typically the main subject)
                        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        x, y, w, h = faces_sorted[0]
                        
                        # Crop face from original image
                        cropped_np = preview_image_np[y:y+h, x:x+w]
                        face_cropped_image = Image.fromarray(cropped_np)
                        detection_successful = True
                        
                        # Draw a bounding rectangle on original preview image
                        cv2.rectangle(preview_image_np, (x, y), (x+w, y+h), (37, 99, 235), 3) # Royal Blue border
                        st.success("✔️ Face successfully localized in image frame.")
                    else:
                        st.warning("⚠️ **Face Detection Warning**: No face detected in the frame. "
                                   "Processing the full image directly.")
                
                # Fallback to full image if face cropping is disabled or detection failed
                if face_cropped_image is None:
                    face_cropped_image = input_image
                    
                # Perform prediction
                try:
                    # Preprocess cropped region (or full image)
                    tensor_input = preprocess_image(face_cropped_image)
                    
                    # Perform model prediction and log runtime
                    pred_label, pred_confidence, probs_dict, latency = predict_emotion(tensor_input, model, device)
                    
                    # Setup visual containers for the main prediction metrics
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="text-align: center; color: #475569; font-weight: 600; font-size: 1.1rem;">
                            Predicted Emotion Label
                        </div>
                        <div class="prediction-highlight">{pred_label}</div>
                        <div class="confidence-text">
                            Confidence: <b>{pred_confidence * 100:.2f}%</b>
                        </div>
                        <div class="inference-meta">
                            ⚡ <b>Inference Latency:</b> {latency:.2f} ms &nbsp; | &nbsp; ⚙️ <b>Device:</b> {str(device).upper()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display probability bar chart using pandas dataframe
                    df = pd.DataFrame({
                        "Emotion": list(probs_dict.keys()),
                        "Probability": list(probs_dict.values())
                    })
                    
                    # Set label as index to prevent Streamlit from plotting index integers
                    df = df.set_index("Emotion")
                    
                    st.write("**Probability Distribution across Classes:**")
                    st.bar_chart(df)
                    
                    # Visual comparison of original image with bounding box (if face detected) and the crop
                    if detection_successful:
                        col_det_1, col_det_2 = st.columns(2)
                        with col_det_1:
                            st.image(
                                preview_image_np, 
                                caption="Detected Bounding Box", 
                                use_column_width=True
                            )
                        with col_det_2:
                            st.image(
                                face_cropped_image, 
                                caption="Crop sent to Deep Learning Model", 
                                use_column_width=True
                            )
                    
                except Exception as eval_err:
                    st.error(f"🔴 **Evaluation Error**: An error occurred during image feedforward processing.\n\n*Error details: {eval_err}*")
                    st.info("💡 Make sure your image input is non-corrupted and model checkpoint settings are correct.")
