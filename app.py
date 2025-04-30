import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="COVID-19 X-Ray Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Define a simpler color scheme
PRIMARY_COLOR = "#4361ee"
SECONDARY_COLOR = "#3a0ca3"
POSITIVE_COLOR = "#ef476f"
NEGATIVE_COLOR = "#06d6a0"
BG_COLOR = "#f8f9fa"

# Custom CSS for a clean, simple interface with visual appeal
st.markdown(
    f"""
<style>
    /* Main container styling */
    .main {{
        background-color: {BG_COLOR};
        padding: 1rem;
    }}
    
    /* Header styling */
    .header-container {{
        background-color: {PRIMARY_COLOR};
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }}
    
    .logo-text {{
        font-weight: bold;
        font-size: 1.5rem;
    }}
    
    /* Card styling */
    .card {{
        background-color: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        border-left: 4px solid {PRIMARY_COLOR};
    }}
    
    /* Upload container */
    .upload-container {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }}
    
    /* Prediction boxes */
    .prediction-box {{
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }}
    
    .positive {{
        background-color: #fff5f7;
        border-left: 4px solid {POSITIVE_COLOR};
    }}
    
    .negative {{
        background-color: #f0fdf9;
        border-left: 4px solid {NEGATIVE_COLOR};
    }}
    
    /* Confidence meter */
    .confidence-meter {{
        height: 10px;
        border-radius: 5px;
        margin-top: 10px;
        background-color: #e9ecef;
    }}
    
    .confidence-value {{
        height: 100%;
        border-radius: 5px;
    }}
    
    /* Section headers */
    .section-header {{
        color: {SECONDARY_COLOR};
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
    }}
    
    /* File uploader */
    .st-emotion-cache-1erivf3, .st-emotion-cache-1gulkj5 {{
        border: 2px dashed #d1d5db;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        background-color: #f9fafb;
    }}
    
    /* Disclaimer */
    .disclaimer {{
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.9rem;
        color: #6c757d;
        border-left: 3px solid #6c757d;
        margin-top: 1rem;
    }}
    
    /* Stats container */
    .stats-container {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }}
    
    .stat-box {{
        background-color: white;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
        flex: 1;
        margin: 0 0.3rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}
    
    .stat-value {{
        font-size: 1.3rem;
        font-weight: bold;
        color: {PRIMARY_COLOR};
    }}
    
    .stat-label {{
        color: #6c757d;
        font-size: 0.8rem;
    }}
    
    /* Remove debug info */
    .stDeployButton, footer {{
        display: none !important;
    }}
</style>
""",
    unsafe_allow_html=True,
)


# Function to load the model
@st.cache_resource
def load_covid_model(model_path):
    return load_model(model_path)


# Function to preprocess the image
def preprocess_image(img):
    # Resize the image to match VGG19 input size (224x224)
    img = img.resize((224, 224))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Check if the image is grayscale (single channel)
    if len(img_array.shape) == 3 and img_array.shape[2] == 1:
        # Repeat the single channel to create 3 channels (RGB)
        img_array = np.repeat(img_array, 3, axis=-1)

    # Expand dimensions to add batch size (1 image)
    img_array = np.expand_dims(img_array, axis=0)

    # VGG19 preprocessing
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)

    return img_array


# Header with logo
st.markdown(
    """
<div class="header-container">
    <div class="logo-text">COVID-19 X-Ray Analyzer</div>
</div>
""",
    unsafe_allow_html=True,
)

# Introduction
st.markdown(
    """
<div class="card">
    <div class="section-header">Welcome</div>
    <p>Upload a chest X-ray image to analyze for potential COVID-19 markers.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Simple stats
st.markdown(
    """
<div class="stats-container">
    <div class="stat-box">
        <div class="stat-value">94.85%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    <div class="stat-box">
        <div class="stat-value">VGG19</div>
        <div class="stat-label">Neural Network</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Model path input
model_path = st.text_input(
    "Path to model file",
    "C:\\Users\\Ash\\Documents\\Code\\ML\\Covid Prediction Using X-Ray Images\\vgg19-Covid-19-94.85.h5",
)

# File uploader
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">Uploaded X-ray</div>', unsafe_allow_html=True
        )
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Check if model path exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.info("Please ensure the model path is correct and the file exists.")
    else:
        try:
            # Load model
            with st.spinner("Analyzing X-ray..."):
                model = load_covid_model(model_path)
                processed_img = preprocess_image(img)
                prediction = model.predict(processed_img)[0][0]

            # Display results
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-header">Analysis Results</div>',
                    unsafe_allow_html=True,
                )

                # Determine prediction class
                is_covid = prediction > 0.5
                confidence = prediction if is_covid else 1 - prediction
                confidence_percentage = float(confidence) * 100

                # Display prediction
                if is_covid:
                    st.markdown(
                        f"""
                    <div class="prediction-box positive">
                        <h3 style="color: {POSITIVE_COLOR};">COVID-19 Detected</h3>
                        <p>Confidence: <b>{confidence_percentage:.2f}%</b></p>
                        <div class="confidence-meter">
                            <div class="confidence-value" style="width: {confidence_percentage}%; background-color: {POSITIVE_COLOR};"></div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                    <div class="prediction-box negative">
                        <h3 style="color: {NEGATIVE_COLOR};">No COVID-19 Detected</h3>
                        <p>Confidence: <b>{confidence_percentage:.2f}%</b></p>
                        <div class="confidence-meter">
                            <div class="confidence-value" style="width: {confidence_percentage}%; background-color: {NEGATIVE_COLOR};"></div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            with col2:
                st.error(f"Error during prediction: {str(e)}")
                st.info(
                    "This could be due to model compatibility issues or image format problems."
                )
import pandas as pd
import io

st.markdown("---")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-header">Batch X-ray Prediction</div>', unsafe_allow_html=True
)

batch_files = st.file_uploader(
    "Upload multiple images (batch)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if batch_files and os.path.exists(model_path):
    with st.spinner("Processing batch..."):
        model = load_covid_model(model_path)
        results = []

        for file in batch_files:
            try:
                img = Image.open(file)
                processed_img = preprocess_image(img)
                pred = model.predict(processed_img)[0][0]
                label = "COVID-19" if pred > 0.5 else "No COVID-19"
                confidence = pred if pred > 0.5 else 1 - pred

                results.append(
                    {
                        "Filename": file.name,
                        "Prediction": label,
                        "Confidence (%)": f"{confidence * 100:.2f}",
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "Filename": file.name,
                        "Prediction": "Error",
                        "Confidence (%)": str(e),
                    }
                )

        # Create dataframe
        df_results = pd.DataFrame(results)

        # Show in app
        st.dataframe(df_results)

        # Convert to CSV and provide download
        csv_data = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV Results",
            data=csv_data,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
st.markdown("</div>", unsafe_allow_html=True)


# Add disclaimer
st.markdown(
    """
<div class="disclaimer">
    <strong>Disclaimer:</strong> This tool is for educational and demonstration purposes only. 
    Always consult with healthcare professionals for medical advice.
</div>
""",
    unsafe_allow_html=True,
)

# Add a simple expander for model information
with st.expander("About the Model"):
    st.write(
        """
    This application uses a VGG19-based deep learning model fine-tuned on COVID-19 X-ray images.
    
    - **Model Architecture**: VGG19 (transfer learning)
    - **Reported Accuracy**: 94.85%
    - **Input**: Chest X-ray images (224×224 pixels)
    - **Output**: Binary classification (COVID-19 positive or negative)
    """
    )
