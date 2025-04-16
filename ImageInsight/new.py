import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
import requests
import io
import os
import warnings

# Set environment variable to disable oneDNN optimizations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Neither `max_length` nor `max_new_tokens` has been set")
warnings.filterwarnings("ignore", category=FutureWarning, message="resume_download is deprecated")

# Initialize BLIP processor and model for captioning and VQA
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Optional: Define a fallback API or use a simple object detection model
def fallback_caption(image_base64):
    # Fallback URL (example, needs your actual API endpoint if used)
    url = "https://fallback-caption-api.com/detect"
    headers = {"Authorization": f"Bearer YOUR_API_KEY"}
    response = requests.post(url, headers=headers, json={"image": image_base64})
    return response.json().get("caption", "No objects detected in fallback.")

# Streamlit app setup
st.set_page_config(page_title="Enhanced Image Caption & QA", layout="wide")
st.title("Enhanced Image Caption & Q&A System")

# Image processing and conversion functions
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB").resize((512, 512))  # Standardize format and size
    return image


def encode_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_caption_with_blip(image):
    inputs = processor(image, return_tensors="pt").to(model.device)
    # Set max_new_tokens to control the response length (e.g., 20 tokens)
    outputs = model.generate(**inputs, max_new_tokens=20)
    return processor.decode(outputs[0], skip_special_tokens=True)

def get_answer_with_blip(image, question):
    # Add question to the processor input
    inputs = processor(image, question=question, return_tensors="pt").to(model.device)
    # Generate answer
    outputs = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(outputs[0], skip_special_tokens=True)

# User interface for file upload
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to base64
        image_base64 = encode_image_to_base64(image)

        # Generate caption with BLIP model
        try:
            with st.spinner("Generating caption..."):
                caption = get_caption_with_blip(image)
        except Exception as e:
            st.error(f"BLIP model error: {e}")
            # Attempt fallback caption if BLIP fails
            caption = fallback_caption(image_base64)

        # Display the generated caption
        if caption:
            st.markdown("### Caption:")
            st.write(caption)

# Q&A section for Visual Question Answering (VQA)
with col2:
    st.markdown("### Ask about the image")

    if uploaded_file is not None:
        question = st.text_input("Your question:")
        if st.button("Ask"):
            if question:
                st.markdown(f"**Q:** {question}")
                try:
                    with st.spinner("Getting answer..."):
                        answer = get_answer_with_blip(image, question)
                    st.markdown(f"**A:** {answer}")
                except Exception as e:
                    st.error(f"Error in generating answer: {e}")
    else:
        st.info("Please upload an image to ask questions.")

st.markdown("---")
