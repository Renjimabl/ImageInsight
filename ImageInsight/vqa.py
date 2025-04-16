import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import base64
import requests
import io

# Initialize BLIP processor and model for captioning and VQA
try:
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-6.7b")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip2-opt-6.7b")
except Exception as e:
    st.error(f"Error loading models: {e}")
    raise e


# Optional: Define a fallback API or use a simple object detection model
def fallback_caption(image_base64):
    # Fallback URL (example, needs your actual API endpoint if used)
    url = "https://fallback-caption-api.com/detect"
    headers = {"Authorization": f"Bearer YOUR_API_KEY"}
    response = requests.post(url, headers=headers, json={"image": image_base64})
    return response.json().get("caption", "No objects detected in fallback.")


# Streamlit app setup
st.set_page_config(page_title="Enhanced Image Caption & QA", layout="wide")
st.title("Enhanced Image Caption AND Q&A System")


# Image processing and conversion functions
def preprocess_image(uploaded_image):
    try:
        image = Image.open(uploaded_image).convert("RGB").resize((512, 512))  # Standardize format and size
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def encode_image_to_base64(image):
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None


def get_caption_with_blip(image):
    try:
        inputs = caption_processor(image, return_tensors="pt").to(caption_model.device)
        outputs = caption_model.generate(**inputs, max_new_tokens=20)
        return caption_processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return None


def get_answer_with_blip2(image, question):
    try:
        inputs = vqa_processor(image, question=question, return_tensors="pt").to(vqa_model.device)
        answer = vqa_model.generate(**inputs)
        return vqa_processor.decode(answer[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return None


# User interface for file upload
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Process the uploaded image
        image = preprocess_image(uploaded_file)
        if image is not None:
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Convert image to base64
            image_base64 = encode_image_to_base64(image)

            # Generate caption with BLIP model
            try:
                with st.spinner("Generating caption..."):
                    caption = get_caption_with_blip(image)
                    if caption:
                        st.markdown("### Caption:")
                        st.write(caption)
                    else:
                        st.error("Failed to generate a caption.")
            except Exception as e:
                st.error(f"BLIP model error: {e}")
                # Attempt fallback caption if BLIP fails
                if image_base64:
                    caption = fallback_caption(image_base64)
                    st.markdown("### Caption:")
                    st.write(caption)

# Q&A section (placeholder for possible extension)
with col2:
    st.markdown("### Ask about the image")

    if uploaded_file is not None:
        question = st.text_input("Your question:")
        if st.button("Ask"):
            if question:
                st.markdown(f"**Q:** {question}")
                # Generate answer using BLIP2 VQA model
                try:
                    answer = get_answer_with_blip2(image, question)
                    if answer:
                        st.markdown(f"**A:** {answer}")
                    else:
                        st.error("Failed to generate an answer.")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
    else:
        st.info("Please upload an image to ask questions.")

st.markdown("---")
