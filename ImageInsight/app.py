# app.py
import streamlit as st
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForConditionalGeneration
import base64
import io

# Set up page layout
st.set_page_config(page_title="ImageInsight: Caption & Q&A", layout="wide")
st.title("ImageInsight: Enhanced Image Captioning & Visual Question Answering")

# Initialize models and processors
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Image processing and conversion functions
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    return image

def encode_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Caption generation with BLIP model
def get_caption_with_blip(image):
    inputs = blip_processor(image, return_tensors="pt").to(blip_model.device)
    outputs = blip_model.generate(**inputs, max_new_tokens=20)
    return blip_processor.decode(outputs[0], skip_special_tokens=True)

# Visual Question Answering with ViLT model
def get_answer(image, text):
    try:
        encoding = vilt_processor(image, text, return_tensors="pt")
        outputs = vilt_model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = vilt_model.config.id2label[idx]
        return answer
    except Exception as e:
        return str(e)

# Set up Streamlit interface
col1, col2 = st.columns([1, 1])

# Image upload and caption generation
with col1:
    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate caption with BLIP model
        try:
            with st.spinner("Generating caption..."):
                caption = get_caption_with_blip(image)
                st.markdown("### Caption:")
                st.write(caption)
        except Exception as e:
            st.error(f"Error generating caption: {e}")

# Q&A Section
with col2:
    st.markdown("### Ask a question about the image")

    if uploaded_file is not None:
        question = st.text_input("Enter your question:")
        if st.button("Ask Question"):
            if question:
                st.markdown(f"**Q:** {question}")
                answer = get_answer(image, question)
                st.markdown(f"**A:** {answer}")
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please upload an image to ask questions.")

st.markdown("---")
