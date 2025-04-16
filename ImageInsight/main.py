import streamlit as st
from PIL import Image
import io
import base64
import os
import requests  # Import requests to make API calls

# Set your Gemini API key here
GEMINI_API_KEY = 'AIzaSyCeyj7m254qHqs6ZTtsleag_pObMzDmEBU'  # Replace with your actual Gemini API key

st.set_page_config(page_title="Image Caption & QA", layout="wide")


def encode_image_to_base64(image_file):
    if image_file is not None:
        bytes_data = image_file.getvalue()
        base64_string = base64.b64encode(bytes_data).decode('utf-8')
        return base64_string
    return None


def get_image_caption(image_base64):
    url = "https://api.gemini.com/v1/image/caption"  # Hypothetical URL
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "image": image_base64,
        "prompt": "Provide a single sentence, specific caption for this image. Focus only on the main elements and action. Keep it under 15 words."
    }

    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    return response_data.get('caption', 'No caption generated')


def get_image_qa(image_base64, question):
    url = "https://api.gemini.com/v1/image/qa"  # Hypothetical URL
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "image": image_base64,
        "question": question
    }

    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    return response_data.get('answer', 'No answer generated')


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'image_caption' not in st.session_state:
    st.session_state.image_caption = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

st.title("Image Caption & QA")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        image_base64 = encode_image_to_base64(uploaded_file)

        if st.session_state.current_image != image_base64:
            st.session_state.current_image = image_base64
            with st.spinner('Generating caption...'):
                caption = get_image_caption(image_base64)
                st.session_state.image_caption = caption
                st.session_state.chat_history = []

        if st.session_state.image_caption:
            st.markdown("### Caption:")
            st.write(st.session_state.image_caption)

with col2:
    st.markdown("### Ask about the image")

    for message in st.session_state.chat_history:
        role = "Q" if message.startswith("Q:") else "A"
        st.markdown(f"**{role}:** {message}")

    if uploaded_file is not None:
        question = st.text_input("Your question:")
        if st.button("Ask"):
            if question:
                user_message = f"Q: {question}"  # Simply format the user message
                st.session_state.chat_history.append(user_message)

                with st.spinner('Getting answer...'):
                    answer = get_image_qa(image_base64, question)
                    ai_message = f"A: {answer}"  # Format the AI message
                    st.session_state.chat_history.append(ai_message)

    else:
        st.info("Upload an image first")

st.markdown("---")