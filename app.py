import os

os.environ['XDG_CONFIG_HOME'] = '.streamlit'
os.environ['XDG_CACHE_HOME'] = '.streamlit'
os.environ['STREAMLIT_HOME'] = '.streamlit'

import streamlit as st
from transformers import pipeline
from PIL import Image


st.title("Reviews Sentiment Classifier")

@st.cache_resource
def load_text_model():
    return pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")

text_classifier = load_text_model()


text = st.text_input("Write your review here")

if text:
    output = text_classifier(text)
    st.write("Classification output:")
    st.json(output)

# Image classification
st.title("Image Classifier")

@st.cache_resource
def load_image_model():
    return pipeline(model="microsoft/resnet-50")

image_classifier = load_image_model()


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    result = image_classifier(image)
    st.write(f"I am '**{100*result[0]['score']:.04f}**'% sure that this image contains a '**{result[0]['label'].capitalize()}**'")
    st.write("Result:")
    st.write(result)

