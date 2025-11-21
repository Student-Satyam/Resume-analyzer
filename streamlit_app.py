import streamlit as st
import os
from PyPDF2 import PdfReader
from transformers import pipeline
import io

# Initialize the Hugging Face pipeline for text generation
# This should ideally be cached with st.cache_resource for Streamlit apps
@st.cache_resource
def get_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = get_generator()

def extract_text_from_pdf(file_object):
    # PdfReader expects a file-like object, so we pass it directly
    reader = PdfReader(file_object)
    text = ""
    for p in reader.pages:
        text += p.extract_text() or "\n"
    return text

def analyze_resume(text):
    prompt = f"Read the following resume text and give:\n1) A short summary (2-3 lines)\n2) Top 5 strengths\n3) Top 5 improvements\n4) List job roles that match the resume\nResume:\n{text}"
    # For Hugging Face, we typically pass the prompt directly to the generator
    # The output will be a list of dictionaries, take the 'generated_text' from the first one
    resp = generator(prompt, max_length=500, num_return_sequences=1)
    return resp[0]['generated_text']

st.set_page_config(layout="wide")
st.title("Resume Analyzer with Hugging Face")
st.write("Upload a PDF resume and get an AI-powered analysis!")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # To read file as bytes, then use io.BytesIO to make it file-like
    bytes_data = uploaded_file.getvalue()
    file_object = io.BytesIO(bytes_data)

    st.subheader("Extracted Resume Text:")
    resume_text = extract_text_from_pdf(file_object)
    st.text_area("", resume_text, height=300)

    if st.button("Analyze Resume"):
        if resume_text:
            with st.spinner("Analyzing resume... This might take a moment."):
                analysis_result = analyze_resume(resume_text)
            st.subheader("Analysis Results:")
            st.info(analysis_result)
        else:
            st.warning("Could not extract text from the PDF. Please try another file.")
else:
    st.info("Please upload a PDF file to begin the analysis.")
