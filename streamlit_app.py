import streamlit as st
import os
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer
import io
import json
import re
import base64

# --- Configuration --- #
MODEL_NAME = "google/flan-t5-large" # Upgraded model
MAX_INPUT_TOKENS = 450 # Adjusted for typical T5 context window (512 - prompt length)

# --- Streamlit Setup --- #
st.set_page_config(layout="wide", page_title="Advanced Resume Analyzer")

# --- Caching for efficiency --- #
@st.cache_resource(show_spinner="Loading AI model...")
def get_generator_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    generator = pipeline("text2text-generation", model=MODEL_NAME, tokenizer=tokenizer)
    return generator, tokenizer

generator, tokenizer = get_generator_and_tokenizer()

# --- Helper Functions --- #
def extract_text_from_pdf(file_object):
    reader = PdfReader(file_object)
    text = ""
    for p in reader.pages:
        page_text = p.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def analyze_resume(text):
    # Truncate input text to fit model's max input length
    encoded_input = tokenizer.encode(text, max_length=MAX_INPUT_TOKENS, truncation=True)
    truncated_text = tokenizer.decode(encoded_input, skip_special_tokens=True)

    prompt = f"""
    You are an AI assistant specialized in resume analysis. Your task is to analyze the provided resume text
    and output a single JSON object. The JSON object must contain the following keys:
    'summary': A concise summary (2-3 sentences) of the resume. Describe the candidate's core skills and experience.
    'strengths': A list of top 5 key strengths identified in the resume. Focus on quantifiable achievements and relevant skills.
    'improvements': A list of top 5 constructive suggestions or areas for improvement for the resume. Focus on clarity, completeness, and impact.
    'job_roles': A list of 3-5 suitable job titles or roles that match the resume's qualifications, ordered by relevance.

    Resume Text to Analyze:
    ```
    {truncated_text}
    ```

    Your entire output must be a single, valid JSON object, enclosed within a markdown code block like this:
    ```json
    {{
        "summary": "...",
        "strengths": [...],
        "improvements": [...],
        "job_roles": [...]
    }}
    ```
    Do not include any narrative, explanation, or additional text before or after the JSON code block.
    Begin your response directly with the opening ```json.
    """

    # Use max_new_tokens for generation length
    resp = generator(prompt, max_new_tokens=1024, num_return_sequences=1, do_sample=True, temperature=0.7, top_p=0.9)

    raw_output = resp[0]['generated_text'].strip()

    # Robust JSON extraction and parsing
    analysis_result = {}

    # 1. Attempt to extract JSON from markdown code block (preferred)
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_output)
    if json_match:
        parsed_json_text = json_match.group(1).strip()
    else:
        # 2. Fallback: try to clean and fix common model output issues if no markdown block
        cleaned_output = re.sub(r'^(json|JSON)\s*', '', raw_output, flags=re.IGNORECASE).strip()
        # If the output starts with a quote or a key (indicating a missing outer brace) and not already wrapped
        if cleaned_output and not (cleaned_output.startswith('{') and cleaned_output.endswith('}')):
             if re.match(r'"\w+":', cleaned_output) or re.match(r'\w+:', cleaned_output):
                 parsed_json_text = '{' + cleaned_output + '}'
             else:
                 parsed_json_text = cleaned_output # Use as is if it doesn't look like an object content
        else:
            parsed_json_text = cleaned_output

    try:
        analysis_result = json.loads(parsed_json_text)
    except json.JSONDecodeError as e:
        st.error(f"AI output was not valid JSON. Error: {e}")
        st.code(f"Raw AI Output:\n{raw_output}\n\nAttempted to parse:\n{parsed_json_text}")
        analysis_result = {"error": "Failed to parse AI output as JSON. See logs for details.", "raw_output": raw_output}

    return analysis_result

def get_download_link(data, filename, text):
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">{text}</a>'
    return href

# --- Streamlit UI --- #
st.title("🧠 Advanced Resume Analyzer")
st.markdown("Upload a PDF resume and get an AI-powered analysis with strengths, improvements, and job role suggestions.")

with st.sidebar:
    st.header("Upload Your Resume")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], help="Please upload a clear, text-based PDF resume.")
    process_button = st.button("Analyze Resume", type="primary", use_container_width=True)


if uploaded_file is not None:
    # Read file as bytes and use io.BytesIO to make it file-like for PyPDF2
    bytes_data = uploaded_file.getvalue()
    file_object = io.BytesIO(bytes_data)

    st.subheader("1. Extracted Resume Text")
    with st.expander("View Raw Text", expanded=False):
        resume_text = extract_text_from_pdf(file_object)
        if resume_text:
            st.text_area("", resume_text, height=300, disabled=True, label_visibility="collapsed")
        else:
            st.warning("Could not extract text from the PDF. It might be an image-based PDF. Please try another file.")

    if process_button and resume_text:
        st.subheader("2. AI Analysis Results")
        with st.spinner("Analyzing resume... This might take a moment (approx. 30-60 seconds)."):
            analysis_result = analyze_resume(resume_text)

        if "error" in analysis_result:
            st.error("An error occurred during analysis. Please check the console or try again.")
        else:
            # Display structured results
            with st.expander("Summary", expanded=True):
                st.info(analysis_result.get("summary", "No summary provided."))

            with st.expander("Top Strengths", expanded=True):
                if analysis_result.get("strengths"):
                    for i, strength in enumerate(analysis_result["strengths"]):
                        st.markdown(f"- **{strength}**")
                else:
                    st.markdown("No strengths identified.")

            with st.expander("Areas for Improvement", expanded=True):
                if analysis_result.get("improvements"):
                    for i, improvement in enumerate(analysis_result["improvements"]):
                        st.warning(f"- {improvement}")
                else:
                    st.markdown("No improvements suggested.")

            with st.expander("Matching Job Roles", expanded=True):
                if analysis_result.get("job_roles"):
                    for i, role in enumerate(analysis_result["job_roles"]):
                        st.success(f"- {role}")
                else:
                    st.markdown("No job roles suggested.")

            st.markdown("--- Jardar")
            st.download_button(
                label="Download Analysis as JSON",
                data=json.dumps(analysis_result, indent=2),
                file_name="resume_analysis.json",
                mime="application/json",
                help="Download the complete AI analysis in JSON format."
            )

    elif process_button and not resume_text:
        st.warning("Please upload a PDF file and ensure text can be extracted before analyzing.")

elif uploaded_file is None and process_button:
    st.warning("Please upload a PDF file to begin the analysis.")
else:
    st.info("Upload a PDF resume in the sidebar to get started!")
