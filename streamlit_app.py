import streamlit as st
import os
from PyPDF2 import PdfReader
from transformers import pipeline
import io
import json # Import json for parsing output
import re   # Import regex for extracting JSON block

# Initialize the Hugging Face pipeline for text generation
# This should ideally be cached with st.cache_resource for Streamlit apps
@st.cache_resource
def get_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = get_generator()

def extract_text_from_pdf(file_object):
    reader = PdfReader(file_object)
    text = ""
    for p in reader.pages:
        text += p.extract_text() or "\n"
    return text

def analyze_resume(text):
    # 1. Truncate input text to fit model's max input length (typically 512 for FLAN-T5-base)
    # The tokenizer is part of the pipeline's underlying model/tokenizer.
    # We use 450 tokens to leave some buffer for the prompt structure itself.
    max_input_tokens = 450
    encoded_input = generator.tokenizer.encode(text, max_length=max_input_tokens, truncation=True)
    truncated_text = generator.tokenizer.decode(encoded_input, skip_special_tokens=True)

    # 2. Modify prompt to ask for structured JSON output and be very strict about it
    # Instruct the model to wrap its JSON in a markdown code block.
    prompt = f"""
    You are an AI assistant specialized in resume analysis. Your task is to analyze the provided resume text
    and output a single JSON object. The JSON object must contain the following keys:
    'summary': A concise summary (2-3 sentences) of the resume.
    'strengths': A list of top 5 key strengths identified in the resume.
    'improvements': A list of top 5 constructive suggestions or areas for improvement for the resume.
    'job_roles': A list of suitable job titles or roles that match the resume's qualifications.

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

    # 3. Use max_new_tokens instead of max_length
    resp = generator(prompt, max_new_tokens=768, num_return_sequences=1)

    raw_output = resp[0]['generated_text'].strip()

    # 4. Extract the JSON block using regex and parse it
    parsed_json_text = ""
    analysis_result = {}

    # Attempt to extract JSON from markdown code block (preferred)
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_output)
    if json_match:
        parsed_json_text = json_match.group(1).strip()
    else:
        # Fallback if no markdown block found: try to clean and fix common model output issues
        # Remove leading 'json ' if present, case-insensitive
        cleaned_output = re.sub(r'^(json|JSON)\s*', '', raw_output, flags=re.IGNORECASE).strip()

        # If the output starts with a quote or a key (indicating a missing outer brace)
        # AND it doesn't already start/end with curly braces, try to wrap it.
        # This handles cases like: "summary": "...", "strengths": [...]
        if cleaned_output and not (cleaned_output.startswith('{') and cleaned_output.endswith('}')):
             # Check for common patterns that should be inside an object
             if re.match(r'"\w+":', cleaned_output) or re.match(r'\w+:', cleaned_output):
                 parsed_json_text = '{' + cleaned_output + '}'
             else:
                 parsed_json_text = cleaned_output # Use as is if it doesn't look like an object content
        else:
            parsed_json_text = cleaned_output # Use the cleaned output as is


    try:
        analysis_result = json.loads(parsed_json_text)
    except json.JSONDecodeError as e:
        # Fallback if JSON parsing fails - return raw text or a structured error
        st.warning(f"Failed to parse AI analysis as JSON. Error: {e}. Raw output below:")
        analysis_result = {
            "error": "Failed to parse AI output as JSON.",
            "raw_text": raw_output, # Keep original raw output for debugging
            "json_error": str(e),
            "attempted_json_parse_text": parsed_json_text # Show what was actually attempted to parse
        }

    return analysis_result

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
    st.text_area("Resume Text", resume_text, height=300, label_visibility="hidden")

    if st.button("Analyze Resume"):
        if resume_text:
            with st.spinner("Analyzing resume... This might take a moment."):
                analysis_result = analyze_resume(resume_text)
            st.subheader("Analysis Results:")
            # Display structured results
            if isinstance(analysis_result, dict):
                if "error" in analysis_result:
                    st.error(analysis_result["error"])
                    st.code(analysis_result["raw_text"])
                    if "json_error" in analysis_result:
                        st.error(f"JSON Parsing Error Details: {analysis_result['json_error']}")
                    if "attempted_json_parse_text" in analysis_result:
                        st.code(f"Attempted to parse:\n{analysis_result['attempted_json_parse_text']}")
                else:
                    st.write("**Summary:**")
                    st.info(analysis_result.get("summary", "N/A"))
                    st.write("**Top 5 Strengths:**")
                    for strength in analysis_result.get("strengths", []):
                        st.success(f"- {strength}")
                    st.write("**Top 5 Improvements:**")
                    for improvement in analysis_result.get("improvements", []):
                        st.warning(f"- {improvement}")
                    st.write("**Matching Job Roles:**")
                    for role in analysis_result.get("job_roles", []):
                        st.info(f"- {role}")
            else:
                st.info(analysis_result)
        else:
            st.warning("Could not extract text from the PDF. Please try another file.")
else:
    st.info("Please upload a PDF file to begin the analysis.")
