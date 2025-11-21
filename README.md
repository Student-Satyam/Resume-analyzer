# Resume Analyzer with Hugging Face

This is a Streamlit application that allows users to upload a PDF resume, extract its text, and then use a Hugging Face pre-trained model (FLAN-T5-base) to analyze the resume. The analysis provides a short summary, top strengths, top improvements, and suggested job roles.

## Features

- PDF resume upload functionality.
- Text extraction from uploaded PDF files.
- AI-powered resume analysis using Hugging Face's `text2text-generation` pipeline.
- Displays extracted text and detailed analysis results.

## Setup and Installation

To run this application, you need to have Python installed. Follow these steps to set up the project:

1.  **Clone the repository (if applicable) or save the application code.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file with the following content:
    ```
    streamlit
    PyPDF2
    transformers
    torch # Or tensorflow, depending on your transformers backend. PyTorch is common.
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `transformers` library often requires either `torch` or `tensorflow`. `torch` is generally recommended for Hugging Face models.*

4.  **Save the application code:**
    Save the provided Python code into a file named `app.py` in your project directory.

## How to Run

Navigate to the directory where you saved `app.py` in your terminal or command prompt, and run the Streamlit application using the following command:

```bash
streamlit run app.py
This will open the application in your web browser, typically at http://localhost:8501.

Usage
Upload a PDF: Click the "Choose a PDF file" button and select your resume in PDF format.
View Extracted Text: The application will display the text extracted from your PDF.
Analyze Resume: Click the "Analyze Resume" button to get the AI-generated analysis.
Review Results: The analysis results, including summary, strengths, improvements, and job roles, will be displayed.
Example
Once the app is running:

Upload a sample resume.pdf file.
Click 'Analyze Resume'.
View the AI-generated insights.
Enjoy analyzing your resumes! ```
