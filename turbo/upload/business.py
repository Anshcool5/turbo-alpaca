from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader

def load_pdf(pdf_file):
    """
    Extracts text from a PDF file.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n\n"  # Add spacing between pages
    return text.strip() if text else "No readable text found in the PDF."

def run_business_analysis(file_path):
    """
    Reads a resume PDF, formats a prompt, and invokes the LLM (using the chatty model)
    to generate a business idea aligned with the candidate's skills.
    """
    # Extract text from PDF
    with open(file_path, 'rb') as file:
        resume_text = load_pdf(file)

    if not resume_text or resume_text == "No readable text found in the PDF.":
        return "🚨 Error: Could not extract readable text from the PDF."

    # Initialize the ChatGroq LLM with the chatty model
    chat = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

    # Define the prompt template for business idea generation
    prompt = ChatPromptTemplate.from_template("""
    You are an AI specializing in business idea generation.
    Based on the resume below, suggest a structured business idea that aligns with the candidate's skills.

    Resume Content:
    {resume_text}

    Provide the response in a **strict JSON format** with the following structure:
    {{
        "name": "A concise and compelling name for the business idea",
        "explanation": "A detailed explanation of what the business is, how it works, and why it aligns with the candidate's skills",
        "how_to_set_up": [
            {{
                "step": 1,
                "title": "Step Title",
                "description": "Detailed description of this step in setting up the business"
            }},
            {{
                "step": 2,
                "title": "Step Title",
                "description": "Detailed description of this step in setting up the business"
            }},
            ...
        ]
    }}

    Ensure the output is **valid JSON format** with no extra text, explanations, or commentary outside of the JSON object, and do not add quotes, because it messes up with the JSON format.
    """)

    # Format the prompt with the extracted resume text
    formatted_prompt = prompt.format(resume_text=resume_text)
    # Invoke the LLM to generate a response
    ai_response = chat.invoke(formatted_prompt)
    return ai_response.content
