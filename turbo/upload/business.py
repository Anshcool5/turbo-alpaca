from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
import json

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

    Ensure the output is **valid JSON format** with no extra text, explanations, or commentary outside of the JSON object.
    """)

    # Format the prompt with the extracted resume text
    formatted_prompt = prompt.format(resume_text=resume_text)
    # Invoke the LLM to generate a response
    ai_response = chat.invoke(formatted_prompt)
    
    # Clean up the response to ensure it's valid JSON
    try:
        # Remove any leading/trailing non-JSON text
        json_start = ai_response.content.find('{')
        json_end = ai_response.content.rfind('}') + 1
        json_str = ai_response.content[json_start:json_end]
        
        # Parse the JSON to ensure it's valid
        json_data = json.loads(json_str)
        return json.dumps(json_data, indent=4)
    except json.JSONDecodeError as e:
        return f"🚨 Error: Failed to parse the LLM response as JSON. {e}"

# Example usage
# result = run_business_analysis("path_to_resume.pdf")
# print(result)