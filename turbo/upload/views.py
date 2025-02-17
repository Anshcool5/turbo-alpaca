from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from dotenv import load_dotenv
from langchain.schema import Document
# from langchain.llms import Ollama  # Use LangChain's Ollama LLM integration
from transformers import pipeline
import json
import ijson
import sys
from .file_handling import create_file_record
from django.core.files.storage import default_storage
import pandas as pd
from django.conf import settings
from plotly.offline import plot
from django.db import connection

import datetime
from django.http import JsonResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .chatty import run_llm
from .business_idea_analysis import run_idea

from .business import run_business_analysis

from .parser import parse_business_idea

# Replace Ollama with Hugging Face LLM
#llm = pipeline("text-generation", model="gpt2")

# Load environment variables from .env file
load_dotenv()

# Initialize the Ollama LLM (Llama 3.2 model)
# llm = Ollama(model="llama3")

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"  # Example: "us-west1-gcp-free"

# Create an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create or connect to a Pinecone index
index_name = "document-embeddings"
index_name = "resumes"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  # Use "gcp" or "azure" if needed
            region="us-east-1"  # Use your preferred region
        )
    )

# Connect to the Pinecone index
index = pc.Index(index_name)

# Home view
def home(request):
    return render(request, "upload/home.html")

# Function to extract text from a PDF file
def load_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to split text into Document objects with metadata
def split_text_into_documents(text, source):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    # Convert chunks into Document objects with metadata
    documents = [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks]
    return documents

# Function to embed text chunks
def embed_text_chunks(documents):
    # Use the correct method 'embed_documents' to generate embeddings
    embeddings = embedding_model.embed_documents([doc.page_content for doc in documents])
    return embeddings


def process_large_json(file):
    text = ""
    parser = ijson.items(file, "item")  # Adjust the JSON path based on your structure
    for item in parser:
        text += json.dumps(item) + "\n"
        if len(text) > 10000:  # Process in chunks of 10,000 characters
            yield text
            text = ""  # Reset text buffer
    if text:
        yield text


def upload_file(request):
    if request.method == "POST" and request.FILES.get("uploaded_file"):
        uploaded_file = request.FILES["uploaded_file"]
        file_name = uploaded_file.name.lower()

        # Initialize variables
        documents = []

        try:
            # Create the file record and save the file
            file_record = create_file_record(request.user, uploaded_file, file_name)
            
            # Determine the path to the saved file
            file_path = os.path.join(default_storage.location, 'uploads', file_name)
            #print("file_path", file_path)
            # Extract text based on file type
            if file_name.endswith(".pdf"):
                # Process PDF file by reading from the file path
                with open(file_path, 'rb') as file:
                    text = load_pdf(file)
                # Split text into chunks and convert to Document objects
                documents = split_text_into_documents(text, source=file_name)

            elif file_name.endswith(".json"):
                # Process JSON file in chunks using process_large_json
                with open(file_path, 'r') as file:
                    for chunk in process_large_json(file):
                        documents.extend(split_text_into_documents(chunk, source=file_name))

            else:
                messages.error(request, "Unsupported file format")
                return render(request, "upload/home.html")

            #print(documents[:10])  # Print the first few documents for debugging
            # Validate metadata size before uploading
            for doc in documents:
                metadata_size = sys.getsizeof(doc.metadata)
                if metadata_size > 40960:  # 40 KB limit
                    messages.error(request, f"Metadata size exceeds limit: {metadata_size} bytes")
                    return render(request, "upload/home.html")

            # Generate embeddings and store in Pinecone using LangChain's PineconeVectorStore
            try:
                vectorstore = PineconeVectorStore.from_documents(
                    documents=documents,
                    embedding=embedding_model,
                    index_name=index_name
                )
                messages.success(request, "File uploaded successfully!")
            except Exception as e:
                messages.error(request, f"Failed to upload to Pinecone: {str(e)}")
                return redirect("home")  # Redirect to home page if error occurs in Pinecone upload


            # After successful upload, refresh the recent files list
            recent_files = update_file_list(request)
            return render(request, "upload/home.html", {"recent_files": recent_files})
        
        except Exception as e:
            messages.error(request, f"An error occurred: {str(e)}")
            return redirect("home")  # Redirect to home page if any error occurs

        return render(request, "upload/home.html")

    return render(request, "upload/home.html")



# def generate_idea(request):
#     index_name = "resumes"
#     if request.method == "POST" and request.FILES.get("uploaded_file"):
#         uploaded_file = request.FILES["uploaded_file"]
#         file_name = uploaded_file.name.lower()

#         # Initialize variables
#         documents = []

#         try:
#             # Create the file record and save the file
#             file_record = create_file_record(request.user, uploaded_file, file_name)
            
#             # Determine the path to the saved file
#             file_path = os.path.join(default_storage.location, 'uploads', file_name)
#             #print("file_path", file_path)
#             # Extract text based on file type
#             if file_name.endswith(".pdf"):
#                 # Process PDF file by reading from the file path
#                 with open(file_path, 'rb') as file:
#                     text = load_pdf(file)
#                 # Split text into chunks and convert to Document objects
#                 documents = split_text_into_documents(text, source=file_name)
#             else:
#                 messages.error(request, "Unsupported file format")
#                 return render(request, "upload/resume.html")

#             #print(documents[:10])  # Print the first few documents for debugging
#             # Validate metadata size before uploading
#             for doc in documents:
#                 metadata_size = sys.getsizeof(doc.metadata)
#                 if metadata_size > 40960:  # 40 KB limit
#                     messages.error(request, f"Metadata size exceeds limit: {metadata_size} bytes")
#                     return render(request, "upload/resume.html")

#             # Generate embeddings and store in Pinecone using LangChain's PineconeVectorStore
#             try:
#                 vectorstore = PineconeVectorStore.from_documents(
#                     documents=documents,
#                     embedding=embedding_model,
#                     index_name=index_name
#                 )
#                 messages.success(request, "File uploaded successfully!")
#                 return redirect("generate")
#             except Exception as e:
#                 messages.error(request, f"Failed to upload to Pinecone: {str(e)}")
#                 return redirect("generate")  # Redirect to home page if error occurs in Pinecone upload


#             # After successful upload, refresh the recent files list
#             recent_files = update_file_list(request)
#             return render()
        
#         except Exception as e:
#             messages.error(request, f"An error occurred: {str(e)}")
#             return redirect("generate")  # Redirect to home page if any error occurs

#         return render(request, "upload/home.html")

#     return render(request, "upload/resume.html")


def generate_idea(request):
    if request.method == "POST" and request.FILES.get("uploaded_file"):
        uploaded_file = request.FILES["uploaded_file"]
        file_name = uploaded_file.name.lower()
        file_path = os.path.join(default_storage.location, "uploads", file_name)

        try:
            # Save the file
            with default_storage.open(file_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Run AI business analysis
            ai_response = run_business_analysis(file_path)
            #print(ai_response, "working?")
            # Parse JSON response
            # print(ai_response)
            # print(type(ai_response))
            obj = parse_business_idea(ai_response)
            print("obj", obj)
            #string
            name = obj["name"]
            #string
            explanation = obj["explanation"]
            #list
            steps = obj["how_to_set_up"]
            print(steps)
            
            return render(request, "upload/resume.html", {"name": name, "explanation": explanation, "steps": steps})

        except Exception as e:
            return JsonResponse({"error": str(e)})

    return render(request, "upload/resume.html")


def update_file_list(request):
    try:
        # Get logged-in username
        username = request.user.username

        # Debugging: Print the username
        #print(f"Logged-in username: {username}")

        # Get user_id from auth_user table
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM auth_user WHERE username = %s", [username])
            user_row = cursor.fetchone()

        if not user_row:
            return []  # If user is not found, return an empty list

        user_id = user_row[0]  # Extract user ID

        # Query the last 5 most recently uploaded files (descending order by uploaded_at)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT file_name, uploaded_at
                FROM upload_file
                WHERE user_id = %s
                ORDER BY uploaded_at DESC
                LIMIT 5
                """, 
                [user_id]
            )
            files = cursor.fetchall()

        # Format the result into a list of dictionaries
        recent_files = [{"file_name": file[0], "uploaded_at": file[1]} for file in files]
        #print('recent files',recent_files)
        return recent_files

    except Exception as e:
        return {"error": str(e)}  # Return error in case of failure

# Home view, now with the call to update_file_list
def home(request):
    recent_files = update_file_list(request)  # Get the 5 most recently uploaded files

    # Debugging: Check if recent_files has been fetched
    #print(f"Recent Files: {recent_files}")

    if isinstance(recent_files, dict):  # In case of error, pass it to the template
        messages.error(request, f"Error fetching recent files: {recent_files.get('error')}")
        recent_files = []

    return render(request, "upload/home.html", {"recent_files": recent_files})



# User registration view
def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Log the user in after registration
            return redirect("home")  # Redirect to the home page
    else:
        form = UserCreationForm()
    return render(request, "upload/registration.html", {"form": form})

# User login view
def user_login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("home")  # Redirect to the home page
        else:
            # Return an error message
            return render(request, 'upload/login.html', {'error': 'Invalid username or password'})
    return render(request, 'upload/login.html')

# User logout view
def user_logout(request):
    logout(request)
    return redirect("login")  # Redirect to the login page

from django.core.mail import send_mail
from django.http import HttpResponse
import os

def test_email(request):
    try:
        send_mail(
            'Test Email',
            'This is a test email.',
            os.getenv("EMAIL_HOST_USER"),  # FROM address
            ['melritacyriac123@gmail.com'],  # Replace with a valid recipient email address
            fail_silently=False,
        )
        return HttpResponse("Test email sent.")
    except Exception as e:
        return HttpResponse(f"Error: {e}")



def query_documents(request):
    if request.method == "POST":
        query_text = request.POST.get("query")
        if not query_text:
            return render(request, "upload/query_documents.html", {"error": "Query text is required"})

        # Get logged-in username
        username = request.user.username

        # Get user_id from auth_user table
        with connection.cursor() as cursor:
            cursor.execute("SELECT id FROM auth_user WHERE username = %s", [username])
            user_row = cursor.fetchone()

        if not user_row:
            return render(request, "upload/query_documents.html", {"error": "User not found in database"})

        user_id = user_row[0]  # Extract user ID

        # Get filenames from upload_file table, excluding .json files
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT file_name FROM upload_file WHERE user_id = %s AND LOWER(file_name) NOT LIKE %s AND LOWER(file_name) LIKE %s", 
                [user_id, '%json', f'%{query_text}%']
            )
            files = cursor.fetchall()

        file_names = [file[0] for file in files]  # Extract filenames

        if not file_names:
            return render(request, "upload/query_documents.html", {"error": "No files found for this user"})

        #print(file_names)

        # Convert query to an embedding

        # Render the results on the query_documents page
        return render(request, "upload/query_documents.html", {"results": file_names})

    return render(request, "upload/query_documents.html")



@csrf_exempt
def chatbot_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "")
            
            #print("user_message", user_message)
            if user_message:
                response = run_llm(user_message, request.user)
                return JsonResponse({"response": response})
            else:
                return JsonResponse({"response": "Please enter a valid question."})
        except Exception as e:
            return JsonResponse({"response": f"An error occurred: {str(e)}"})
    
    return JsonResponse({"response": "Invalid request method."})


# from .data_analysis_func import (
#     calculate_total_revenue_data, plot_total_revenue,
#     calculate_profit_margin_data, plot_profit_margin,
#     calculate_number_of_transactions_data, plot_number_of_transactions,
#     calculate_peak_sales_period_data, plot_peak_sales_period,
#     calculate_seasonal_fluctuations_data, plot_seasonal_fluctuations,
#     calculate_customer_churn_data, plot_customer_churn,
#     get_best_sellers_data, plot_best_sellers,
#     get_worst_sellers_data, plot_worst_sellers,
#     get_stock_levels_data, plot_stock_levels,
#     forecast_stock_data, plot_forecast_stock,
#     suggest_stock_ordering_data, plot_stock_ordering,
#     calculate_stock_valuation_data, plot_stock_valuation,
#     check_stock_expiry_data, plot_stock_expiry,
#     calculate_stock_spoilage_data, plot_stock_spoilage,
#     forecast_sales_prophet_data, plot_sales_prophet,
#     perform_customer_segmentation_data, plot_customer_segmentation,
#     correlation_heatmap_data, plot_correlation_heatmap
# )

def evaluate(request):
    return render(request, "upload/evaluate.html")

def process_idea(request):

    if request.method == 'POST':
        # Retrieve form data
        print('REQYESTTTT',request)
        breakpoint
        idea_name = request.POST.get('idea_name')
        idea_text = request.POST.get('idea_text')
        industry = request.POST.get('industry')

        # Call your Python function with the form data // returns metric values
        metrics  = run_idea(idea_name, idea_text, industry)
        # Pass the metrics to the template
        return render(request, 'upload/llamafinal.html', {'metrics': metrics})
    

def dashboard(request):
    plot_dir = os.path.join(settings.MEDIA_ROOT, "plots")
    plot_urls = []
    try:
        for file in os.listdir(plot_dir):
            if file.endswith(".html"):
                # Build the URL from MEDIA_URL (ensure MEDIA_URL is set correctly in settings.py)
                file_url = os.path.join(settings.MEDIA_URL, "plots", file)
                plot_urls.append(file_url)
    except Exception as e:
        print("Error loading plot files:", e)
    
    context = {
        "plot_urls": plot_urls,
    }
    return render(request, "upload/dashboard.html", context)


# def dashboard(request):
#     # Load your data (adjust the path as needed)
#     file_name = "file_data.json"
#     file_path = os.path.join(default_storage.location, 'uploads', file_name)
#     data = pd.read_json('C:/Users/prana/Documents/Projects/hackED2025/turbo-alpaca/turbo/upload/file_data.json')
    
#     # Calculate the revenue data using your function
#     revenue_data = calculate_total_revenue_data(data)
    
#     # Generate the plot URL if data is available
#     revenue_plot_url = None
#     if revenue_data and revenue_data.get("grouped_data") is not None:
#         # plot_total_revenue has been updated to save the figure as an HTML file and return its URL
#         revenue_plot_url = plot_total_revenue(revenue_data["grouped_data"])
    
#     # Pass the plot URL in the context
#     context = {
#         "revenue_plot_url": revenue_plot_url,
#     }
#     return render(request, "upload/dashboard.html", context)

