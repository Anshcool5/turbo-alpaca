from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import json

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login, authenticate, logout

# Load the Hugging Face embedding model (You can choose a different model from Hugging Face)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def home(request):
    return render(request, "home.html")

# Function to extract text from a PDF file
def load_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split text into smaller chunks
def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate embeddings using Hugging Face model
def embed_text_chunks_with_huggingface(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    return embeddings.tolist()  # Convert tensor to list for JSON response

# Django View to upload and process file
def upload_file(request):
    if request.method == "POST" and request.FILES.get("uploaded_file"):
        uploaded_file = request.FILES["uploaded_file"]
        file_name = uploaded_file.name.lower()

        # Extract text based on file type
        if file_name.endswith(".pdf"):
            text = load_pdf(uploaded_file)
        else:
            return JsonResponse({"error": "Unsupported file format"}, status=400)

        # Split text into smaller chunks
        chunks = split_text_into_chunks(text)

        # Get embeddings using Hugging Face model
        embeddings = embed_text_chunks_with_huggingface(chunks)

        # Return response with embeddings
        return JsonResponse({"embeddings": embeddings})

    return render(request, "upload.html")


from django.shortcuts import render, redirect


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Log the user in after registration
            return redirect('home')  # Redirect to a home page or dashboard
    else:
        form = UserCreationForm()
    return render(request, 'registration.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # Redirect to a home page or dashboard
        else:
            # Return an error message
            return render(request, 'accounts/login.html', {'error': 'Invalid username or password'})
    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    return redirect('home')  # Redirect to the home page or login page