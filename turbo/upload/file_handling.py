from django.http import HttpResponse, Http404
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .models import File
import shutil
import os


def create_file_record(user, uploaded_file, file_name):
    """
    Creates a File record in the database and saves the file to the media folder.
    Args:
        user: The logged-in user (request.user).
        uploaded_file: The file uploaded by the user.
        file_name: The name of the uploaded file.
    Returns:
        The created File record.
    """
    # Define the path where the file will be saved
    target_directory = os.path.join(default_storage.location, 'uploads')
    file_path = os.path.join(target_directory, file_name)

    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)  # Create the directory if it doesn't exist

    # Use shutil to copy the file to the media folder
    with uploaded_file.open('rb') as src_file:
        with default_storage.open(file_path, 'wb') as dest_file:
            shutil.copyfileobj(src_file, dest_file)

    # Create a File record in the database, associated with the logged-in user
    file_record = File.objects.create(
        user=user,  # Associate the file with the logged-in user
        file_name=file_name
    )

    return file_record


def user_files(request):
    if request.user.is_authenticated:
        # Get all files uploaded by the logged-in user
        files = File.objects.filter(user=request.user)
        return render(request, "user_files.html", {"files": files})
    else:
        return redirect("login")


def download_file(request, file_id):
    if request.user.is_authenticated:
        try:
            # Get the file record, ensuring it belongs to the logged-in user
            file_record = File.objects.get(id=file_id, user=request.user)
            file_path = default_storage.path(f"uploads/{file_record.file_name}")
            with open(file_path, "rb") as file:
                response = HttpResponse(file.read(), content_type="application/octet-stream")
                response["Content-Disposition"] = f"attachment; filename={file_record.file_name}"
                return response
        except File.DoesNotExist:
            raise Http404("File not found")
    else:
        return redirect("login")
    
    

# def create_file_record(user, uploaded_file):
#     """
#     Creates a File record in the database and saves the file to the media folder.
#     Args:
#         user: The logged-in user (request.user).
#         uploaded_file: The file uploaded by the user.
#     Returns:
#         The created File record.
#     """
#     file_name = uploaded_file.name.lower()

#     # Save the file to the media folder
#     file_path = default_storage.save(f"uploads/{file_name}", ContentFile(uploaded_file.read()))

#     # Create a File record in the database, associated with the logged-in user
#     file_record = File.objects.create(
#         user=user,  # Associate the file with the logged-in user
#         file_name=file_name
#     )

#     return file_record