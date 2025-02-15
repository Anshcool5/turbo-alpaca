from django.shortcuts import render, redirect
#import httpresponse
from django.http import HttpResponse
# Create your views here.


def upload_file(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("uploaded_file")
        if uploaded_file:
            # Just reading the file for now
            # content = uploaded_file.read().decode("utf-8")  # Assuming a text file
            print("WORKING")  # Logs the content for debugging
            return HttpResponse("File uploaded successfully!")
        else:
            return HttpResponse("No file uploaded.", status=400)

    return render(request, "upload.html")