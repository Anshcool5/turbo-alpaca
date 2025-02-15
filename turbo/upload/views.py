from django.shortcuts import render

def home(request):
    return render(request, "home.html")

def upload_file(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("uploaded_file")
        if uploaded_file:
            print("WORKING")  # Logs the content for debugging
            return HttpResponse("File uploaded successfully!")
        else:
            return HttpResponse("No file uploaded.", status=400)
    return render(request, "upload.html")