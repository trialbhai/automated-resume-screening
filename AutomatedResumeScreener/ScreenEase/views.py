from django.shortcuts import render , redirect
from .models import Resume
from .utils import extract_text_from_pdf, extract_text_from_docx , extract_resume_details
import os
from django.core.files.storage import FileSystemStorage
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from .models import UploadedResume

def home(request):
    return render(request, 'home.html')

def UploadResume(request):
    if request.method == "POST" and request.FILES["resume"] :
        if not request.user.is_authenticated:  
         return redirect("login")  
        
        resume_file = request.FILES["resume"]
        resume_instance = UploadedResume(file=resume_file)
        resume_instance.save()
        resume = Resume.objects.create(user=request.user, resume_file=resume_file)
        file_path = resume.resume_file.path

        if file_path.endswith(".pdf"):
            resume.parsed_text = extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            resume.parsed_text = extract_text_from_docx(file_path)

        # Extract details
        extracted_info = extract_resume_details(resume.parsed_text)
        resume.skills = extracted_info["skills"]
        resume.education = extracted_info["education"]
        resume.experience = extracted_info["experience"]

        resume.save()
        return render(request, "upload_success.html", {"resume": resume})

    return render(request, "upload_resume.html")






def upload_success(request):
    uploaded_files = UploadedResume.objects.all()  # Fetch files from DB
    return render(request, 'upload_success.html', {'uploaded_files': uploaded_files})




def Login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            return render(request, "login.html", {"error": "Invalid credentials"})
    return render(request, "login.html")

def Register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        if password != confirm_password:
            return render(request, "register.html", {"error": "Passwords do not match"})

        user = User.objects.create_user(username=username, password=password)
        user.save()
        return redirect("login")

    return render(request, "register.html")



