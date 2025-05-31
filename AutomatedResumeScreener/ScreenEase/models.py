from django.db import models
from django.contrib.auth.models import User
from django.contrib import admin

class Resume(models.Model):
    """Stores resume files with parsed text and extracted information."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, unique=True)
    work = models.CharField(max_length=50, blank=True, null=True)
    resume_file = models.FileField(upload_to="resumes/", unique=True)
    parsed_text = models.TextField(blank=True, null=True)
    skills = models.JSONField(blank=True, null=True)
    education = models.TextField(blank=True, null=True)
    experience = models.TextField(blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Resume #{self.id} - {self.user.username}"

class Employee(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    linkedin_profile = models.URLField(blank=True, null=True)
    resume = models.OneToOneField(Resume, on_delete=models.SET_NULL, null=True, blank=True)
    profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    phone_number = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self):
        return self.user.username

class Employer(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    company_name = models.CharField(max_length=255, blank=True, null=True)
    company_description = models.TextField(blank=True, null=True)
    company_website = models.URLField(blank=True, null=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    linkedin_profile = models.URLField(blank=True, null=True)

    def __str__(self):
        return f"Employer: {self.user.username} - {self.company_name}"

class UploadedResume(models.Model):
    """Model for storing uploaded resumes separately."""
    file = models.FileField(upload_to='uploads/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Uploaded Resume: {self.file.name}"

@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    """Admin display configuration for Resume model."""
    list_display = ('user', 'resume_file', 'uploaded_at')
    search_fields = ('user__username', 'skills', 'education')
    ordering = ('-uploaded_at',)

@admin.register(UploadedResume)
class UploadedResumeAdmin(admin.ModelAdmin):
    """Admin display configuration for UploadedResume model."""
    list_display = ('file', 'created_at')
    search_fields = ('file',)
    ordering = ('-created_at',)
