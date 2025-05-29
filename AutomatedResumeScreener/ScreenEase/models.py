from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.contrib import admin
import datetime



class Resume(models.Model):
    """Stores resume files with parsed text and extracted information."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    resume_file = models.FileField(upload_to="resumes/", unique=True)
    parsed_text = models.TextField(blank=True, null=True)
    skills = models.JSONField(blank=True, null=True)
    education = models.TextField(blank=True, null=True)
    experience = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True, blank=True)

    def __str__(self):
        return f"Resume #{self.id} - {self.user.username}"


class UploadedResume(models.Model):
    """Model for storing uploaded resumes separately."""
    file = models.FileField(upload_to='uploads/')

    created_at = models.DateTimeField(auto_now_add=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True, blank=True)
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
