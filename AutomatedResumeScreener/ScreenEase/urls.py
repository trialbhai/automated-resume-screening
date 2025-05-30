from django.urls import path
from .views import home , UploadResume , Login, Register ,logout
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

urlpatterns = [
    path('', home, name='home'),
    path('upload/', UploadResume, name='upload_resume'),
    path("login/", Login, name="login"),
    path("register/", Register, name="register"),
    path('logout/', logout, name='logout'),

]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)