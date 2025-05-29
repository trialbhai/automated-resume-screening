from django.urls import path
from .views import home , UploadResume , Login, Register
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

urlpatterns = [
    path('home/', home, name='home'),
    path('upload/', UploadResume, name='upload_resume'),
    path("login/", Login, name="login"),
    path("register/", Register, name="register"),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('password_change/', auth_views.PasswordChangeView.as_view(), name='password_change'),

]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)