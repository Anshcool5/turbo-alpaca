from django.contrib import admin
from django.urls import path, include
from upload import views

urlpatterns = [
    path('', views.home, name='home'),
    path("", include("upload.urls")),
]