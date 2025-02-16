from django.urls import path
from . import views

urlpatterns = [
    path('', views.user_login, name='login'),  # User login
    #path('', views.home, name='home'),  # Home page
    path('upload/', views.upload_file, name='upload_file'),  # File upload view

    # Authentication views
    path('home/', views.home, name='home'),  # Home page
    path('register/', views.register, name='register'),  # User registration
    path('login/', views.user_login, name='login'),  # User login
    path('logout/', views.user_logout, name='logout'),  # User logoutx
    path('query/', views.query_pinecone, name='query')
]