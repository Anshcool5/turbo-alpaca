from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', views.user_login, name='login'),  # User login
    #path('', views.home, name='home'),  # Home page
    path('upload/', views.upload_file, name='upload_file'),  # File upload view

    # Password Reset URLs
    path('password_reset/', auth_views.PasswordResetView.as_view(template_name='password_reset.html'), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='password_reset_complete.html'), name='password_reset_complete'),
    path('test_email/', views.test_email, name='test_email'),
    
    # Authentication views
    path('home/', views.home, name='home'),  # Home page
    path('register/', views.register, name='register'),  # User registration
    path('login/', views.user_login, name='login'),  # User login
    path('logout/', views.user_logout, name='logout'),  # User logout
    path("query_documents/", views.query_documents, name="query_documents"),
    path("chatty/", views.chatbot_view, name="chatty"),
    path("generate/", views.generate_idea, name="generate"),
    # Dashboard
    path("dashboard/", views.dashboard, name="dashboard"),
    path('evaluate/', views.evaluate, name='evaluate'),  # Update 'generate_idea' to your view function name

    #chatbot
]