from django.contrib import admin
from django.urls import path, include
from upload import views
from django.conf import settings  # Import the settings module
from django.conf.urls.static import static

urlpatterns = [
    path("", include("upload.urls")),
    # path("chat/", include("chatbot.urls")),
    path('admin/', admin.site.urls),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
