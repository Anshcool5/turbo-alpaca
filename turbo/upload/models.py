from django.db import models
from django.contrib.auth.models import User



# File model
class File(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="files")
    file_name = models.CharField(max_length=255)  # Store the file name
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Timestamp of upload

    def __str__(self):
        return self.file_name