from django.db import models
from django.contrib.auth.models import User

class Scorecard(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    questions = models.JSONField()  # Stores questions and their options

    def __str__(self):
        return self.title


class AudioFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='audio_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Audio file for {self.user.username} uploaded at {self.uploaded_at}"
