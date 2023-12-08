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
    file_name = models.CharField(max_length=255, default='unnamed_file')
    drive_file_id = models.CharField(max_length=255, default='NOT_SET')  # Google Drive file ID
    upload_date = models.DateTimeField(auto_now_add=True)
    # Additional fields as needed

    def __str__(self):
        return self.file_name



class EvaluationResult(models.Model):
    audio_file = models.ForeignKey(AudioFile, on_delete=models.CASCADE)
    scorecard = models.ForeignKey(Scorecard, on_delete=models.SET_NULL, null=True)
    result_data = models.JSONField()
    evaluated_on = models.DateTimeField(auto_now_add=True)
    # Additional fields as needed

    def __str__(self):
        return f"Result for {self.audio_file}"

