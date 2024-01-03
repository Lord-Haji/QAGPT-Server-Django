from django.db import models
from django.contrib.auth.models import User

class Scorecard(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    questions = models.JSONField()  # Stores questions and their options

    def __str__(self):
        return self.title
        

def user_directory_path(instance, filename):
    # File will be uploaded to MEDIA_ROOT/user_<name>/<filename>
    return '{0}/{1}'.format(instance.user.username, filename)

class AudioFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)  # To store the original file name
    audio = models.FileField(upload_to=user_directory_path, null=True)
    transcription = models.TextField(null=True, blank=True)
    upload_date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['id']),
        ]

    def __str__(self):
        return self.file_name
    
class Evaluation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    audio_files = models.ManyToManyField(AudioFile)
    scorecard = models.ForeignKey(Scorecard, on_delete=models.CASCADE)
    scorecard_title = models.CharField(max_length=100, null=True)
    result = models.JSONField()  # Stores the result of the evaluation
    pdf_report = models.FileField(upload_to='evaluation_reports/', null=True, blank=True)
    individual_reports = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Evaluation {self.id} by {self.user.username}"