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
    upload_date = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return self.file_name






# class EvaluationResult(models.Model):
#     audio_file = models.ForeignKey(AudioFile, on_delete=models.CASCADE)
#     scorecard = models.ForeignKey(Scorecard, on_delete=models.SET_NULL, null=True)
#     result_data = models.JSONField()
#     evaluated_on = models.DateTimeField(auto_now_add=True)
#     # Additional fields as needed

#     def __str__(self):
#         return f"Result for {self.audio_file}"

