from django.db import models
from django.contrib.auth.models import User


class Category(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100, unique=True)
    keywords = models.JSONField(default=list)

    class Meta:
        verbose_name_plural = "Categories"

    def __str__(self):
        return f"{self.user.username} - {self.name}"


class Scorecard(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    category = models.OneToOneField(
        Category,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="scorecard",
        verbose_name="Category",
    )
    questions = models.JSONField()  # Stores questions and their options

    def __str__(self):
        return self.title


def user_audio_directory_path(instance, filename):
    # File will be uploaded to MEDIA_ROOT/user_<name>/audio_files/<filename>
    return "{0}/audio_files/{1}".format(instance.user.username, filename)


def user_knowledge_base_directory_path(instance, filename):
    # File will be uploaded to MEDIA_ROOT/user_<id>/knowledge_bases/<filename>
    return "{0}/knowledge_bases/{1}".format(instance.user.username, filename)


def user_evaluation_report_directory_path(instance, filename):
    # File will be uploaded to MEDIA_ROOT/user_<name>/evaluation_reports/<filename>
    return "{0}/evaluation_reports/{1}".format(
        instance.evaluation_job.user.username, filename
    )


class Vocabulary(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    words = models.JSONField(default=list)

    def __str__(self):
        return f"Vocabulary for {self.user.username}"


class KnowledgeBase(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="knowledge_bases"
    )
    pdf = models.FileField(
        upload_to=user_knowledge_base_directory_path, null=True, blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Knowledge Base PDF {self.pdf.name} for {self.user.username}"


class AudioFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    audio = models.FileField(upload_to=user_audio_directory_path, null=True)
    duration_seconds = models.FloatField(default=0.0)  # Default in seoonds
    transcription = models.OneToOneField(
        "Transcript", on_delete=models.CASCADE, null=True, blank=True
    )
    upload_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["id"]),
        ]

    def __str__(self):
        return self.file_name


class Transcript(models.Model):
    audio_file = models.OneToOneField(
        AudioFile, on_delete=models.CASCADE, related_name="transcript"
    )

    def __str__(self):
        return f"Transcript for {self.audio_file.file_name}"


class Utterance(models.Model):
    transcript = models.ForeignKey(
        Transcript, on_delete=models.CASCADE, related_name="utterances"
    )
    speaker_label = models.CharField(max_length=2)
    start_time = models.FloatField()
    end_time = models.FloatField()
    confidence = models.FloatField()
    text = models.TextField()
    low_confidence_words = models.JSONField(default=dict)

    def __str__(self):
        return f"Speaker {self.speaker_label}: {self.text[:30]}..."


class EvaluationJob(models.Model):
    class StatusChoices(models.TextChoices):
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    audio_files = models.ManyToManyField(AudioFile)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=10, choices=StatusChoices.choices, default=StatusChoices.PROCESSING
    )

    def __str__(self):
        return f"Evaluation Job {self.id} by {self.user.username} ({self.status})"


class Evaluation(models.Model):
    class StatusChoices(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        # Add more status choices as needed

    evaluation_job = models.ForeignKey(
        EvaluationJob,
        on_delete=models.CASCADE,
        related_name="evaluations",
        blank=True,
        null=True,
    )
    audio_file = models.ForeignKey(AudioFile, on_delete=models.CASCADE, null=True)
    scorecard = models.ForeignKey(Scorecard, on_delete=models.CASCADE, null=True)
    result = models.JSONField()
    pdf_report = models.FileField(
        upload_to=user_evaluation_report_directory_path, null=True, blank=True
    )
    status = models.CharField(
        max_length=10, choices=StatusChoices.choices, default=StatusChoices.PENDING
    )

    def __str__(self):
        return (
            f"Evaluation of {self.audio_file.file_name} with "
            f"{self.scorecard.title} - {self.status}"
        )
