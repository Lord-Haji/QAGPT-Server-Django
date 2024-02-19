from django.db import models
from django.contrib.auth.models import User


def user_directory_path(instance, filename, subfolder):
    # File will be uploaded to MEDIA_ROOT/user_<name>/<subfolder>/<filename>
    return f"{instance.user.username}/{subfolder}/{filename}"


class Category(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100, unique=True)
    keywords = models.JSONField(default=list)

    class Meta:
        verbose_name_plural = "Categories"

    def __str__(self):
        return f"Category: {self.name} (User: {self.user.username})"


class Vocabulary(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    words = models.JSONField(default=list)

    class Meta:
        verbose_name_plural = "Vocabularies"

    def __str__(self):
        return f"Vocabulary (User: {self.user.username})"


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
        return f"Scorecard: {self.title} (User: {self.user.username})"


class KnowledgeBase(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="knowledge_bases"
    )
    pdf = models.FileField(
        upload_to=lambda instance, filename: user_directory_path(
            instance, filename, "knowledge_bases"
        )
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Knowledge Base: {self.pdf.name} (User: {self.user.username})"


class AudioFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    audio = models.FileField(
        upload_to=lambda instance, filename: user_directory_path(
            instance, filename, "audio_files"
        )
    )
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
        return f"Audio File: {self.file_name} (User: {self.user.username})"


class Transcript(models.Model):
    audio_file = models.OneToOneField(
        AudioFile, on_delete=models.CASCADE, related_name="transcript"
    )

    def __str__(self):
        return f"Transcript for Audio File: {self.audio_file.file_name}"


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
        return f"Utterance: {self.text[:30]}... (Speaker: {self.speaker_label})"


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
        return (
            f"Evaluation Job {self.id} (User: {self.user.username},"
            f"Status: {self.status})"
        )


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
        upload_to=lambda instance, filename: user_directory_path(
            instance, filename, "evaluation_reports"
        )
    )
    status = models.CharField(
        max_length=10, choices=StatusChoices.choices, default=StatusChoices.PENDING
    )

    def __str__(self):
        return (
            f"Evaluation of Audio File: {self.audio_file.file_name} with "
            f"Scorecard: {self.scorecard.title} (Status: {self.status})"
        )
