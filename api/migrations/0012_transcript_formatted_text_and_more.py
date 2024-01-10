# Generated by Django 5.0 on 2024-01-10 05:48

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0011_transcript_alter_audiofile_transcription_utterance_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="transcript",
            name="formatted_text",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="utterance",
            name="low_confidence_words",
            field=models.JSONField(default=list),
        ),
        migrations.DeleteModel(
            name="Word",
        ),
    ]
