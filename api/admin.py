from django.contrib import admin
from .models import (
    Category,
    Scorecard,
    AudioFile,
    Evaluation,
    Transcript,
    Utterance,
    KnowledgeBase,
)

admin.site.register(Category)
admin.site.register(Scorecard)
admin.site.register(AudioFile)
admin.site.register(Evaluation)
admin.site.register(Transcript)
admin.site.register(Utterance)
admin.site.register(KnowledgeBase)
