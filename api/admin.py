from django.contrib import admin
from .models import (
    Scorecard,
    AudioFile,
    Evaluation,
    EvaluationJob,
    Transcript,
    Utterance,
    KnowledgeBase,
)


admin.site.register(Scorecard)
admin.site.register(AudioFile)
admin.site.register(EvaluationJob)
admin.site.register(Evaluation)
admin.site.register(Transcript)
admin.site.register(Utterance)
admin.site.register(KnowledgeBase)
