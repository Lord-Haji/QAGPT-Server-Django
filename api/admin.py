from django.contrib import admin
from .models import (
    Category,
    Vocabulary,
    Scorecard,
    KnowledgeBase,
    AudioFile,
    Transcript,
    Utterance,
    EvaluationJob,
    Evaluation,
)


admin.site.register(Category)
admin.site.register(Vocabulary)
admin.site.register(Scorecard)
admin.site.register(KnowledgeBase)
admin.site.register(AudioFile)
admin.site.register(Transcript)
admin.site.register(Utterance)
admin.site.register(EvaluationJob)
admin.site.register(Evaluation)
