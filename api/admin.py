from django.contrib import admin
from .models import (
    Category,
    Scorecard,
    AudioFile,
    Evaluation,
    EvaluationJob,
    Transcript,
    Utterance,
    Vocabulary,
    KnowledgeBase,
)


admin.site.register(Category)
admin.site.register(Scorecard)
admin.site.register(AudioFile)
admin.site.register(EvaluationJob)
admin.site.register(Evaluation)
admin.site.register(Transcript)
admin.site.register(Utterance)
admin.site.register(Vocabulary)
admin.site.register(KnowledgeBase)
