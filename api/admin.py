from django.contrib import admin
from .models import Scorecard, AudioFile, Evaluation, Transcript, Utterance


admin.site.register(Scorecard)
admin.site.register(AudioFile)
admin.site.register(Evaluation)
admin.site.register(Transcript)
admin.site.register(Utterance)
