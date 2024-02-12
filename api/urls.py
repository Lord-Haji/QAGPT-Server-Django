from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    CategoryViewSet,
    ScorecardViewSet,
    AudioFileViewSet,
    register,
    evaluate_audio_files,
    evaluate_audio_files_auto,
    get_evaluation,
    get_utterance_with_transcript,
    combine_and_upload_audio,
    generate_and_retrieve_evaluation_report,
)

from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenBlacklistView,
)

router = DefaultRouter()
router.register(r"categories", CategoryViewSet)
router.register(r"scorecards", ScorecardViewSet)
router.register(r"audiofiles", AudioFileViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path("register/", register, name="register"),
    path("token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("token/blacklist/", TokenBlacklistView.as_view(), name="token_blacklist"),
    path(
        "combine_and_upload_audio/",
        combine_and_upload_audio,
        name="combine_and_upload_audio",
    ),
    path(
        "audio-files/<int:audio_file_id>/utterance-with-transcript/",
        get_utterance_with_transcript,
        name="utterance_with_transcript",
    ),
    path("evaluate/", evaluate_audio_files, name="evaluate_audio_files"),
    path("evaluate-auto/", evaluate_audio_files_auto, name="evaluate_audio_files_auto"),
    path("evaluation/<int:evaluation_job_id>/", get_evaluation, name="get_evaluation"),
    # Disable paths
    # path("evaluations/", get_evaluation, name="get_evaluation_all"),
    # path(
    #     "evaluation/<int:evaluation_id>/pdf_report/",
    #     generate_and_retrieve_report,
    #     name="generate_and_retrieve_report",
    # ),
    path(
        "evaluation/<int:evaluation_id>/report/",
        generate_and_retrieve_evaluation_report,
        name="generate_and_retrieve_evaluation_report",
    ),
]
