from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    register,
    user_profile_view,
    CategoryViewSet,
    ScorecardViewSet,
    AudioFileViewSet,
    VocabularyViewSet,
    KnowledgeBaseViewSet,
    TranscriptViewSet,
    evaluate_audio_files,
    evaluate_audio_files_auto,
    get_evaluation,
    combine_and_upload_audio,
    generate_and_retrieve_evaluation_report,
)

from rest_framework_simplejwt.views import (
    TokenRefreshView,
    TokenBlacklistView,
)

router = DefaultRouter()
router.register(r"categories", CategoryViewSet)
router.register(r"scorecards", ScorecardViewSet)
router.register(r"audiofiles", AudioFileViewSet)
router.register(r"vocabularies", VocabularyViewSet)
router.register(r"knowledgebases", KnowledgeBaseViewSet)
router.register(r"transcripts", TranscriptViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path("register/", register, name="register"),
    path("auth/", include("trench.urls")),
    path("auth/", include("trench.urls.jwt")),
    # path("token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("token/blacklist/", TokenBlacklistView.as_view(), name="token_blacklist"),
    path("user/profile/", user_profile_view, name="user-profile"),
    path(
        "combine_and_upload_audio/",
        combine_and_upload_audio,
        name="combine_and_upload_audio",
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
