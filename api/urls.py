from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ScorecardViewSet, AudioFileViewSet, register, evaluate_audio_files, get_evaluation, combine_and_upload_audio, generate_and_retrieve_report

router = DefaultRouter()
router.register(r'scorecards', ScorecardViewSet)
router.register(r'audiofiles', AudioFileViewSet)

from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenBlacklistView
)

urlpatterns = [
    path('', include(router.urls)),

    path('register/', register, name='register'),
    path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('token/blacklist/', TokenBlacklistView.as_view(), name='token_blacklist'),
    
    path('combine_and_upload_audio/', combine_and_upload_audio, name='combine_and_upload_audio'),
    path('evaluate/', evaluate_audio_files, name='evaluate_audio_files'),
    path('evaluation/<int:evaluation_id>/', get_evaluation, name='get_evaluation'),
    path('evaluations/', get_evaluation, name='get_evaluation_all'),
    path('evaluation/<int:evaluation_id>/report', generate_and_retrieve_report, name='generate_and_retrieve_report')
]
 