from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ScorecardViewSet, AudioFileViewSet, register

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

]
 