from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken
from .models import Scorecard, AudioFile
from .serializers import ScorecardSerializer, AudioFileSerializer, UserSerializer

class ScorecardViewSet(viewsets.ModelViewSet):
    queryset = Scorecard.objects.all()
    serializer_class = ScorecardSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Ensures users can only access their own scorecards
        return Scorecard.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class AudioFileViewSet(viewsets.ModelViewSet):
    queryset = AudioFile.objects.all()
    serializer_class = AudioFileSerializer
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get_queryset(self):
        # Ensures users can only access their own audio files
        return AudioFile.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # Extract the audio file from the request
        audio_file = self.request.data.get('audio')
        # Save the audio file along with the user who uploaded it
        serializer.save(user=self.request.user, audio=audio_file)

# User registration view
@api_view(['POST'])
def register(request):
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
        user = User.objects.create_user(**serializer.validated_data)
        refresh = RefreshToken.for_user(user)
        return Response({
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        })
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
