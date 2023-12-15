import threading
from django.shortcuts import render
from django.contrib.auth.models import User
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken

from .tasks import perform_evaluation
from .models import Scorecard, AudioFile, Evaluation
from .serializers import EvaluationSerializer, ScorecardSerializer, AudioFileSerializer, UserSerializer
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

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

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def evaluate_audio_files(request):
    user = request.user
    audio_file_ids = request.data.get('audio_file_ids', [])
    scorecard_id = request.data.get('scorecard_id')

    audio_files = AudioFile.objects.filter(id__in=audio_file_ids, user=user)
    scorecard = Scorecard.objects.get(id=scorecard_id, user=user)

    # Create a placeholder evaluation object
    evaluation = Evaluation.objects.create(
        user=user,
        scorecard=scorecard,
        result={'status': 'processing'}
    )
    evaluation.audio_files.set(audio_files)
    
    # Start the evaluation in a background thread
    evaluation_thread = threading.Thread(target=perform_evaluation, 
                                         args=(user, audio_file_ids, scorecard_id, evaluation))
    evaluation_thread.start()

    # Return the evaluation job ID immediately
    serializer = EvaluationSerializer(evaluation)
    return Response(serializer.data)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_evaluation(request, evaluation_id):
    user = request.user
    try:
        evaluation = Evaluation.objects.get(id=evaluation_id, user=user)
        serializer = EvaluationSerializer(evaluation)
        return Response(serializer.data)
    except Evaluation.DoesNotExist:
        return Response({'error': 'Evaluation not found'}, status=404)
