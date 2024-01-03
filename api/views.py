import threading
import os
from django.shortcuts import render
from django.conf import settings
from django.contrib.auth.models import User
from django.core.files import File
from django.core.files.base import ContentFile
from django.http import FileResponse
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken

from .tasks import perform_evaluation, combine_audio, generate_combined_filename, generate_pdf_report, generate_pdf_report_for_audio_file
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

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def combine_and_upload_audio(request):
    user = request.user
    audio_files = request.FILES.getlist('audio_files')  # Assuming multiple files are uploaded with the same key

    if not audio_files:
        return Response({'error': 'No audio files provided'}, status=status.HTTP_400_BAD_REQUEST)

    # Call a function to combine the audio files
    combined_audio, combined_audio_format = combine_audio(audio_files)

    # Saving the combined audio file in AudioFile model
    combined_filename = generate_combined_filename(audio_files, combined_audio_format)
    combined_file_content = ContentFile(combined_audio)

    combined_audio_file = AudioFile.objects.create(user=user, audio=combined_file_content, file_name=combined_filename)
    combined_audio_file.audio.save(combined_filename, combined_file_content)

    return Response({
        'message': 'Audio files combined successfully',
        'combined_audio_file_id': combined_audio_file.id
    })

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
        scorecard_title=scorecard.title,
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
def get_evaluation(request, evaluation_id=None):
    user = request.user
    if evaluation_id is None:
        evaluations = Evaluation.objects.filter(user=user)
        serializer = EvaluationSerializer(evaluations, many=True)
        return Response(serializer.data)
    else:
        try:
            evaluation = Evaluation.objects.get(id=evaluation_id, user=user)
            serializer = EvaluationSerializer(evaluation)
            return Response(serializer.data)
        except Evaluation.DoesNotExist:
            return Response({'error': 'Evaluation not found'}, status=404)
        
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def generate_and_retrieve_report(request, evaluation_id):
    try:
        evaluation = Evaluation.objects.get(id=evaluation_id, user=request.user)
    except Evaluation.DoesNotExist:
        return Response({'error': 'Evaluation not found'}, status=404)

    # Generate the report if it doesn't exist
    if not evaluation.pdf_report:
        report_path = generate_pdf_report(evaluation)
        evaluation.pdf_report.save(report_path, File(open(os.path.join(settings.MEDIA_ROOT, report_path), 'rb')))
        evaluation.save()

    # Serve the PDF file as a response
    return FileResponse(evaluation.pdf_report, as_attachment=True, filename=evaluation.pdf_report.name)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def generate_and_retrieve_audio_file_report(request, evaluation_id, audio_file_id):
    try:
        evaluation = Evaluation.objects.get(id=evaluation_id, user=request.user)
        if not evaluation.audio_files.filter(id=audio_file_id).exists():
            return Response({'error': 'Audio file not part of the evaluation'}, status=404)

        # Check if the report for this audio file already exists
        report_path = evaluation.individual_reports.get(str(audio_file_id))
        if not report_path:
            # Generate the report for this audio file 
            report_path = generate_pdf_report_for_audio_file(audio_file_id, evaluation)  # Implement this function
            evaluation.individual_reports[str(audio_file_id)] = report_path
            evaluation.save(update_fields=['individual_reports'])

        # Serve the report
        file_path = os.path.join(settings.MEDIA_ROOT, report_path)
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=os.path.basename(file_path))

    except Evaluation.DoesNotExist:
        return Response({'error': 'Evaluation not found'}, status=404)