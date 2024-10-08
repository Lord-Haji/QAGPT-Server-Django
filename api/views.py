import threading
import os
from pydub import AudioSegment
import io
from django.core.files.base import ContentFile
from django.http import FileResponse
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken
from .tasks import (
    perform_evaluation,
    combine_audio,
    generate_combined_filename,
    generate_pdf_report_for_evaluation,
)
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
from .serializers import (
    UserRegistrationSerializer,
    UserProfileSerializer,
    CategorySerializer,
    VocabularySerializer,
    ScorecardSerializer,
    KnowledgeBaseSerializer,
    UtteranceSerializer,
    EvaluationJobSerializer,
    TranscriptSerializer,
    AudioFileSerializer,
)


# User registration view
@api_view(["POST"])
def register(request):
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response(
            {
                "user": {
                    "username": user.username,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "email": user.email,
                },
                "refresh": str(refresh),
                "access": str(refresh.access_token),
            },
            status=status.HTTP_201_CREATED,
        )
    else:
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_profile_view(request):
    user = request.user
    serializer = UserProfileSerializer(user)
    return Response(serializer.data)


class CategoryViewSet(viewsets.ModelViewSet):
    """
    A viewset for viewing and editing category instances.
    """

    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # This ensures that users can only access their own categories
        return Category.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # Assign the current user as the owner of the category
        serializer.save(user=self.request.user)


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
        audio_file = self.request.data.get("audio")
        # Calculate duration of the audio file
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_file.read()))
        duration_seconds = (
            len(audio_segment) / 1000.0
        )  # Convert milliseconds to seconds

        serializer.save(
            user=self.request.user, audio=audio_file, duration_seconds=duration_seconds
        )


class VocabularyViewSet(viewsets.ModelViewSet):
    queryset = Vocabulary.objects.all()
    serializer_class = VocabularySerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Vocabulary.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class KnowledgeBaseViewSet(viewsets.ModelViewSet):
    queryset = KnowledgeBase.objects.all()
    serializer_class = KnowledgeBaseSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def get_queryset(self):
        return KnowledgeBase.objects.filter(user=self.request.user)


class TranscriptViewSet(viewsets.ModelViewSet):
    queryset = Transcript.objects.all()
    serializer_class = TranscriptSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # This ensures that users can only access their own transcripts
        return Transcript.objects.filter(audio_file__user=self.request.user)

    def update(self, request, *args, **kwargs):
        transcript = self.get_object()
        serializer = self.get_serializer(transcript, data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        # Handle Utterance updates if provided in the request
        utterances_data = request.data.get("utterances")
        if utterances_data:
            for utterance_data in utterances_data:
                utterance_id = utterance_data.get("id")
                if utterance_id:
                    utterance = Utterance.objects.get(
                        id=utterance_id, transcript=transcript
                    )
                    utterance_serializer = UtteranceSerializer(
                        utterance, data=utterance_data
                    )
                    if utterance_serializer.is_valid():
                        utterance_serializer.save()

        return Response(serializer.data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def combine_and_upload_audio(request):
    user = request.user
    audios = request.FILES.getlist("audios")

    if not audios:
        return Response(
            {"error": "No audio files provided"}, status=status.HTTP_400_BAD_REQUEST
        )

    # Call a function to combine the audio files
    combined_audio, combined_audio_format = combine_audio(audios)

    # Create a ContentFile for the combined audio
    combined_file_content = ContentFile(combined_audio)

    # Calculate the duration of the combined audio
    combined_audio_segment = AudioSegment.from_file(io.BytesIO(combined_audio))
    combined_duration_seconds = (
        len(combined_audio_segment) / 1000.0
    )  # Convert milliseconds to seconds

    # Generate a filename for the combined audio file
    combined_filename = generate_combined_filename(audios, combined_audio_format)

    # Create a new AudioFile instance with the combined audio
    combined_audio_file = AudioFile(
        user=user,
        audio=combined_file_content,
        file_name=combined_filename,
        duration_seconds=combined_duration_seconds,
    )

    combined_audio_file.audio.save(combined_filename, combined_file_content)
    # Save the AudioFile instance
    combined_audio_file.save()

    response_serializer = AudioFileSerializer(combined_audio_file)
    return Response(response_serializer.data)


# Currently uses multi level threading
# for concurrently running transcription and evaluation.
# Revert to a queue based system for better scalability.
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def evaluate_audio_files(request):
    user = request.user
    audio_file_ids = request.data.get("audio_file_ids", [])
    scorecard_id = request.data.get("scorecard_id")

    audio_files = AudioFile.objects.filter(id__in=audio_file_ids, user=user)

    # Create an EvaluationJob instance
    evaluation_job = EvaluationJob.objects.create(
        user=user, status=EvaluationJob.StatusChoices.PROCESSING
    )
    evaluation_job.audio_files.set(audio_files)
    print("Evaluation job created", evaluation_job.id, scorecard_id)
    # Start the evaluation process in a background thread
    evaluation_thread = threading.Thread(
        target=perform_evaluation, args=(evaluation_job.id, scorecard_id)
    )
    evaluation_thread.start()
    print("Evaluation thread started", evaluation_job.id, scorecard_id)

    # Serialize and return the EvaluationJob
    serializer = EvaluationJobSerializer(evaluation_job)
    return Response(serializer.data)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def evaluate_audio_files_auto(request):
    user = request.user
    audio_file_ids = request.data.get("audio_file_ids", [])

    # Retrieve AudioFile instances for the user
    audio_files = AudioFile.objects.filter(id__in=audio_file_ids, user=user)

    if not audio_files:
        return Response({"error": "No audio files found"}, status=400)

    # Create an EvaluationJob instance
    evaluation_job = EvaluationJob.objects.create(
        user=user, status=EvaluationJob.StatusChoices.PROCESSING
    )
    evaluation_job.audio_files.set(audio_files)

    # Start the evaluation process in a background thread
    evaluation_thread = threading.Thread(
        target=perform_evaluation, args=(evaluation_job.id, user.id)
    )
    evaluation_thread.start()

    # Serialize the EvaluationJob
    serializer = EvaluationJobSerializer(evaluation_job)

    # Return the serialized EvaluationJob data
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_evaluation(request, evaluation_job_id):
    user = request.user

    try:
        # Fetch the EvaluationJob that belongs to the user and has the given ID
        evaluation_job = EvaluationJob.objects.get(id=evaluation_job_id, user=user)

        # Serialize the EvaluationJob along with its related Evaluations
        serializer = EvaluationJobSerializer(evaluation_job)
        return Response(serializer.data)

    except EvaluationJob.DoesNotExist:
        return Response({"error": "Evaluation Job not found"}, status=404)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def generate_and_retrieve_evaluation_report(request, evaluation_id):
    try:
        evaluation = Evaluation.objects.get(
            id=evaluation_id, evaluation_job__user=request.user
        )

        if not evaluation.pdf_report:
            # Generate the report if it doesn't exist
            generate_pdf_report_for_evaluation(evaluation)  # Ensure this saves the file

        # Use the 'path' attribute to get the absolute path
        file_path = evaluation.pdf_report.path

        return FileResponse(
            open(file_path, "rb"),
            as_attachment=True,
            filename=os.path.basename(file_path),
        )

    except Evaluation.DoesNotExist:
        return Response({"error": "Evaluation not found"}, status=404)
    except FileNotFoundError:
        return Response({"error": "File not found"}, status=404)
