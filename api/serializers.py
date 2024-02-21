from rest_framework import serializers
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
from .tasks import get_user_evaluation_stats
from django.contrib.auth.models import User
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import json


class UserSerializer(serializers.ModelSerializer):
    total_evaluated_duration = serializers.SerializerMethodField()
    total_evaluated_files = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = [
            "username",
            "password",
            "email",
            "total_evaluated_duration",
            "total_evaluated_files",
        ]
        extra_kwargs = {"password": {"write_only": True}}

    def get_total_evaluated_duration(self, user):
        return get_user_evaluation_stats(user)["total_minutes"]

    def get_total_evaluated_files(self, user):
        return get_user_evaluation_stats(user)["total_files"]


class CategorySerializer(serializers.ModelSerializer):
    """
    Serializer for the Category model.

    Attributes:
        keywords (List[str]): List of keywords associated with the category.

    Meta:
        model (Category): The Category model.
        fields (List[str]): List of fields to include in the serialized representation.
        read_only_fields (List[str]): List of fields that should be read-only.
    """

    keywords = serializers.ListField(child=serializers.CharField(max_length=100))

    class Meta:
        model = Category
        fields = ["id", "user", "name", "keywords"]
        read_only_fields = ["user"]


class VocabularySerializer(serializers.ModelSerializer):
    """
    Serializer for the Vocabulary model.

    Attributes:
        words (List[str]): List of words associated with the vocabulary.

    Meta:
        model (Vocabulary): The Vocabulary model.
        fields (List[str]): List of fields to include in the serialized representation.
        read_only_fields (List[str]): List of fields that should be read-only.
    """

    words = serializers.ListField(child=serializers.CharField(max_length=100))

    class Meta:
        model = Vocabulary
        fields = ["id", "user", "words"]
        read_only_fields = ["user"]


scorecard_schema = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,  # Min 2 options
            },
            "correct": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "score": {"type": "number"},
            "use_knowledge_base": {"type": "boolean"},
        },
        "required": ["text", "options", "correct", "score"],
        "additionalProperties": False,
    },
}


class ScorecardSerializer(serializers.ModelSerializer):
    """
    Serializer class for the Scorecard model.

    Serializes the following fields:
    - id (int): The ID of the scorecard.
    - title (str): The title of the scorecard.
    - category (CategorySerializer): The category associated with the scorecard.
    - questions (list): The list of questions in the scorecard.

    Attributes:
        category (CategorySerializer): Serializer for the category field.

    Meta:
        model (Scorecard): The model class associated with this serializer.
        fields (list): The fields to include in the serialized representation.

    Methods:
        validate_questions: Validates the questions field of the serialized data.
    """

    category = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(), allow_null=True, required=False
    )

    def validate_category(self, value):
        if value is not None:
            user = self.context["request"].user
            if value.user != user:
                raise serializers.ValidationError(
                    "You cannot assign a category that does not belong to you."
                )
        return value

    class Meta:
        model = Scorecard
        fields = ["id", "title", "category", "questions"]

    def validate_questions(self, value):
        """
        Validates the questions field of the serialized data.

        Args:
            value (str or list): The value of the questions field.

        Returns:
            list: The validated questions field.

        Raises:
            serializers.ValidationError: If the questions field is invalid.
        """
        # Convert value to Python object if it's a string
        if isinstance(value, str):
            value = json.loads(value)

        # Validate the questions field
        try:
            validate(instance=value, schema=scorecard_schema)
        except ValidationError as e:
            raise serializers.ValidationError(f"Invalid questions format: {e.message}")

        for question in value:
            if "correct" in question and "options" in question:
                if not all(
                    answer in question["options"] for answer in question["correct"]
                ):
                    raise serializers.ValidationError(
                        "All correct answers must be among the provided options."
                    )
            # Handle use_knowledge_base field
            if "use_knowledge_base" in question:
                if not isinstance(question["use_knowledge_base"], bool):
                    raise serializers.ValidationError(
                        "The 'use_knowledge_base' field must be a boolean."
                    )

        # Check if the sum of scores is 100
        total_score = sum(question.get("score", 0) for question in value)
        if total_score != 100:
            raise serializers.ValidationError(
                "The sum of all question scores must be 100."
            )

        return value


class KnowledgeBaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeBase
        fields = ["id", "pdf", "created_at", "updated_at"]


class UtteranceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Utterance
        fields = [
            "id",
            "speaker_label",
            "start_time",
            "end_time",
            "confidence",
            "text",
            "low_confidence_words",
        ]


class TranscriptSerializer(serializers.ModelSerializer):
    utterances = UtteranceSerializer(many=True, read_only=True)

    class Meta:
        model = Transcript
        fields = ["audio_file", "utterances"]


class AudioFileSerializer(serializers.ModelSerializer):
    transcription = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = AudioFile
        fields = [
            "id",
            "user",
            "file_name",
            "audio",
            "duration_seconds",
            "transcription",
            "upload_date",
        ]
        read_only_fields = ["user", "duration_seconds", "file_name", "upload_date"]


class EvaluationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Evaluation
        fields = ["evaluation_job", "audio_file", "status", "scorecard", "result"]


class EvaluationJobSerializer(serializers.ModelSerializer):
    evaluations = EvaluationSerializer(many=True, read_only=True)

    class Meta:
        model = EvaluationJob
        fields = [
            "id",
            "user",
            "audio_files",
            "status",
            "evaluations",
            "created_at",
            "completed_at",
        ]
