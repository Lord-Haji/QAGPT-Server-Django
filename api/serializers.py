from rest_framework import serializers
from .models import (
    Category,
    Scorecard,
    AudioFile,
    Transcript,
    Utterance,
    Evaluation,
    KnowledgeBase,
)
from django.contrib.auth.models import User
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import json


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


class CategorySerializer(serializers.ModelSerializer):
    keywords = serializers.ListField(
        child=serializers.CharField(max_length=100)
    )
    class Meta:
        model = Category
        fields = ["id", "name", "keywords"]


class ScorecardSerializer(serializers.ModelSerializer):
    category = CategorySerializer(read_only=True)

    class Meta:
        model = Scorecard
        fields = ["id", "title", "category", "questions"]

    def validate_questions(self, value):
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


class UtteranceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Utterance
        fields = [
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
        fields = ["audio_file", "text", "utterances"]


class AudioFileSerializer(serializers.ModelSerializer):
    transcription = TranscriptSerializer(read_only=True)

    class Meta:
        model = AudioFile
        fields = ["id", "user", "file_name", "audio", "transcription", "upload_date"]
        read_only_fields = ["user", "upload_date"]

    # def create(self, validated_data):
    #     # You can add additional logic here if needed
    #     return AudioFile.objects.create(**validated_data)


class KnowledgeBaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeBase
        fields = ["user", "pdf", "created_at", "updated_at"]


class EvaluationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Evaluation
        fields = [
            "id",
            "user",
            "audio_files",
            "scorecard",
            "scorecard_title",
            "result",
            "pdf_report",
            "individual_reports",
            "created_at",
            "completed_at",
        ]


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["username", "password", "email"]
        extra_kwargs = {"password": {"write_only": True}}
