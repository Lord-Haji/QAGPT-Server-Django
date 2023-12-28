from rest_framework import serializers
from .models import Scorecard, AudioFile, Evaluation
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
                "minItems": 2 # Min 2 options
            },
            "correct": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1
            },
            "score": {"type": "number"}
        },
        "required": ["text", "options", "correct", "score"],
        "additionalProperties": False
    }
}





class ScorecardSerializer(serializers.ModelSerializer):
    class Meta:
        model = Scorecard
        fields = ['id', 'title', 'questions']

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
            if 'correct' in question and 'options' in question:
                if not all(answer in question['options'] for answer in question['correct']):
                    raise serializers.ValidationError("All correct answers must be among the provided options.")
                
        # Check if the sum of scores is 100
        total_score = sum(question.get('score', 0) for question in value)
        if total_score != 100:
            raise serializers.ValidationError("The sum of all question scores must be 100.")


        return value



class AudioFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioFile
        fields = ['id', 'user', 'file_name', 'audio', 'transcription', 'upload_date']
        read_only_fields = ['user', 'upload_date']

    # def create(self, validated_data):
    #     # You can add additional logic here if needed
    #     return AudioFile.objects.create(**validated_data)
    
class EvaluationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Evaluation
        fields = ['id', 'user', 'audio_files', 'scorecard', 'scorecard_title', 'result', 'created_at', 'completed_at']

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username', 'password', 'email']
        extra_kwargs = {'password': {'write_only': True}}
