from .models import Evaluation, AudioFile, Scorecard
from django.utils import timezone

from openai import OpenAI
client = OpenAI()

def perform_evaluation(user, audio_file_ids, scorecard_id, evaluation):
    evaluations = []
    # scorecard = Scorecard.objects.get(id=scorecard_id)
    # print(scorecard.questions)
    construct_prompt()
    for audio_file_id in audio_file_ids:
        # Fetch the audio file
        audio_file = AudioFile.objects.get(id=audio_file_id)

        # Perform evaluation for this audio file
        # This is a placeholder - replace with your actual evaluation logic
        responses = perform_audio_evaluation(audio_file.audio.path)

        evaluations.append({
            "audio_file_id": audio_file_id,
            "responses": responses
        })

    # Update the evaluation with the final result
    final_result = {
        "status": "completed",
        "evaluations": evaluations
    }
    Evaluation.objects.filter(id=evaluation.id).update(
        result=final_result,
        completed_at=timezone.now()
    )

def perform_audio_evaluation(audio_file_path):
    # Placeholder for your actual evaluation logic
    # This should return a list of responses for the given audio file
    
    # transcription = transcribe_audio_file(audio_file_path)
    
    return [
        # {"question": "Q1", "options": ["Yes", "No"], "llm_response": "Yes"},
        # ... more questions and responses for this audio file ...
        
    ]

class ScorecardEvaluator:
    def __init__(self, scorecard_id, audio_file_path):
        self.scorecard = Scorecard.objects.get(id=scorecard_id)
        self.questions = self.scorecard.questions
        self.audio_file_path = audio_file_path
        self.transcript = ""
        self.questions_and_options = ""
        
    def transcribe(self):
        with open(self.audio_file_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language="en",
                prompt=self.construct_prompt(),
                temperature=0.3,
                response_format="text"
            )
        self.transcript = transcript
        return self.transcript

    def construct_prompt(self):
        text = ""
        for question in self.questions:
            text += f"{question['text']} Options: {' or '.join(question['options'])}\n"
        self.questions_and_options = text.strip()
        return self.questions_and_options