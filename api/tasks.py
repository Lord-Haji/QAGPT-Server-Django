from .models import Evaluation, AudioFile, Scorecard
from django.utils import timezone
import json

import google.generativeai as genai
from openai import OpenAI

client = OpenAI()

def perform_evaluation(user, audio_file_ids, scorecard_id, evaluation):
    evaluations = []
    # scorecard = Scorecard.objects.get(id=scorecard_id)
    # print(scorecard.questions)
    for audio_file_id in audio_file_ids:
        # Fetch the audio file
        print("Evaluating audio file with id: ", audio_file_id)
        audio_file = AudioFile.objects.get(id=audio_file_id)

        # Perform evaluation for this audio file
        # This is a placeholder - replace with your actual evaluation logic
        # responses = perform_audio_evaluation(audio_file.audio.path)
        
        evaluator = ScorecardEvaluator(scorecard_id, audio_file_id)

        evaluations.append({
            "audio_file_id": audio_file_id,
            "responses": evaluator.evaluate()
        })
        print("finished evaluating audio file with id: ", audio_file_id)

    # Update the evaluation with the final result
    final_result = {
        "status": "completed",
        "evaluations": evaluations
    }
    Evaluation.objects.filter(id=evaluation.id).update(
        result=final_result,
        completed_at=timezone.now()
    )


class ScorecardEvaluator:
    def __init__(self, scorecard_id, audio_file_id):
        self.scorecard = Scorecard.objects.get(id=scorecard_id)
        self.questions = self.scorecard.questions
        self.audio_file_path = AudioFile.objects.get(id=audio_file_id).audio.path
        self.transcript = ""
        self.questions_and_options = ""
        
    def transcribe(self):
        with open(self.audio_file_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language="en",
                prompt="1st Energy, 1st Saver, cooling-off period, NMI, MIRN, RACT",
                temperature=0.3,
                response_format="text"
            )
        self.transcript = transcript.replace("\n", " ")
        return self.transcript

    def construct_prompt(self):
        text = ""
        for question in self.questions:
            text += f"{question['text']} Options: {' or '.join(question['options'])}\n"
        
        for i, question in enumerate(self.questions):
            text += f"{i+1}. {question['text']} Options: {' or '.join(question['options'])}\n"
        self.questions_and_options = text.strip()
        print(self.questions_and_options)
        return self.questions_and_options
    
    def evaluate(self):
        self.transcribe()
        self.construct_prompt()
        
        schema_string = [{"question":"string","options":["string"],"llm_response":"string","reason":"string"}]
         
        sys_prompt = (
            f"You are a Quality Assurance Analyst who is tasked evaluate the transcript "
            f"based on the following questions and choose a given option with proper reasoning. \n"
            f"{self.questions_and_options}\n"
            f"Your Output should be in JSON with the keys being "
            f"question(''), options([]), llm_response('') and reason('')\n"
            f"In the following JSON Schema: for every question:"
            f"{{'questions': {schema_string}}}"
            f"With the question and options being the original ones provided and llm_response "
            f"being the option you chose and reason being the reason you chose that option\n"
        )
        user_prompt = f"Here is the transcript: \n{self.transcript}"
        prompt = f"{sys_prompt}\n{user_prompt}"
        
        messages = [
            {'role':'user',
             'parts': [prompt]}
        ]

        generation_config = genai.GenerationConfig(
            temperature=0
        )

        model = genai.GenerativeModel(model_name="gemini-pro",
                                      generation_config=generation_config)

        response = model.generate_content(messages)
        print("Hi from Gemini-Pro")
        
        response_dict = json.loads(response.text.strip('`').replace('json\n', ''))
        return response_dict
    
    # def evaluate_gpt(self):
    #     self.transcribe()
    #     self.construct_prompt()
        
    #     schema_string = [{"question":"string","options":["string"],"llm_response":"string","reason":"string"}]
         
    #     sys_prompt = (
    #         f"You are a Quality Assurance Analyst who is tasked evaluate the transcript "
    #         f"based on the following questions and choose a given option with proper reasoning. \n"
    #         f"{self.questions_and_options}\n"
    #         f"Your Output should be in JSON with the keys being "
    #         f"question(''), options([]), llm_response('') and reason('')\n"
    #         f"In the following JSON Schema: for every question:"
    #         f"{{'questions': {schema_string}}}"
    #         f"With the question and options being the original ones provided and llm_response "
    #         f"being the option you chose and reason being the reason you chose that option\n"
    #     )
    #     user_prompt = f"Here is the transcript: \n{self.transcript}"
        
    #     response = client.chat.completions.create(
    #         model="gpt-4-1106-preview",
    #         response_format={ "type": "json_object" },
    #         messages=[
    #             {"role": "system", "content": sys_prompt},
    #             {"role": "user", "content": user_prompt}
    #         ],
    #         temperature=0
    #     )
    #     response_dict = json.loads(response.choices[0].message.content)
    #     return response_dict