from .models import Evaluation, AudioFile, Scorecard
from django.utils import timezone
import json
import re
import os
import io
import tempfile
from pydub import AudioSegment
import google.generativeai as genai
from openai import OpenAI


def combine_audio(audio_files):
    combined = AudioSegment.empty()
    for file in audio_files:
        file_content = file.read()
        sound = AudioSegment.from_file(io.BytesIO(file_content))
        combined += sound

    # Export combined audio to a byte stream
    combined_audio_format = 'mp3'  # Change format as needed
    combined_audio_io = io.BytesIO()
    combined.export(combined_audio_io, format=combined_audio_format)
    combined_audio_io.seek(0)  # Reset pointer to the beginning of the byte stream

    return combined_audio_io.read(), combined_audio_format

def generate_combined_filename(audio_files, file_format):
    # Extracting the base names (without extensions) and concatenating
    base_names = [os.path.splitext(os.path.basename(file.name))[0] for file in audio_files]
    combined_name = '__'.join(base_names)

    # Limit the length of the filename to a reasonable number (e.g., 255 characters)
    max_length = 255 - len(file_format) - 1  # accounting for file extension and dot
    if len(combined_name) > max_length:
        combined_name = combined_name[:max_length]

    return f"{combined_name}.{file_format}"

def milliseconds_until_sound(sound, silence_threshold_in_decibels=-20.0, chunk_size=10):
    trim_ms = 0  # ms
    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold_in_decibels and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms

def trim_start(filepath):
    # Determine the format of the audio file
    file_extension = os.path.splitext(filepath)[1].lower()
    if file_extension not in ['.wav', '.mp3']:
        raise ValueError("Unsupported audio format. Only WAV and MP3 are supported.")

    audio_format = 'wav' if file_extension == '.wav' else 'mp3'
    audio = AudioSegment.from_file(filepath, format=audio_format)
    start_trim = milliseconds_until_sound(audio)
    trimmed = audio[start_trim:]

    # Create a temporary file for the trimmed audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as temp_audio:
        trimmed.export(temp_audio.name, format="wav")
        return temp_audio.name

def segment_audio(audio, segment_length_ms=60000):
    start_time = 0
    segments = []

    while start_time < len(audio):
        segment = audio[start_time:start_time + segment_length_ms]
        segments.append(segment)
        start_time += segment_length_ms

    return segments

client = OpenAI()

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            language="en",
            # prompt="1st Energy, 1st Saver, cooling-off period, NMI, MIRN, RACT",
            # temperature=0,
            response_format="text"
        )
    return transcript

def clean_transcription(text):
    phrases = re.split(r'(?<=[.!?]) +', text)
    cleaned_phrases = [phrases[0]]
    for i in range(1, len(phrases)):
        if phrases[i].lower() != phrases[i-1].lower():
            cleaned_phrases.append(phrases[i])
    cleaned_text = ' '.join(cleaned_phrases)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def perform_evaluation(user, audio_file_ids, scorecard_id, evaluation):
    evaluations = []
    
    audio_files = AudioFile.objects.filter(id__in=audio_file_ids)
    
    for audio_file in audio_files:
        evaluator = ScorecardEvaluator(scorecard_id, audio_file.id)

        evaluations.append({
            "audio_file_id": audio_file.id,
            "responses": evaluator.run()
        })
        print("finished evaluating audio file with id: ", audio_file.id)

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
        self.audio_file_object = AudioFile.objects.get(id=audio_file_id)
        self.transcript = ""
        self.questions_and_options = ""
        
    def transcribe(self):
        if not self.audio_file_object.transcription:
            print("Not found in cache, transcribing.....")

            # Trim the original audio file
            trimmed_audio_path = trim_start(self.audio_file_object.audio.path)
            audio_format = os.path.splitext(self.audio_file_object.audio.path)[1].lower()[1:]  # 'wav' or 'mp3'
            trimmed_audio = AudioSegment.from_file(trimmed_audio_path, format=audio_format)

            # Segment the trimmed audio into a temporary directory
            one_minute = 60 * 1000  # Duration for each segment in milliseconds
            start_time = 0
            i = 0
            temp_segments_dir = "temp_segments_directory"  # Temporary directory for audio segments

            if not os.path.isdir(temp_segments_dir):
                os.makedirs(temp_segments_dir)

            while start_time < len(trimmed_audio):
                segment = trimmed_audio[start_time:start_time + one_minute]
                segment_filename = f"temp_segment_{i:02d}.{audio_format}"
                segment_path = os.path.join(temp_segments_dir, segment_filename)
                segment.export(segment_path, format=audio_format)
                start_time += one_minute
                i += 1

            # Transcribe each audio segment
            audio_files = sorted(
                [f for f in os.listdir(temp_segments_dir) if f.endswith(f".{audio_format}")],
                key=lambda f: int(''.join(filter(str.isdigit, f)))
            )
            transcriptions = [transcribe_audio(os.path.join(temp_segments_dir, file)) for file in audio_files]

            # Concatenate and save the transcription
            full_transcript = clean_transcription(' '.join(transcriptions))
            self.audio_file_object.transcription = full_transcript
            self.audio_file_object.save()

            # Clean up: Remove the temporary audio segment files
            os.remove(trimmed_audio_path)
            for file in audio_files:
                os.remove(os.path.join(temp_segments_dir, file))

        self.transcript = self.audio_file_object.transcription
        return self.transcript

    def construct_prompt(self):
        text = ""
        for question in self.questions:
            text += f"{question['text']} Options: {' or '.join(question['options'])}\n"
        
        for i, question in enumerate(self.questions):
            text += f"{i+1}. {question['text']} Options: {' or '.join(question['options'])}\n"
        self.questions_and_options = text.strip()
        return self.questions_and_options
    
    def evaluate(self):
        schema_string = [{"question":"string","options":["string"],"llm_response":"string","reason":"string"}]
         
        sys_prompt = (
            f"You are a Quality Assurance Analyst who is tasked evaluate the transcript "
            f"based on the following questions and choose a given option with proper reasoning. \n"
            f"{self.questions_and_options}\n"
            f"Your Output should be in JSON with the keys being "
            f"question(''), options([]), llm_response('') and reason('')\n"
            f"In the following JSON Schema: for every question:"
            f"{{'scorecard': {schema_string}}}"
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
        
        print(response.prompt_feedback)
        evaluation_results = response_to_dict(response.text)
        detailed_responses = []
        total_score = 0

        for question, ai_response in zip(self.questions, evaluation_results.get('scorecard', [])):
            correct = ai_response['llm_response'] in question['correct']
            if correct:
                total_score += question['score']
            detailed_responses.append({
                "question": question['text'],
                "options": question['options'],
                "llm_response": ai_response['llm_response'],
                "reason": ai_response.get('reason', ''),
                "correct": correct,
                "question_score": question['score']
            })

        evaluation_dict = {
            "score": total_score,
            "responses": detailed_responses
        }

        return evaluation_dict
    
    def qa_comment(self):
        schema_string = [{"name": "string", "dob": "string", "contactnumber": "string", "email": "string", "postaladdress": "string", "summary": "string", "comment": {"strength": "string", "improvement": "string"}}]
        prompt = (
            f"Extract the following data from the transcript: Name, Date Of Birth(DD/MM/YYYY), Contact Number, Email, Postal Address, Contact Number\n"
            f"Summary: Write a short summary of the call covering all important aspects\n"
            f"Comment: Provide coaching tips on how the agent improve? Specifically, for insights on areas like communication clarity, empathy, problem-solving efficiency, and handling difficult situations. Also Highlight areas where the Agent's performance is strong and effective.\n"
            f"If not captured in transcript then the value should be 'Not Found'\n"
            f"Your Output should be in JSON with the keys being "
            f"name(''), dob(''), contactnumber(''), email(''), postaladdress(''), summary(''), and comment({{}}).\n"
            f"In the following JSON Schema: for every question:"
            f"{{'qa': {schema_string}}}"
            f"Here is the transcript:\n"
            f"{self.transcript}\n"
        )
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
        return response_to_dict(response.text)
    
    def run(self):
        self.transcribe()
        self.construct_prompt()
        evaluation_dict = self.evaluate()
        qa_dict = self.qa_comment()
        return {**evaluation_dict, **qa_dict}

def response_to_dict(response_text):
    
    formatted_text = re.sub(r'^```JSON\n|```json\n|```$', '', response_text, flags=re.MULTILINE)
    response_dict = json.loads(formatted_text)
    return response_dict

def transcript_postprocessing(transcript):
    transcript = transcript.replace("\n", " ")
    
    prompt = (
        f"You are an intelligent assistant specializing in customer support calls"
        f"your task is to process transcripts of earnings calls, ensuring that all references to"
    )
    
    
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