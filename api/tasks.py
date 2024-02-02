from .models import (
    Evaluation,
    EvaluationJob,
    AudioFile,
    Scorecard,
    Utterance,
    Transcript,
)
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone
from django.conf import settings
from django.template.loader import render_to_string
from weasyprint import HTML
import json
from concurrent.futures import ThreadPoolExecutor
import re
import os
import io
from pydub import AudioSegment
import assemblyai as aai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def combine_audio(audio_files):
    combined = AudioSegment.empty()
    for file in audio_files:
        file_content = file.read()
        sound = AudioSegment.from_file(io.BytesIO(file_content))
        combined += sound

    # Export combined audio to a byte stream
    combined_audio_format = "mp3"  # Change format as needed
    combined_audio_io = io.BytesIO()
    combined.export(combined_audio_io, format=combined_audio_format)
    combined_audio_io.seek(0)  # Reset pointer to the beginning of the byte stream

    return combined_audio_io.read(), combined_audio_format


def generate_combined_filename(audio_files, file_format):
    # Extracting the base names (without extensions) and concatenating
    base_names = [
        os.path.splitext(os.path.basename(file.name))[0] for file in audio_files
    ]
    combined_name = "__".join(base_names)

    # Limit the length of the filename to a reasonable number (e.g., 255 characters)
    max_length = 255 - len(file_format) - 1  # accounting for file extension and dot
    if len(combined_name) > max_length:
        combined_name = combined_name[:max_length]

    return f"{combined_name}.{file_format}"


def get_context(user, query):
    if not user.knowledge_base or not user.knowledge_base.pdf:
        return None

    try:
        loader = PyMuPDFLoader(user.knowledge_base.pdf.path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=125, chunk_overlap=25)
        splitted_documents = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(
            splitted_documents,
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        )
        retriever = vectorstore.as_retriever()
        relevant_documents = retriever.get_relevant_documents(query)
        extracted_content = [doc.page_content for doc in relevant_documents]
        return ",".join(extracted_content)
    except Exception as e:  # replace with the actual expected exception
        print(f"An error occurred while getting context: {e}")
        return None


def transcribe(audio_file_object):
    if audio_file_object.transcription is None:
        FILE_URL = audio_file_object.audio.path
        config = aai.TranscriptionConfig(speaker_labels=True).set_redact_pii(
            policies=[
                aai.PIIRedactionPolicy.credit_card_number,
                aai.PIIRedactionPolicy.credit_card_expiration,
                aai.PIIRedactionPolicy.credit_card_cvv,
            ],
            redact_audio=True,
        )

        transcriber = aai.Transcriber()
        transcript_data = transcriber.transcribe(FILE_URL, config=config)

        # Create a new Transcript instance
        transcript_instance = Transcript.objects.create(
            audio_file=audio_file_object, text=transcript_data.text
        )

        LOW_CONFIDENCE_THRESHOLD = 0.8

        # Populate Utterance models
        for utterance in transcript_data.utterances:
            low_conf_words = {}

            for word in utterance.words:
                # Check if the confidence is below the threshold
                if word.confidence < LOW_CONFIDENCE_THRESHOLD:
                    low_conf_words[word.text] = {
                        "confidence": word.confidence,
                        "start": word.start,
                        "end": word.end,
                    }
            Utterance.objects.create(
                transcript=transcript_instance,
                speaker_label=utterance.speaker,
                start_time=utterance.start,
                end_time=utterance.end,
                confidence=utterance.confidence,
                text=utterance.text,
                low_confidence_words=low_conf_words,
            )

        # Update the AudioFile object to link the Transcript
        audio_file_object.transcription = transcript_instance
        audio_file_object.save()

        return transcript_data.text
    else:
        return audio_file_object.transcription.text


# Preserve legacy perform_evaluation
#
# def perform_evaluation(user, audio_file_ids, scorecard_id, evaluation):
#     evaluations = []

#     audio_files = AudioFile.objects.filter(id__in=audio_file_ids)

#     try:
#         for audio_file in audio_files:
#             evaluator = ScorecardEvaluator(scorecard_id, audio_file.id)

#             evaluations.append(
#                 {"audio_file_id": audio_file.id, "responses": evaluator.run()}
#             )
#             print("finished evaluating audio file with id: ", audio_file.id)

#         # Update the evaluation with the final result
#         final_result = {"status": "completed", "evaluations": evaluations}
#         Evaluation.objects.filter(id=evaluation.id).update(
#             result=final_result, completed_at=timezone.now()
#         )
#     except Exception as e:
#         print(f"An error occurred during evaluation: {e}")
#         Evaluation.objects.filter(id=evaluation.id).delete()


def perform_single_evaluation(evaluation_job, scorecard, audio_file):
    try:
        evaluator = ScorecardEvaluator(scorecard.id, audio_file.id)
        evaluation_result = evaluator.run()

        # Create a new Evaluation instance for each audio file
        # Use Django's transaction.atomic to ensure thread safety on database operations
        with transaction.atomic():
            Evaluation.objects.create(
                evaluation_job=evaluation_job,
                audio_file=audio_file,
                scorecard=scorecard,
                result=evaluation_result,
                status=Evaluation.StatusChoices.COMPLETED,
            )
        print(f"Finished evaluating audio file with id: {audio_file.id}")
    except Exception as e:
        print(f"An error occurred during evaluation of file {audio_file.id}: {e}")


def perform_evaluation(evaluation_job_id, scorecard_id):
    try:
        evaluation_job = EvaluationJob.objects.get(id=evaluation_job_id)
        scorecard = Scorecard.objects.get(id=scorecard_id)
        audio_files = evaluation_job.audio_files.all()

        with ThreadPoolExecutor(max_workers=5) as executor:
            for audio_file in audio_files:
                executor.submit(
                    perform_single_evaluation, evaluation_job, scorecard, audio_file
                )

        # Update the EvaluationJob status
        evaluation_job.status = EvaluationJob.StatusChoices.COMPLETED
        evaluation_job.completed_at = timezone.now()
        evaluation_job.save()

    except Exception as e:
        print(f"An error occurred during the evaluation job: {e}")
        evaluation_job.status = EvaluationJob.StatusChoices.FAILED
        evaluation_job.save()


class ScorecardEvaluator:
    def __init__(self, scorecard_id, audio_file_id):
        self.scorecard = Scorecard.objects.get(id=scorecard_id)
        self.questions = self.scorecard.questions
        self.audio_file_object = AudioFile.objects.get(id=audio_file_id)
        self.transcript = ""
        self.questions_and_options = ""

    def transcribe(self):
        self.transcript = transcribe(self.audio_file_object)
        return self.transcript

    def construct_prompt(self):
        text = ""

        for i, question in enumerate(self.questions):
            question_prompt = ""
            question_prompt += (
                f"{i+1}. {question['text']} Options: "
                f"{' or '.join(question['options'])}"
            )

            if question.get("use_knowledge_base", False):
                # Specific instruction or context for using the knowledge base
                # Replace 'Specific Context/Instruction' with actual content as needed
                context = get_context(self.scorecard.user, question["text"])
                if context:
                    question_prompt += (
                        f"[For this question specifically, "
                        f"use the following additional "
                        f"context: ####{context}####]"
                    )

            text += question_prompt + "\n"
        self.questions_and_options = text.strip()
        return self.questions_and_options

    def evaluate(self):
        schema_string = [
            {
                "question": "string",
                "options": ["string"],
                "llm_response": "string",
                "reason": "string",
            }
        ]
        sys_prompt = (
            f"You are a Quality Assurance Analyst who is tasked evaluate the transcript "  # noqa: E501
            f"based on the following questions and choose a given option with proper reasoning. \n"  # noqa: E501
            f"{self.questions_and_options}\n"
            f"Your Output should be in JSON with the keys being "
            f"question(''), options([]), llm_response('') and reason('')\n"
            f"In the following JSON Schema: for every question:"
            f"{{'scorecard': {schema_string}}}"
            f"With the question and options being the original ones provided and llm_response "  # noqa: E501
            f"being the option you chose and reason being the reason you chose that option\n"  # noqa: E501
        )
        user_prompt = f"Here is the transcript: \n{self.transcript}"
        prompt = f"{sys_prompt}\n{user_prompt}"

        # messages = [{"role": "user", "parts": [prompt]}]

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.0,
            max_output_tokens=8192,
        )
        response = llm.invoke(prompt)
        evaluation_results = response_to_dict(response.content)
        detailed_responses = []
        total_score = 0

        for question, ai_response in zip(
            self.questions, evaluation_results.get("scorecard", [])
        ):
            correct = ai_response["llm_response"] in question["correct"]

            if correct:
                total_score += question["score"]
            detailed_responses.append(
                {
                    "question": question["text"],
                    "options": question["options"],
                    "llm_response": ai_response["llm_response"],
                    "reason": ai_response.get("reason", ""),
                    "correct": correct,
                    "question_score": question["score"],
                }
            )

        evaluation_dict = {"score": total_score, "responses": detailed_responses}

        return evaluation_dict

    def qa_comment(self):
        schema_string = [
            {
                "name": "string",
                "dob": "string",
                "contactnumber": "string",
                "email": "string",
                "postaladdress": "string",
                "summary": "string",
                "comment": {"strength": "string", "improvement": "string"},
            }
        ]
        prompt = (
            f"Extract the following data from the transcript: Name, Date Of Birth(DD/MM/YYYY), Contact Number, Email, Postal Address, Contact Number\n"  # noqa: E501
            f"Summary: Write a short summary of the call covering all important aspects\n"  # noqa: E501
            f"Comment: Provide coaching tips on how the agent improve? Specifically, for insights on areas like communication clarity, empathy, problem-solving efficiency, and handling difficult situations. Also Highlight areas where the Agent's performance is strong and effective.\n"  # noqa: E501
            f"If not captured in transcript then the value should be 'Not Found'\n"
            f"Your Output should be in JSON with the keys being "
            f"name(''), dob(''), contactnumber(''), email(''), postaladdress(''), summary(''), and comment({{}}).\n"  # noqa: E501
            f"In the following JSON Schema:"
            f"{{'qa': {schema_string}}}"
            f"Here is the transcript:\n"
            f"{self.transcript}\n"
        )
        # messages = [{"role": "user", "parts": [prompt]}]

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.0,
            # max_output_tokens=8192,
        )

        response = llm.invoke(prompt)
        return response_to_dict(response.content)

    def run(self):
        self.transcribe()
        self.construct_prompt()
        evaluation_dict = self.evaluate()
        qa_dict = self.qa_comment()
        return {**evaluation_dict, **qa_dict}


def response_to_dict(response_text):
    formatted_text = re.sub(
        r"^```JSON\n|```json\n|```$", "", response_text, flags=re.MULTILINE
    )
    response_dict = json.loads(formatted_text)
    return response_dict


def generate_pdf_report(evaluation):
    html_string = render_to_string(
        "api/evaluation_report.html", {"evaluation": evaluation}
    )
    report_filename = f"evaluation_report_{evaluation.id}.pdf"
    report_path = os.path.join(
        settings.MEDIA_ROOT, f"evaluation_reports/{evaluation.id}/{report_filename}"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    HTML(string=html_string).write_pdf(report_path)
    return os.path.join(f"evaluation_reports/{evaluation.id}/", report_filename)


def generate_pdf_report_for_evaluation(evaluation):
    html_string = render_to_string(
        "api/evaluation_report.html", {"evaluation": evaluation}
    )
    report_filename = f"evaluation_report_{evaluation.id}.pdf"
    report_content = HTML(string=html_string).write_pdf()

    # Saving the PDF content to the pdf_report field
    evaluation.pdf_report.save(report_filename, ContentFile(report_content))
    evaluation.save()

    return evaluation.pdf_report.url  # Returns the URL to access this file
