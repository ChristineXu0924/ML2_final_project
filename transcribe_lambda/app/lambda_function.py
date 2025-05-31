import sys
import os
import tempfile
print("LOADED MODULES:", sys.modules.keys())

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import whisper
from transcribe import transcribe_whisper, transcribe_wav2vec
import tempfile
import boto3
from pathlib import Path

print("Lambda handler loaded successfully.")

s3 = boto3.client("s3", region_name="us-east-1")
os.environ["XDG_CACHE_HOME"] = "/tmp"
os.environ["TRANSFORMERS_CACHE"] = "/tmp"
os.environ["HF_HOME"] = "/tmp"

def lambda_handler(event: dict, context: dict) -> dict:
    """
    AWS Lambda function handler to transcribe audio files using either Whisper or Wav2Vec2 models.
    
    Args:
        event (dict): Event data containing S3 bucket and key for the audio file, and model choice.
        context (dict): Context object provided by AWS Lambda.
        
    Returns:
        dict: Transcription result and model used.
    """
    bucket = event["bucket"]
    key = event["key"]
    model_choice = event.get("model", "Whisper")  # Default to Whisper

    # Download audio
    suffix = Path(key).suffix
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    s3.download_file(bucket, key, tmp_audio.name)

    # Transcribe based on selected model
    if model_choice == "Wav2Vec2":
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/tmp")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir="/tmp").to("cpu")
        transcript = transcribe_wav2vec(tmp_audio.name, processor, model, device="cpu")
    else:
        model = whisper.load_model("base", download_root="/tmp")
        transcript = transcribe_whisper(tmp_audio.name, model)

    return {
        "status": "success",
        "model_used": model_choice,
        "transcript": transcript
    }

