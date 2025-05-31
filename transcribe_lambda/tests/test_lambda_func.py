import os
import tempfile
import pytest
import boto3
from moto import mock_aws
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.lambda_function import lambda_handler


@pytest.fixture
def s3_bucket_with_audio():
    # Set up a fake S3 bucket with a dummy .wav file
    bucket = "test-bucket"
    key = "test_audio.wav"
    content = b"\x00\x00\x00\x00"  # fake binary data

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(content)
            f.flush()
            s3.upload_file(f.name, bucket, key)

        yield {
            "bucket": bucket,
            "key": key,
        }


def test_lambda_transcribes_whisper(monkeypatch, s3_bucket_with_audio):
    # Monkeypatch the whisper model to return dummy output
    class DummyWhisper:
        def transcribe(self, path):
            return {"text": "hello from dummy whisper"}

    monkeypatch.setattr("whisper.load_model", lambda _: DummyWhisper())

    event = {
        "bucket": s3_bucket_with_audio["bucket"],
        "key": s3_bucket_with_audio["key"],
        "model": "Whisper"
    }

    result = lambda_handler(event, context={})

    assert result["status"] == "success"
    assert result["transcript"] == "hello from dummy whisper"
    assert result["model_used"] == "Whisper"


def test_lambda_transcribes_wav2vec2(monkeypatch, s3_bucket_with_audio):
    # Dummy model that mocks .to() and returns self
    class DummyModel:
        def to(self, device):
            return self

    # Dummy transcribe function
    def dummy_transcribe_wav2vec(audio_path, processor, model, device):
        return "hello from dummy wav2vec2"

    # Patch the necessary components
    monkeypatch.setattr("app.lambda_function.transcribe_wav2vec", dummy_transcribe_wav2vec)
    monkeypatch.setattr("app.lambda_function.Wav2Vec2Processor.from_pretrained", lambda _: "dummy_processor")
    monkeypatch.setattr("app.lambda_function.Wav2Vec2ForCTC.from_pretrained", lambda _: DummyModel())

    event = {
        "bucket": s3_bucket_with_audio["bucket"],
        "key": s3_bucket_with_audio["key"],
        "model": "Wav2Vec2"
    }

    result = lambda_handler(event, context={})

    assert result["status"] == "success"
    assert result["transcript"] == "hello from dummy wav2vec2"
    assert result["model_used"] == "Wav2Vec2"

