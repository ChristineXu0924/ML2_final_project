# tests/test_pipeline.py

import pytest
import whisper
import spacy
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
from src.transcribe import transcribe_whisper, transcribe_wav2vec
from src.translate import translate_to_chinese

@pytest.fixture
def example_audio():
    return "tests/test.flac"

def test_pipeline(example_audio):
    # Transcribe with Wav2Vec2
    device = torch.device("cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    transcript_wav2vec = transcribe_wav2vec(example_audio, processor, wav2vec_model, device)
    assert isinstance(transcript_wav2vec, str)
    assert len(transcript_wav2vec.strip()) > 0
    print("\nWav2Vec2 transcript:", transcript_wav2vec)

    # Transcribe with Whisper
    whisper_model = whisper.load_model("base")
    transcript_whisper = transcribe_whisper(example_audio, whisper_model)
    assert isinstance(transcript_whisper, str)
    assert len(transcript_whisper.strip()) > 0
    print("\nWhisper transcript:", transcript_whisper)

    # Summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(transcript_whisper, min_length=5, max_length=20)[0]["summary_text"]
    assert isinstance(summary, str)
    assert len(summary.strip()) > 0
    print("\nSummary:", summary)

    # Named Entity Recognition using local model
    nlp = spacy.load("ner_custom_model")
    doc = nlp(transcript_whisper)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    assert isinstance(entities, list)
    print("\nNamed Entities:", entities)

    # Translation to Chinese
    translation = translate_to_chinese(transcript_whisper)
    assert isinstance(translation, str)
    assert len(translation.strip()) > 0
    print("\nChinese translation:", translation)
