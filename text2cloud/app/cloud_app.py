"""Streamlit app for audio transcription, summarization, NER, and translation."""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st
import spacy
import torch
import whisper
import boto3
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForCTC
import random

# Ensure src folder is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.transcribe import transcribe_wav2vec, transcribe_whisper
from src.translate import translate_to_chinese
from src.s3_utils import list_audio_files, load_ner_model_from_s3, trigger_lambda, upload_file


# Load AWS environment variables
env_path = Path(__file__).resolve().parent.parent / "config" / "secrets.env"
load_dotenv(dotenv_path=env_path)

BUCKET = os.getenv("BUCKET_NAME").strip()
PREFIX_DATA = os.getenv("BUCKET_PREFIX_DATA").strip()
PREFIX_MODEL = os.getenv("BUCKET_PREFIX_MODEL").strip()
LAMBDA_FUNC = os.getenv("LAMBDA_FUNCTION_NAME").strip()
s3 = boto3.client("s3", region_name="us-east-1")

# Initialize session statse
if "model_locked" not in st.session_state:
    st.session_state.model_locked = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""

@st.cache_resource
def get_config():
    """Load YAML configuration."""
    return load_config()

config = get_config()

@st.cache_data
def download_from_s3(bucket, s3_key):
    """Download a file from S3 to a temporary location."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(s3_key).suffix)
    s3.download_file(bucket, s3_key, tmp.name)
    return tmp.name

@st.cache_resource
def load_models():
    """Load models and resources."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    processor = Wav2Vec2Processor.from_pretrained(
        config["models"]["transcription"]["wave_processor"]
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        config["models"]["transcription"]["wave_model"]
    ).to(device)

    return {
        "wav2vec": {
            "processor": processor,
            "model": model,
            "device": device,
        },
        "ner": spacy.load(str((Path(__file__).parent.parent / config["models"]["ner_model"]).resolve())),
        "sum": pipeline("summarization", model=config["models"]["summarization"]["sum_model"]),
        "whisper": whisper.load_model(config["models"]["transcription"]["whisper_model"]),
    }

models = load_models()

# Sidebar for model selection
st.sidebar.title("Model Selection")
st.sidebar.markdown(
    """
### Transcription Model Recommendations:
- **Whisper**: Produces text with proper punctuation and capitalization. Better for general use.
- **Wav2Vec2**: Better performance on LibriSpeech-like audio but lacks punctuation and proper capitalization.
"""
)

# Model selection and locking
if not st.session_state.model_locked:
    selected = st.sidebar.selectbox(
        "Choose transcription method", ["", "Whisper", "Wav2Vec2"], index=0
    )
    st.session_state.selected_model = selected
    if selected:
        st.sidebar.success(f"✅ Done: {selected} selected")
else:
    st.sidebar.info(f"🔒 Model locked: {st.session_state.selected_model}")

if st.sidebar.button("🔄 Reset Model Selection"):
    st.session_state.model_locked = False
    st.session_state.selected_model = ""
    st.rerun()

# App title
st.title("🎙️ Speech-to-Text and Audio Insights via AWS Lambda, S3 & ECS")

# Block until model is selected
if not st.session_state.selected_model:
    st.warning("Please select a transcription model from the sidebar before uploading audio.")
else:
    st.session_state.model_locked = True

    input_method = st.radio("Choose input method", ["Upload Audio File", "Select Test File from S3"])

    uploaded_file = None
    selected_file = None
    tmp_path = None

    if input_method == "Upload Audio File":
        uploaded_file = st.file_uploader("Upload a .wav or .flac audio file", type=["wav", "flac"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            s3_key = f"streamlit_inputs/{Path(tmp_path).name}"
            s3.upload_file(tmp_path, BUCKET, s3_key)
            st.audio(uploaded_file)

    elif input_method == "Select Test File from S3":
        test_files = list_audio_files(BUCKET, PREFIX_DATA)
        test_files = random.sample(test_files, 10)  # Randomly select 5 files
        selected_file = st.selectbox("Select a test file", test_files)
        if selected_file:
            tmp_path = download_from_s3(BUCKET, selected_file)
            s3_key = selected_file
            st.audio(tmp_path)
    
    # Part 2: Transcribe audio, generate summaries, NER, and translation
    if tmp_path:
        st.write("Invoking Lambda and transcribing with ", st.session_state.selected_model, "...")
        
        lambda_response = trigger_lambda(BUCKET, s3_key, st.session_state.selected_model, LAMBDA_FUNC)
        st.write("Lambda response:", lambda_response)
        transcript = lambda_response.get("transcript", "")

        st.subheader("Transcript")
        st.write(transcript)

        st.subheader("Named Entities")

        # Load NER model from S3
        cloud_ner_model, temp_dir = load_ner_model_from_s3(BUCKET, PREFIX_MODEL)
        doc = cloud_ner_model(transcript)

        entities = [(ent.text, ent.label_) for ent in doc.ents]
        st.write(entities if entities else "No named entities found.")

        word_count = len(transcript.split())

        try:
            if word_count < 42:
                st.subheader("The audio is too short. Generating tiny summary")
                tiny = models["sum"](
                    transcript,
                    min_length=config["settings"]["min_length_tiny"],
                    max_length=config["settings"]["max_length_tiny"],
                    do_sample=False,
                )[0]["summary_text"]
                st.write(tiny)
            else:
                st.subheader("Long Summary")
                long = models["sum"](
                    transcript,
                    min_length=config["settings"]["min_length_large"],
                    max_length=config["settings"]["max_length_large"],
                    do_sample=False,
                )[0]["summary_text"]
                st.write(long)

                st.subheader("Short Summary")
                short = models["sum"](
                    transcript,
                    min_length=config["settings"]["min_length_small"],
                    max_length=config["settings"]["max_length_small"],
                    do_sample=False,
                )[0]["summary_text"]
                st.write(short)

                st.subheader("Tiny Summary")
                tiny = models["sum"](
                    transcript,
                    min_length=config["settings"]["min_length_tiny"],
                    max_length=config["settings"]["max_length_tiny"],
                    do_sample=False,
                )[0]["summary_text"]
                st.write(tiny)

        except Exception as e:  # pylint: disable=broad-exception-caught
            st.warning(f"Summary generation error: {e}")

        st.subheader("Chinese Translation (via NLLB-200)")
        try:
            translation = translate_to_chinese(transcript)
            st.code(translation, language="zh")
        except Exception as e:  # pylint: disable=broad-exception-caught
            st.warning(f"Translation error: {e}")

        Path(tmp_path).unlink(missing_ok=True)
