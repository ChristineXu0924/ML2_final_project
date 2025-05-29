"""Streamlit app for audio transcription, summarization, NER, and translation."""

import sys
import tempfile
from pathlib import Path

import streamlit as st
import spacy
import torch
import whisper
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForCTC

# Ensure src folder is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.transcribe_clean import transcribe_wav2vec, transcribe_whisper
from src.translate import translate_to_chinese

# Initialize session state
if "model_locked" not in st.session_state:
    st.session_state.model_locked = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = ""

@st.cache_resource
def get_config():
    """Load YAML configuration."""
    return load_config()

config = get_config()

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
st.title("🎙️ Audio Summarizer and NER Extractor")

# Block until model is selected
if not st.session_state.selected_model:
    st.warning("Please select a transcription model from the sidebar before uploading audio.")
else:
    uploaded_file = st.file_uploader("Upload an audio file (.wav or .flac)", type=["wav", "flac"], disabled=not st.session_state.selected_model)

    if uploaded_file is not None:
        st.session_state.model_locked = True

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.audio(uploaded_file)
        st.write("Transcribing with", st.session_state.selected_model, "...")
        
        def transcribe_audio_func(audio_path, models_dict):
            """Transcribe audio using selected method."""
            method = st.session_state.selected_model
            if method == "Wav2Vec2":
                processor = models_dict["wav2vec"]["processor"]
                model = models_dict["wav2vec"]["model"]
                device = models_dict["wav2vec"]["device"]
                return transcribe_wav2vec(audio_path, processor, model, device)
            return transcribe_whisper(audio_path, models_dict["whisper"])

        transcript = transcribe_audio_func(tmp_path, models)

        st.subheader("Transcript")
        st.write(transcript)

        st.subheader("Named Entities")
        doc = models["ner"](transcript)
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
