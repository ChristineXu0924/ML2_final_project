# streamlit_app.py
# streamlit_app.py
import streamlit as st
import spacy
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForCTC
import tempfile
from pathlib import Path
import os
import yaml
import torch
import soundfile as sf
import librosa
from src.translate import translate_to_chinese


# Load configuration
@st.cache_resource
def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
config = load_config()  

# Load models
@st.cache_resource
def load_models():
    # Choose device (M1/M2/M3 = "mps", fallback to CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Wav2Vec2 model and processor from config
    processor = Wav2Vec2Processor.from_pretrained(config["models"]["transcription"]["wave_processor"])
    model = Wav2Vec2ForCTC.from_pretrained(config["models"]["transcription"]["wave_model"]).to(device)
    
    return {
        "wav2vec": {
            "processor": processor,
            "model": model,
            "device": device
        },
        "ner": spacy.load(config["ner_model"]),
        "tiny_sum": pipeline("summarization", model=config["models"]["summarization"]["tiny"]),
        "long_sum": pipeline("summarization", model=config["models"]["summarization"]["large"]),  
        "short_sum": pipeline("summarization", model=config["models"]["summarization"]["small"])
    }

models = load_models()

# Function to transcribe audio using Wav2Vec2
def transcribe_audio(audio_path, models):
    try:
        speech_array, sampling_rate = sf.read(audio_path)
        
        # Resample if needed
        if sampling_rate != 16000:
            speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
        
        # Process audio
        input_values = models["wav2vec"]["processor"](
            speech_array, 
            return_tensors="pt", 
            sampling_rate=16000
        ).input_values.to(models["wav2vec"]["device"])
        
        # Generate transcription
        with torch.no_grad():
            logits = models["wav2vec"]["model"](input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = models["wav2vec"]["processor"].decode(predicted_ids[0])
        
        return transcription
    except Exception as e:
        return f"Error transcribing audio: {e}"

# App UI
st.title("🎙️ Audio Summarizer and NER Extractor")
uploaded_file = st.file_uploader("Upload an audio file (.wav or .flac)", type=["wav", "flac"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(uploaded_file)
    st.write("Transcribing with Wav2Vec2...")
    transcript = transcribe_audio(tmp_path, models)

    st.subheader("Transcript")
    st.write(transcript)

    # Named Entity Recognition
    st.subheader("Named Entities")
    doc = models["ner"](transcript)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    st.write(ents if ents else "No named entities found.")

    # Determine text length
    word_count = len(transcript.split())

    # Summary
    if word_count < 42:
        st.subheader("The audio is too short. Generate tiny summary")
        try:
            tiny = models["tiny_sum"](transcript)[0]["summary_text"]
            st.write(tiny)
        except Exception as e:
            st.write(f"Error generating summary: {e}")
    else:
        st.subheader("Long Summary")
        long = models["long_sum"](transcript, 
                                 min_length=config["settings"]["min_length_large"], 
                                 max_length=config["settings"]["max_length_large"], 
                                 do_sample=False)[0]["summary_text"]
        st.write(long)

        st.subheader("Short Summary")
        short = models["short_sum"](transcript, 
                                   min_length=config["settings"]["min_length_small"], 
                                   max_length=config["settings"]["max_length_small"], 
                                   do_sample=False)[0]["summary_text"]
        st.write(short)

        st.subheader("Tiny Summary")
        tiny = models["tiny_sum"](transcript)[0]["summary_text"]
        st.write(tiny)

# Chinese Translation (at the end)
st.subheader("Chinese Translation (via NLLB-200)")
try:
    translation = translate_to_chinese(transcript)
    st.code(translation, language='zh')
except Exception as e:
    st.warning(f"Translation error: {e}")

    # Optional cleanup
    Path(tmp_path).unlink(missing_ok=True)
