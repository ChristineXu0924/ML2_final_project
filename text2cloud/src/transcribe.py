"""Audio transcription module using Wav2Vec2 and Whisper models."""

import torch
import soundfile as sf
import librosa

def transcribe_wav2vec(audio_path: str, processor, model, device) -> str:
    """
    Transcribe audio using Wav2Vec2 model.

    Args:
        audio_path (str): Path to the audio file.
        processor: Wav2Vec2 processor.
        model: Wav2Vec2 model.
        device: Device (CPU or GPU).
    """
    try:
        speech_array, sampling_rate = sf.read(audio_path)

        # Resample if needed
        if sampling_rate != 16000:
            speech_array = librosa.resample(
                speech_array, orig_sr=sampling_rate, target_sr=16000
            )

        input_values = processor(
            speech_array, return_tensors="pt", sampling_rate=16000
        ).input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.decode(predicted_ids[0])

    except Exception as err:
        return f"Error in Wav2Vec2 transcription: {err}"

def transcribe_whisper(audio_path: str, model) -> str:
    """
    Transcribe audio using Whisper model.

    Args:
        audio_path (str): Path to the audio file.
        model: Whisper model.
    """
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as err:
        return f"Error in Whisper transcription: {err}"
