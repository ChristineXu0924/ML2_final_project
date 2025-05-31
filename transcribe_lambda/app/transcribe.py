from typing import Any
import torch
import torchaudio
import torchaudio.transforms as T


def transcribe_whisper(audio_path: str, model) -> str:
    """
    Transcribe audio using Whisper model.

    Args:
        audio_path (str): Path to the audio file.
        model: Whisper model.

    Returns:
        str: Transcribed text.
    """
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as err:
        return f"Error in Whisper transcription: {err}"


def transcribe_wav2vec(audio_path: str, processor, model, device: str) -> str:
    """
    Transcribe audio using Wav2Vec2 model.

    Args:
        audio_path (str): Path to the audio file.
        processor: Wav2Vec2 processor.
        model: Wav2Vec2 model.
        device (str): 'cpu' or 'cuda'.

    Returns:
        str: Transcribed text.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)  # waveform: [channel, time]

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        target_sr = 16000
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        speech_array = waveform.squeeze().numpy()  # Convert to numpy array

        input_values = processor(
            speech_array, return_tensors="pt", sampling_rate=target_sr
        ).input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.decode(predicted_ids[0])

    except Exception as err:
        return f"Error in Wav2Vec2 transcription: {err}"
