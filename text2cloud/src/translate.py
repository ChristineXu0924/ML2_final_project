"""Translation module using pretrained Seq2Seq transformers."""

from pathlib import Path
import pandas as pd
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load config
config_path = Path(__file__).parent.parent / "config" / "project_config.yaml"
with open(config_path, "r") as file_handle:
    config = yaml.safe_load(file_handle)

# Load translation model parameters from config
model_name = config["models"]["translation"]["model"]
src_lang = config["models"]["translation"]["src_lang"]
tgt_lang_code = config["models"]["translation"]["tgt_lang"]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Get token ID for target language
tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

# Translation function
def translate_to_chinese(text, max_length=2024):
    """Translate English text to Chinese using the configured translation model."""
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return ""

    # Ensure the text ends with a period for proper sentence splitting
    if not text.strip().endswith("."):
        text += "."

    tokenizer.src_lang = src_lang
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    translated_sentences = []

    for sentence in sentences:
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2
        )
        translated = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        translated = translated.rstrip("，,、。；;")
        translated_sentences.append(translated)

    return " ".join(translated_sentences)
