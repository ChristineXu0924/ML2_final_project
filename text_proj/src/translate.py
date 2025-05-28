import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download('punkt')

model_name = "facebook/nllb-200-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer.src_lang = "eng_Latn"
tgt_lang_code = "zho_Hans"
tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

def translate_to_chinese(text, max_length=2024):
    if not text or not isinstance(text, str) or text.strip() == "":
        return ""
    sentences = nltk.sent_tokenize(text)
    translated = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2
        )
        translated.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return " ".join(translated)
