# ner_module.py
import spacy
import pandas as pd
from typing import List, Tuple

# Ensure model is downloaded
try:
    spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from a given text."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def annotate_dataframe_with_ner(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Adds a 'named_entities' column to the DataFrame based on the specified text column."""
    df = df.copy()
    df["named_entities"] = df[text_column].apply(lambda x: extract_named_entities(str(x)))
    return df

def load_and_process_csv(path: str, text_column: str = "whisper_transcript") -> pd.DataFrame:
    """Loads a CSV and applies NER annotation to it."""
    df = pd.read_csv(path)
    return annotate_dataframe_with_ner(df, text_column)

# Example usage (can be removed or guarded with __name__ == '__main__')
if __name__ == "__main__":
    file_path = "../data/whisper_with_groundtruth_100.csv"
    df_with_ner = load_and_process_csv(file_path)
    print(df_with_ner.head())
