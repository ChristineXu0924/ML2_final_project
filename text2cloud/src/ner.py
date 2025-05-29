import spacy
import pandas as pd
from typing import List, Tuple
from pathlib import Path

# Define path to your custom NER model
model_path = Path(__file__).resolve().parent.parent / "ner_custom_model"

# Load the NER model
try:
    nlp = spacy.load(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load spaCy model from '{model_path}': {e}")

def extract_named_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from a given text string."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def annotate_dataframe_with_ner(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Annotate a DataFrame with a new column 'named_entities',
    extracting entities from the specified text column.
    """
    df = df.copy()
    df["named_entities"] = df[text_column].apply(lambda x: extract_named_entities(str(x)))
    return df

def load_and_process_csv(path: str, text_column: str = "whisper_transcript") -> pd.DataFrame:
    """
    Load a CSV file and apply named entity recognition (NER) to the specified text column.
    Returns a DataFrame with an additional 'named_entities' column.
    """
    df = pd.read_csv(path)
    return annotate_dataframe_with_ner(df, text_column)

if __name__ == "__main__":
    # Example usage for testing
    file_path = "../data/whisper_with_groundtruth_100.csv"
    df_with_ner = load_and_process_csv(file_path)
    print(df_with_ner.head())
