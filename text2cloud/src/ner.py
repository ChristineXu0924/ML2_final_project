"""Named Entity Recognition (NER) module using spaCy."""

from typing import List, Tuple
from pathlib import Path
import spacy
import pandas as pd

# Define path to your custom NER model
MODEL_PATH = Path(__file__).resolve().parent.parent / "ner_custom_model"

# Load the NER model
try:
    nlp = spacy.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load spaCy model from '{MODEL_PATH}': {e}") from e

def extract_named_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from a given text string."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def annotate_dataframe_with_ner(dataframe: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Annotate a DataFrame with a new column 'named_entities',
    extracting entities from the specified text column.
    """
    dataframe = dataframe.copy()
    dataframe["named_entities"] = dataframe[text_column].apply(lambda x: extract_named_entities(str(x)))
    return dataframe

def load_and_process_csv(path: str, text_column: str = "whisper_transcript") -> pd.DataFrame:
    """
    Load a CSV file and apply named entity recognition (NER) to the specified text column.
    Returns a DataFrame with an additional 'named_entities' column.
    """
    dataframe = pd.read_csv(path)
    return annotate_dataframe_with_ner(dataframe, text_column)

if __name__ == "__main__":
    # Example usage for testing
    FILE_PATH = "../data/whisper_with_groundtruth_100.csv"
    dataframe_with_ner = load_and_process_csv(FILE_PATH)
    print(dataframe_with_ner.head())
