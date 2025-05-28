# find_length.py
import pandas as pd
from typing import Tuple

TEXT_COL = "whisper_transcript"

def add_word_count_column(df: pd.DataFrame, text_col: str = TEXT_COL) -> pd.DataFrame:
    """Adds a 'word_count' column to the DataFrame based on a given text column."""
    df = df.copy()
    df["word_count"] = df[text_col].apply(lambda x: len(str(x).split()))
    return df

def mark_long_transcripts(df: pd.DataFrame, word_threshold: float = None) -> Tuple[pd.DataFrame, float]:
    """
    Adds an 'is_long' column to the DataFrame based on the 75th percentile of word count
    or a provided threshold.
    Returns the modified DataFrame and the threshold used.
    """
    if "word_count" not in df.columns:
        df = add_word_count_column(df)

    if word_threshold is None:
        word_threshold = df["word_count"].quantile(0.75)

    df["is_long"] = df["word_count"] > word_threshold
    return df, word_threshold

# Example usage
if __name__ == "__main__":
    path = "../data/whisper_with_groundtruth_100.csv"
    df = pd.read_csv(path)
    df = add_word_count_column(df)
    df, threshold = mark_long_transcripts(df)
    print(f"75th percentile word count threshold: {threshold:.0f} words")
    print(df[["file_path", "word_count", "is_long"]].head())