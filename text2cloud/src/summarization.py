"""Summarization module using BART and PEGASUS transformers."""

import pandas as pd
from transformers import pipeline
from find_length import add_word_count_column, mark_long_transcripts

# Load summarization pipelines
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
pegasus_summarizer = pipeline("summarization", model="google/pegasus-xsum")

# Define Summary Functions
def generate_bart_summary(text, min_len=60, max_len=300):
    """Generate summary using BART model."""
    try:
        return bart_summarizer(
            text, min_length=min_len, max_length=max_len, do_sample=False
        )[0]["summary_text"]
    except Exception as err:
        return f"[ERROR: {err}]"

def generate_pegasus_summary(text):
    """Generate summary using PEGASUS model."""
    try:
        return pegasus_summarizer(text)[0]["summary_text"]
    except Exception as err:
        return f"[ERROR: {err}]"

# Main processing
if __name__ == "__main__":
    # Load and preprocess DataFrame
    PATH = "../data/whisper_with_groundtruth_100.csv"
    dataframe = pd.read_csv(PATH)
    dataframe = add_word_count_column(dataframe)
    dataframe, threshold = mark_long_transcripts(dataframe)

    # Filter long entries only
    long_dataframe = dataframe[dataframe["is_long"]].copy()

    # Apply summarization
    long_dataframe["long_summary"] = long_dataframe["whisper_transcript"].apply(
        lambda x: generate_bart_summary(x, min_len=100, max_len=300)
    )
    long_dataframe["short_summary"] = long_dataframe["whisper_transcript"].apply(
        lambda x: generate_bart_summary(x, min_len=40, max_len=100)
    )
    long_dataframe["tiny_summary"] = long_dataframe["whisper_transcript"].apply(
        generate_pegasus_summary
    )

    # Save output
    long_dataframe.to_csv("long_audio_summaries.csv", index=False)
    print("Summarization complete. Output saved to long_audio_summaries.csv")
