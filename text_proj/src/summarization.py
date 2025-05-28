# summarization.py
import pandas as pd
from transformers import pipeline
from find_length import add_word_count_column, mark_long_transcripts

# Load summarization pipelines
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
pegasus_summarizer = pipeline("summarization", model="google/pegasus-xsum")

# Define Summary Functions
def generate_bart_summary(text, min_len=60, max_len=300):
    try:
        return bart_summarizer(text, min_length=min_len, max_length=max_len, do_sample=False)[0]["summary_text"]
    except Exception as e:
        return f"[ERROR: {e}]"

def generate_pegasus_summary(text):
    try:
        return pegasus_summarizer(text)[0]["summary_text"]
    except Exception as e:
        return f"[ERROR: {e}]"

# Main processing
if __name__ == "__main__":
    # Load and preprocess DataFrame
    path = "../data/whisper_with_groundtruth_100.csv"
    df = pd.read_csv("whisper_with_groundtruth_100.csv")
    df = add_word_count_column(df)
    df, threshold = mark_long_transcripts(df)

    # Filter long entries only
    long_df = df[df["is_long"]].copy()

    # Apply summarization
    long_df["long_summary"] = long_df["whisper_transcript"].apply(lambda x: generate_bart_summary(x, min_len=100, max_len=300))
    long_df["short_summary"] = long_df["whisper_transcript"].apply(lambda x: generate_bart_summary(x, min_len=40, max_len=100))
    long_df["tiny_summary"] = long_df["whisper_transcript"].apply(generate_pegasus_summary)

    # Save output
    long_df.to_csv("long_audio_summaries.csv", index=False)
    print("Summarization complete. Output saved to long_audio_summaries.csv")
