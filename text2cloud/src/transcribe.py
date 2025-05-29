import os
import whisper
from pathlib import Path
import torch
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # Load Whisper model
    model = whisper.load_model("small")  # Try "small" or "medium" for better accuracy

    # Update this to your actual LibriSpeech path
    base_dir = Path("/Users/christinexu/Desktop/MLDS/spring2025/cloud_text_projects/text_proj/data/LS_train100/train-clean-100")

    output_rows = []

    # Traverse all .flac files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".flac"):
                continue

            file_path = os.path.join(root, file)
            print(f"Processing {file_path}...")

            # Transcribe using Whisper
            result = model.transcribe(file_path, language="en")
            whisper_text = result["text"].strip()

            # Extract IDs from filename (e.g. 19-198-0031.flac)
            parts = file.replace(".flac", "").split("-")
            if len(parts) != 3:
                print(f"Skipping malformed file: {file}")
                continue

            speaker_id, chapter_id, sentence_id = parts

            # Construct path to transcript file (e.g. 19-198.trans.txt)
            transcript_file = Path(root) / f"{speaker_id}-{chapter_id}.trans.txt"
            if not transcript_file.exists():
                print(f"Transcript file not found: {transcript_file}")
                continue

            # Load the transcript dictionary once per chapter
            with open(transcript_file, 'r') as f:
                trans_lines = f.readlines()

            transcript_dict = {
                line.strip().split(" ", 1)[0]: line.strip().split(" ", 1)[1]
                for line in trans_lines if " " in line
            }

            uid = f"{speaker_id}-{chapter_id}-{sentence_id}"
            ground_truth = transcript_dict.get(uid, "")

            output_rows.append({
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "sentence_id": sentence_id,
                "file_path": file_path,
                "whisper_transcript": whisper_text,
                # convert ground truth to lowercase for comparison
                "ground_truth": ground_truth.lower()
            })

    # Save results
    df = pd.DataFrame(output_rows)
    df.to_csv("whisper_with_groundtruth_100.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    main()