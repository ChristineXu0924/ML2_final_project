# Cloud Project

## Project Overview:
This project delivers a fully automated, serverless audio analytics pipeline on AWS. Users interact with a Streamlit web interface to upload or select audio files from S3. Depending on the workflow, AWS Lambda or EC2 is triggered to transcribe the audio via Whisper. The text is then processed by a fine-tuned spaCy NER model and summarized using transformer-based models, with optional translation to Chinese using NLLB-200. Outputs are rendered instantly in the web app. The architecture integrates AWS Lambda, EC2, S3, API Gateway, and CloudWatch for scalable, real-time processing with minimal infrastructure management.
 
## Problem Statement:
Audio data is abundant across domains like customer support, healthcare, media, and education — yet much of it remains unstructured and untapped. Traditional manual processing of audio for extracting insights is labor-intensive, error-prone, and unscalable. Organizations lack a streamlined solution to automatically convert audio into structured, meaningful information in real time, especially one that is cost-effective and cloud-native.

## File Structure 


## Unit test 
How to run the Unit tests in Docker:

```bash
cd text2cloud
```

Build docker img: 
```bash
docker build -t cloud2text .
```
Run the test:
```bash
docker run -e PYTHONPATH=/app cloud2text pytest -s tests/test_pipeline.py
```
