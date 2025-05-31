import boto3
import os
from dotenv import load_dotenv
import json
import spacy
import tempfile
from pathlib import Path
import boto3
import random
import sys

# load_dotenv()

# BUCKET_NAME = os.getenv("BUCKET_NAME")
# PREFIX_DATA = os.getenv("BUCKET_PREFIX_DATA")
# PREFIX_MODEL = os.getenv("BUCKET_PREFIX_MODEL")
s3 = boto3.client("s3")

def list_audio_files(bucket: str, prefix: str) -> list:
    """List .flac or .wav files under a prefix in S3"""
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [
        obj["Key"] for obj in response.get("Contents", [])
        if obj["Key"].endswith((".flac", ".wav"))
    ]


def download_file(bucket, key: str, local_path: str):
    """Download a file from S3 to local path"""
    s3.download_file(bucket, key, local_path)

def upload_file(local_path, bucket, key):
    """Upload a file to S3 under the given key"""
    s3.upload_file(local_path, bucket, key)

def file_exists(bucket, key):
    """Check if a key exists in S3"""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def load_ner_model_from_s3(bucket_name: str, prefix: str):
    """
    Downloads a spaCy NER model folder from S3 and loads it into memory.
    """
    # s3 = boto3.client('s3')
    temp_dir = tempfile.TemporaryDirectory()
    local_model_path = Path(temp_dir.name) / "ner_model"

    # Recursively download all files in the prefix
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel_path = Path(key).relative_to(prefix)
            local_file_path = local_model_path / rel_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket_name, key, str(local_file_path))

    print(f"Model downloaded to {local_model_path}")
    return spacy.load(str(local_model_path)), temp_dir  # Keep temp_dir alive


# def test_ner_model(ner_cloud_model):
#     text = "Apple is looking to buy a startup in San Francisco"
#     doc = ner_cloud_model(text)
#     for ent in doc.ents:
#         print(f"{ent.text}: {ent.label_}")


def trigger_lambda(bucket: str, key: str, lambda_function_name: str) -> dict:
    """Trigger a Lambda function with an S3 file path."""
    lambda_client = boto3.client("lambda")
    payload = {
        "bucket": bucket,
        "key": key
    }
    response = lambda_client.invoke(
        FunctionName=lambda_function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    return json.load(response['Payload'])


def fetch_result_json(bucket: str, key: str) -> dict:
    """Fetch a JSON result from S3."""
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))