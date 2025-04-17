import os
import boto3
from datetime import datetime, timezone

S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
S3_RECORDS_KEY = os.environ.get('S3_RECORDS_KEY')
S3_EMBEDDINGS_KEY = os.environ.get('S3_EMBEDDINGS_KEY')
RECORDS_PATH = 'data/records.json'
EMBEDDINGS_PATH = 'data/embeddings.npy'

def download_files_from_s3():
    """
    Checks if the file is present locally; if not, downloads it from S3.
    """
    for KEY, FILE_PATH in [(S3_RECORDS_KEY, RECORDS_PATH), (S3_EMBEDDINGS_KEY, EMBEDDINGS_PATH)]:
        if KEY is None:
            continue   # testing locally, key ommitted
        s3 = boto3.client('s3')
        s3_obj = s3.head_object(Bucket=S3_BUCKET, Key=KEY)
        s3_mtime = s3_obj["LastModified"].astimezone(timezone.utc)

        if os.path.exists(FILE_PATH):
            local_mtime = datetime.fromtimestamp(os.path.getmtime(FILE_PATH), tz=timezone.utc)
        else:
            local_mtime = None

        # Download if missing or older than S3 copy
        if local_mtime is None or local_mtime < s3_mtime:
            os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
            print(f"Downloading {KEY} to {FILE_PATH} from S3...")
            s3.download_file(S3_BUCKET, KEY, FILE_PATH)
            print(f"File {FILE_PATH} downloaded successfully.")
        else:
            print(f"Local file {FILE_PATH} already exists.")