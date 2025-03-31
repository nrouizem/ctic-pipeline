import os
import boto3

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
        if not os.path.exists(FILE_PATH):
            os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
            print(f"Downloading {KEY} to {FILE_PATH} from S3...")
            s3 = boto3.client('s3')
            s3.download_file(S3_BUCKET, KEY, FILE_PATH)
            print(f"File {FILE_PATH} downloaded successfully.")
        else:
            print(f"Local file {FILE_PATH} already exists.")