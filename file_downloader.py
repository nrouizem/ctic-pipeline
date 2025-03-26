import os
import boto3

S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
S3_FILE_KEY = os.environ.get('S3_FILE_KEY')
LOCAL_FILE_PATH = 'bpd_crawled_data.json'  # Adjust to your desired local path

def download_file_from_s3():
    """
    Checks if the file is present locally; if not, downloads from S3.
    """
    if not os.path.exists(LOCAL_FILE_PATH):
        os.makedirs(os.path.dirname(LOCAL_FILE_PATH), exist_ok=True)

        print("Downloading file from S3...")
        s3 = boto3.client('s3')
        s3.download_file(S3_BUCKET, S3_FILE_KEY, LOCAL_FILE_PATH)
        print("File downloaded successfully.")
    else:
        print("Local file already exists.")
