from celery import Celery
from gpt import *
import boto3, uuid, os, json, datetime, io
import re, textwrap
from file_downloader import download_files_from_s3
from models import get_sentence_model
from search import search, filter

# Initialize the Celery app (using Redis as the broker)
broker_url = os.environ.get("CELERY_BROKER_URL")
result_backend = os.environ.get("CELERY_RESULT_BACKEND")
celery = Celery('tasks', 
                broker=broker_url,
                backend=result_backend)             

S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
s3 = boto3.client("s3")

model = None

def _sanitize_kw(prompt):
   """join prompt with '_' and keep only safe chars"""
   return prompt
   safe = re.sub(r"[^A-Za-z0-9\-]+", "_", "_".join(prompt))  # → china_ophthalmology
   return textwrap.shorten(safe, width=40, placeholder="")

def _upload_excel(raw_bytes: bytes, prompt, rid) -> str:
    key = f"results/{_sanitize_kw(prompt)}_{rid}.xlsx"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=raw_bytes,
        ContentType=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        ServerSideEncryption="aws:kms",
    )
    return key

@celery.task(bind=True)
def enrich_data_task(self, prompt, search_types, request_id):
    self.update_state(state='PROGRESS', meta={'status': 'Identifying companies and assets of interest...'})
    global model
    if model is None:
        download_files_from_s3()         # grab JSON + embeddings
        model = get_sentence_model()  # load the SentenceTransformer
        _ = model.encode("warm up")

    records = []
    matched = search(prompt, search_types, model)
    for search_type in search_types:
        filtered = filter(matched, doc_type=search_type)
        if filtered:
            records.extend(filtered)
    
    # Count only GPT‑backed records for progress
    total = len([r for r in records if r.get("type") != "trial"])

    def progress_cb(done, tot):
        pct = int(done * 100 / tot) if tot else 100
        self.update_state(
            state="PROGRESS",
            meta={
                "status": f"Processed {done}/{tot} records",
                "current": done,
                "total": tot,
                "percent": pct,
            },
        )

    # first update so the front‑end sees 0 %
    progress_cb(0, total)
    excel_b64 = enrich(records, prompt, progress_cb=progress_cb)

    # decode & upload
    excel_bytes = base64.b64decode(excel_b64)
    s3_key = _upload_excel(excel_bytes, prompt, request_id)

    # return only a small payload
    return {
        "status": "Task completed!",
        "s3_key": s3_key,
        "filename": os.path.basename(s3_key)
    }