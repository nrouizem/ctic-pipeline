from celery import Celery
from gpt import *
import boto3, uuid, os, json, datetime, io
import re, textwrap

# Initialize the Celery app (using Redis as the broker)
broker_url = os.environ.get("CELERY_BROKER_URL")
result_backend = os.environ.get("CELERY_RESULT_BACKEND")
celery = Celery('tasks', 
                broker=broker_url,
                backend=result_backend)             

S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
s3 = boto3.client("s3")

def _sanitize_kw(keywords):
   """join keywords with '_' and keep only safe chars"""
   safe = re.sub(r"[^A-Za-z0-9\-]+", "_", "_".join(keywords))  # → china_ophthalmology
   return textwrap.shorten(safe, width=40, placeholder="")

def _upload_excel(raw_bytes: bytes, keywords, rid) -> str:
    key = f"results/{_sanitize_kw(keywords)}_{rid}.xlsx"
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
def enrich_data_task(self, records, keywords, request_id):
    self.update_state(state='PROGRESS', meta={'status': 'Researching companies and assets of interest...'})
    excel_b64 = enrich(records, keywords)
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
    excel_b64 = enrich(records, keywords, progress_cb=progress_cb)

    # decode & upload
    excel_bytes = base64.b64decode(excel_b64)
    s3_key = _upload_excel(excel_bytes, keywords, request_id)

    # return only a small payload
    return {
        "status": "Task completed!",
        "s3_key": s3_key,
        "filename": os.path.basename(s3_key)
    }