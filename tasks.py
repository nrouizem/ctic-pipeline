from celery import Celery
from gpt import *

# Initialize the Celery app (using Redis as the broker)
broker_url = os.environ.get("CELERY_BROKER_URL")
result_backend = os.environ.get("CELERY_RESULT_BACKEND")
celery = Celery('tasks', 
                broker=broker_url,
                backend=result_backend)             

@celery.task(bind=True)
def enrich_data_task(self, records, keywords):
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
    
    return {'excel_data': excel_b64, 'status': 'Task completed!'}