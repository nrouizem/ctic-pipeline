from celery import Celery
import time
from fetch import *
from gpt import *
import re
from file_downloader import download_file_from_s3

# Initialize the Celery app (using Redis as the broker)
broker_url = os.environ.get("CELERY_BROKER_URL")
result_backend = os.environ.get("CELERY_RESULT_BACKEND")
celery = Celery('tasks', 
                broker=broker_url,
                backend=result_backend)

download_file_from_s3()                

@celery.task(bind=True)
def enrich_data_task(self, companies):
    self.update_state(state='PROGRESS', meta={'status': 'Researching companies and assets of interest...'})
    file_path = enrich(companies)
    return {'file_path': file_path, 'status': 'Task completed!'}