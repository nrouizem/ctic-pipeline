from celery import Celery
from gpt import *

# Initialize the Celery app (using Redis as the broker)
broker_url = os.environ.get("CELERY_BROKER_URL")
result_backend = os.environ.get("CELERY_RESULT_BACKEND")
celery = Celery('tasks', 
                broker=broker_url,
                backend=result_backend)             

@celery.task(bind=True)
def enrich_data_task(self, records, keywords, search_types):
    self.update_state(state='PROGRESS', meta={'status': 'Researching companies and assets of interest...'})
    excel_b64 = enrich(records, keywords, search_types)
    return {'excel_data': excel_b64, 'status': 'Task completed!'}