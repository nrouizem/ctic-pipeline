from celery import Celery
import time
from fetch import *
from gpt import *
import re

# Initialize the Celery app (using Redis as the broker)
celery = Celery('tasks', 
                broker='redis://localhost:6379/0',
                backend='redis://localhost:6379/0')
                

@celery.task(bind=True)
def enrich_data_task(self, companies):
    self.update_state(state='PROGRESS', meta={'status': 'Researching companies and assets of interest...'})
    file_path = enrich(companies)
    return {'file_path': file_path, 'status': 'Task completed!'}