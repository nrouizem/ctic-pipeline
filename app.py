from flask import Flask, request, render_template, jsonify, send_file
from tasks import enrich_data_task, celery  # celery is our Celery app instance
import re
import uuid
from fetch import *
from file_downloader import download_file_from_s3

def generate_unique_id():
    # Generate a unique integer (here we take the last 8 digits, adjust as needed)
    return int(uuid.uuid4().int % 10**8)

app = Flask(__name__)

download_file_from_s3()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Retrieve the keywords (company names) from the form.
        keywords = request.form.get('keywords')
        print("keywords")
        request_id = generate_unique_id()
        print("id")
        # Split by commas or whitespace.
        keywords = re.split(r'[,\s]+', keywords)
        bpd_data = filter(bpd_relevance(keywords))
        print("data received")

        # Enqueue the enrichment task.
        task = enrich_data_task.delay(bpd_data)
        return render_template('submission.html', 
                               keywords=', '.join(keywords), 
                               output = ', '.join(bpd_data),
                               request_id=request_id,
                               task_id=task.id)
    return render_template('index.html')

@app.route('/status/<task_id>')
def task_status(task_id):
    task = celery.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'status': task.info.get('status', '') if isinstance(task.info, dict) else str(task.info),
            'result': task.info.get('file_path', '') if isinstance(task.info, dict) else task.info,
        }
    else:
        response = {'state': task.state, 'status': str(task.info)}
    return jsonify(response)

@app.route('/download/<task_id>')
def download_file(task_id):
    task = celery.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        file_path = task.info.get('file_path')
        return send_file(file_path,
                         as_attachment=True,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    else:
        return "File not available yet", 404

if __name__ == '__main__':
    app.run(debug=True)