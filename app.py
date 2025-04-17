from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, session, flash
from tasks import enrich_data_task, celery  # celery is our Celery app instance
import re
import uuid
from search import *
from file_downloader import download_files_from_s3
import base64
import io
import os
from sentence_transformers import SentenceTransformer

def generate_unique_id():
    # Generate a unique integer (here we take the last 8 digits, adjust as needed)
    return int(uuid.uuid4().int % 10**8)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
app.config['SEARCH_PASSWORD'] = os.environ.get("SEARCH_PASSWORD")

download_files_from_s3()

# pre-warming cache
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')

@app.route('/', methods=['GET', 'POST'])
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Retrieve the keywords (company names) from the form.
        keywords = request.form.get('keywords')
        search_types = request.form.getlist('search_types')
        request_id = generate_unique_id()
        # Split by commas or whitespace.
        keywords = re.split(r'[,\s]+', keywords)
        # print input to have a way to see what ppl are searching (not great but whatever)
        print("KEYWORDS: ", ', '.join(keywords))
        print("SEARCH TYPES: ", ', '.join(search_types))

        records = []
        output = ""
        for search_type in search_types:
            matched = search(' '.join(keywords), [search_type], model)  # restrict context
            filtered = filter(matched, doc_type=search_type)
            records.extend(filtered)
            print([record["type"] for record in records])
            if search_type != "trial":
                output += ', '.join([record["company"] for record in records])
            if search_type == "trial":
                output += ""

        # Enqueue the enrichment task.
        task = enrich_data_task.delay(records, keywords)
        return render_template('submission.html', 
                               keywords=', '.join(keywords), 
                               output=output,
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
            'status': task.info.get('status', '') if isinstance(task.info, dict) else str(task.info),
            'current': task.info.get('current', 0),
            'total':   task.info.get('total', 0),
            'percent': task.info.get('percent', 0),
        }
    else:
        response = {'state': task.state, 'status': str(task.info)}
    return jsonify(response)

@app.route('/download/<task_id>')
def download_file(task_id):
    keywords = request.args.get('keywords', '')  # defaults to empty string if not provided
    task = celery.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        excel_b64 = task.info.get('excel_data')
        if not excel_b64:
            return "No file data found", 404
        
        excel_bytes = base64.b64decode(excel_b64)
        file_buffer = io.BytesIO(excel_bytes)
        file_buffer.seek(0)
        
        # Use the keywords query parameter in the filename if provided
        download_name = f"{', '.join(keywords.split(','))}_search.xlsx" if keywords else "data.xlsx"
        
        return send_file(
            file_buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=download_name
        )
    elif task.state == 'PENDING':
        return "File not available yet (task pending)", 202
    else:
        return f"Task in state: {task.state}", 202


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == app.config['SEARCH_PASSWORD']:
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            flash('Invalid password, please try again.')
            return render_template('login.html')
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)