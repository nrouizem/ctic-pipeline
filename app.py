from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash, abort
from tasks import enrich_data_task, celery  # celery is our Celery app instance
import uuid
import os
import boto3

def generate_unique_id():
    # Generate a unique integer (here we take the last 8 digits, adjust as needed)
    return int(uuid.uuid4().int % 10**8)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
app.config['SEARCH_PASSWORD'] = os.environ.get("SEARCH_PASSWORD")

S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
s3 = boto3.client("s3")
if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET env‑var is required")

@app.route('/', methods=['GET', 'POST'])
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Retrieve the keywords (company names) from the form.
        prompt = request.form.get('prompt')
        search_types = request.form.getlist('search_types')
        request_id = generate_unique_id()
        # print input to have a way to see what ppl are searching (not great but whatever)
        print("PROMPT: ", prompt)
        print("SEARCH TYPES: ", ', '.join(search_types))

        # Enqueue the enrichment task.
        task = enrich_data_task.delay(prompt, search_types, request_id)
        return render_template('submission.html', 
                               prompt=prompt,
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

@app.route("/download/<task_id>")
def download_file(task_id):
    """Return a 302 redirect to a pre‑signed S3 object, or JSON status if pending."""
    keywords = request.args.get("keywords", "")   # purely for analytics, not used now
    task = celery.AsyncResult(task_id)

    # ---------------------------- PENDING / RUNNING ----------------------------
    if task.state == "PENDING":
        return jsonify({"state": task.state,
                        "status": "File not available yet (task pending)"}), 202

    if task.state not in ("SUCCESS", "FAILURE"):
        # still running (e.g. PROGRESS)
        return jsonify({"state": task.state,
                        "status": task.info.get("status", "")}), 202

    # ---------------------------- FAILURE --------------------------------------
    if task.state == "FAILURE":
        # surface the traceback or a generic error
        return jsonify({"state": task.state, "status": str(task.info)}), 500

    # ---------------------------- SUCCESS --------------------------------------
    # Celery task returns {'status': 'Task completed!', 's3_key': 'results/…xlsx'}
    s3_key = task.info.get("s3_key")
    if not s3_key:
        abort(404, description="Result key missing in task metadata")

    try:

        filename = task.info.get("filename", os.path.basename(s3_key))

        presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": s3_key,
                # Force the browser’s save‑as name
                "ResponseContentDisposition": f'attachment; filename="{filename}"',
            },
            ExpiresIn=900,
            HttpMethod="GET",
        )
    except Exception as e:
        app.logger.exception("Failed to generate pre‑signed URL")
        abort(500, description="Unable to generate download link")

    # Optional: log for audit
    app.logger.info("Redirecting download for %s to %s", task_id, s3_key)

    # Fastest UX: just HTTP‑redirect the browser to S3
    return redirect(presigned_url, code=302)

@app.route("/healthz", methods=["GET","HEAD"])
def healthz():
    return "OK", 200

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