import os
import uuid
import subprocess
import threading
from flask import Flask, render_template, request, jsonify, send_file
import whisper

app = Flask(__name__)
app.config['DOWNLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'downloads')

# Store job status
jobs = {}

def process_video(job_id, youtube_url, model_name):
    """Download and transcribe YouTube video in background."""
    job = jobs[job_id]
    job_folder = os.path.join(app.config['DOWNLOAD_FOLDER'], job_id)
    os.makedirs(job_folder, exist_ok=True)

    mp3_path = os.path.join(job_folder, 'audio.mp3')
    txt_path = os.path.join(job_folder, 'transcript.txt')

    try:
        # Step 1: Download audio
        job['status'] = 'downloading'
        job['message'] = 'Downloading audio from YouTube...'

        result = subprocess.run([
            'yt-dlp', '-x', '--audio-format', 'mp3',
            '-o', mp3_path.replace('.mp3', '.%(ext)s'),
            '--no-playlist',
            youtube_url
        ], capture_output=True, text=True)

        if result.returncode != 0:
            job['status'] = 'error'
            job['message'] = f'Download failed: {result.stderr}'
            return

        # Find the actual mp3 file (yt-dlp might name it differently)
        for f in os.listdir(job_folder):
            if f.endswith('.mp3'):
                actual_mp3 = os.path.join(job_folder, f)
                if actual_mp3 != mp3_path:
                    os.rename(actual_mp3, mp3_path)
                break

        if not os.path.exists(mp3_path):
            job['status'] = 'error'
            job['message'] = 'MP3 file not created'
            return

        # Step 2: Transcribe with Whisper
        job['status'] = 'transcribing'
        job['message'] = f'Transcribing with Whisper ({model_name} model)...'

        model = whisper.load_model(model_name)
        result = model.transcribe(mp3_path)

        # Save transcript
        with open(txt_path, 'w') as f:
            f.write(result['text'])

        job['status'] = 'completed'
        job['message'] = 'Transcription complete!'
        job['mp3_path'] = mp3_path
        job['txt_path'] = txt_path
        job['transcript_preview'] = result['text'][:500] + '...' if len(result['text']) > 500 else result['text']

    except Exception as e:
        job['status'] = 'error'
        job['message'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    youtube_url = data.get('url')
    model_name = data.get('model', 'base')

    if not youtube_url:
        return jsonify({'error': 'No URL provided'}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'starting',
        'message': 'Starting...',
        'url': youtube_url
    }

    # Start processing in background thread
    thread = threading.Thread(target=process_video, args=(job_id, youtube_url, model_name))
    thread.start()

    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])

@app.route('/download/<job_id>/<file_type>')
def download(job_id, file_type):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed'}), 400

    if file_type == 'mp3':
        return send_file(job['mp3_path'], as_attachment=True, download_name='audio.mp3')
    elif file_type == 'txt':
        return send_file(job['txt_path'], as_attachment=True, download_name='transcript.txt')
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host=host, port=port, debug=debug)
