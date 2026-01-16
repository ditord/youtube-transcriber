import os
import uuid
import subprocess
import threading
import torch
from flask import Flask, render_template, request, jsonify, send_file
import whisper

app = Flask(__name__)
app.config['DOWNLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'downloads')

# Store job status
jobs = {}

# Cache for loaded models
model_cache = {}

# Language configurations
LANGUAGE_CONFIGS = {
    'english': {
        'type': 'whisper',
        'language': 'en',
        'name': 'English'
    },
    'russian': {
        'type': 'huggingface',
        'model_id': 'antony66/whisper-large-v3-russian',
        'language': 'russian',
        'name': 'Russian'
    },
    'armenian': {
        'type': 'huggingface',
        'model_id': 'Chillarmo/whisper-large-v3-turbo-armenian',
        'language': 'armenian',
        'name': 'Armenian'
    }
}

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def transcribe_with_whisper(audio_path, model_name, language):
    """Transcribe using standard OpenAI Whisper."""
    cache_key = f"whisper_{model_name}"
    if cache_key not in model_cache:
        model_cache[cache_key] = whisper.load_model(model_name)

    model = model_cache[cache_key]
    result = model.transcribe(audio_path, language=language)
    return result['text']

def transcribe_with_huggingface(audio_path, model_id, language, job):
    """Transcribe using HuggingFace transformers model."""
    from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
    import librosa

    device = get_device()
    torch_dtype = torch.float16 if device in ['cuda', 'mps'] else torch.float32

    cache_key = f"hf_{model_id}"

    if cache_key not in model_cache:
        job['message'] = f'Loading {language} model (first time may take a while)...'

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        model_cache[cache_key] = pipe

    pipe = model_cache[cache_key]

    # Load audio with librosa
    audio, sr = librosa.load(audio_path, sr=16000)

    # Transcribe
    result = pipe(audio, generate_kwargs={"language": language, "max_new_tokens": 448})
    return result['text']

def process_video(job_id, youtube_url, model_name, language):
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

        # Step 2: Transcribe
        job['status'] = 'transcribing'
        lang_config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['english'])
        job['message'] = f'Transcribing ({lang_config["name"]})...'

        if lang_config['type'] == 'whisper':
            # Use standard Whisper for English
            job['message'] = f'Transcribing with Whisper {model_name} ({lang_config["name"]})...'
            transcript = transcribe_with_whisper(mp3_path, model_name, lang_config['language'])
        else:
            # Use HuggingFace model for Russian/Armenian
            job['message'] = f'Transcribing with specialized {lang_config["name"]} model...'
            transcript = transcribe_with_huggingface(
                mp3_path,
                lang_config['model_id'],
                lang_config['language'],
                job
            )

        # Save transcript
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript)

        job['status'] = 'completed'
        job['message'] = 'Transcription complete!'
        job['mp3_path'] = mp3_path
        job['txt_path'] = txt_path
        job['transcript_preview'] = transcript[:500] + '...' if len(transcript) > 500 else transcript

    except Exception as e:
        job['status'] = 'error'
        job['message'] = str(e)
        import traceback
        print(traceback.format_exc())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    youtube_url = data.get('url')
    model_name = data.get('model', 'base')
    language = data.get('language', 'english')

    if not youtube_url:
        return jsonify({'error': 'No URL provided'}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'starting',
        'message': 'Starting...',
        'url': youtube_url
    }

    # Start processing in background thread
    thread = threading.Thread(target=process_video, args=(job_id, youtube_url, model_name, language))
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
