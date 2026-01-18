# YouTube Transcriber

A web-based tool to convert YouTube videos to text using OpenAI's Whisper AI.

## Features

- Paste any YouTube URL to transcribe
- **Multi-language support**: English, Russian, and Armenian
- Choose from multiple Whisper models (tiny, base, small, medium, large) for English
- Specialized fine-tuned models for Russian and Armenian from HuggingFace
- Real-time progress updates
- Download MP3 audio and transcript text files

## Supported Languages

| Language | Model | Source |
|----------|-------|--------|
| English | OpenAI Whisper (selectable size) | Standard |
| Russian | whisper-large-v3-russian | [HuggingFace](https://huggingface.co/ditord/whisper-large-v3-russian) |
| Armenian | whisper-large-v3-turbo-armenian | [HuggingFace](https://huggingface.co/ditord/whisper-large-v3-turbo-armenian) |

**Fallback:** If specialized models are unavailable, the app automatically falls back to standard Whisper with the appropriate language setting.

## Quick Start (macOS)

```bash
./run.sh
```

Then open http://127.0.0.1:5000

---

## Docker

For Docker-based deployment, use the [`docker` branch](https://github.com/ditord/youtube-transcriber/tree/docker):

```bash
git clone -b docker https://github.com/ditord/youtube-transcriber.git
cd youtube-transcriber
docker compose up -d
```

Then open http://localhost:5000

See the [docker branch README](https://github.com/ditord/youtube-transcriber/tree/docker#docker-recommended) for full instructions.

---

## Linux Server Setup

### 1. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and required packages
sudo apt install -y python3.11 python3.11-venv python3-pip ffmpeg git
```

### 2. Install yt-dlp

```bash
sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
sudo chmod a+rx /usr/local/bin/yt-dlp
```

### 3. Clone the Repository

```bash
cd ~
git clone https://github.com/ditord/youtube-transcriber.git
cd youtube-transcriber
```

### 4. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 5. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the Server

For local network access (replace `YOUR_SERVER_IP` with actual IP):

```bash
HOST=0.0.0.0 PORT=5000 python3 app.py
```

Access from any device on your network: `http://YOUR_SERVER_IP:5000`

---

## Run as a Service (Systemd)

Create a systemd service for auto-start:

```bash
sudo nano /etc/systemd/system/youtube-transcriber.service
```

Paste this (update paths and username):

```ini
[Unit]
Description=YouTube Transcriber
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/youtube-transcriber
Environment="HOST=0.0.0.0"
Environment="PORT=5000"
ExecStart=/home/YOUR_USERNAME/youtube-transcriber/venv/bin/python app.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable youtube-transcriber
sudo systemctl start youtube-transcriber
```

Check status:

```bash
sudo systemctl status youtube-transcriber
```

View logs:

```bash
sudo journalctl -u youtube-transcriber -f
```

---

## Firewall (if needed)

```bash
sudo ufw allow 5000/tcp
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `127.0.0.1` | Bind address (`0.0.0.0` for network access) |
| `PORT` | `5000` | Server port |
| `DEBUG` | `false` | Enable debug mode |

---

## Pre-downloading Models

Models are downloaded automatically on first use, but the initial download can be slow. To pre-download models before running the app:

### Install HuggingFace CLI (if not installed)

```bash
pip install huggingface_hub
```

### Download Russian Model (~3GB)

```bash
huggingface-cli download antony66/whisper-large-v3-russian
```

### Download Armenian Model (~3GB)

```bash
huggingface-cli download Chillarmo/whisper-large-v3-turbo-armenian
```

### Custom Cache Location (Optional)

By default, models are cached in `~/.cache/huggingface/hub/`. To use a custom location:

```bash
export HF_HOME=/path/to/your/models
huggingface-cli download antony66/whisper-large-v3-russian
huggingface-cli download Chillarmo/whisper-large-v3-turbo-armenian
```

Set `HF_HOME` before running the app to use models from your custom location.

---

## Whisper Models (English)

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39 MB | Fastest | Lower |
| base | 74 MB | Fast | Good |
| small | 244 MB | Medium | Better |
| medium | 769 MB | Slower | Great |
| large | 1.5 GB | Slowest | Best |

English Whisper models are downloaded automatically on first use.
