# YouTube Transcriber

A web-based tool to convert YouTube videos to text using OpenAI's Whisper AI.

## Features

- Paste any YouTube URL to transcribe
- Choose from multiple Whisper models (tiny, base, small, medium, large)
- Real-time progress updates
- Download MP3 audio and transcript text files

## Quick Start (macOS)

```bash
./run.sh
```

Then open http://127.0.0.1:5000

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

## Whisper Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 39 MB | Fastest | Lower |
| base | 74 MB | Fast | Good |
| small | 244 MB | Medium | Better |
| medium | 769 MB | Slower | Great |
| large | 1.5 GB | Slowest | Best |

Models are downloaded automatically on first use.
