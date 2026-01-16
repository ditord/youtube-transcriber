#!/bin/bash

cd "$(dirname "$0")"

echo "Starting YouTube Transcriber..."
echo "Open http://127.0.0.1:5000 in your browser"
echo "Press Ctrl+C to stop"
echo ""

# For macOS with Homebrew Python
if [ -f /opt/homebrew/opt/python@3.11/bin/python3.11 ]; then
    DEBUG=true /opt/homebrew/opt/python@3.11/bin/python3.11 app.py
else
    DEBUG=true python3 app.py
fi
