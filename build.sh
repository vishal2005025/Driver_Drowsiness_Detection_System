#!/usr/bin/env bash
set -o errexit

# Download and install ffmpeg + ffplay for Render
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar xJ
cp ffmpeg-*-static/ffmpeg ffmpeg-*-static/ffplay /usr/local/bin/
