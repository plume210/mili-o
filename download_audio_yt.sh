#!/bin/bash

# Dossier de destination
OUTPUT_DIR="audio"
mkdir -p "$OUTPUT_DIR"

# Liste de vidéos à télécharger (ajoute autant de liens que tu veux)
YOUTUBE_URLS=(
  "https://www.youtube.com/watch?v=7UXudW2-zx0"
  "https://www.youtube.com/watch?v=dGR5w14FsVo"
)

# Téléchargement et extraction en .wav
for url in "${YOUTUBE_URLS[@]}"; do
  echo "🔻 Downloading $url..."
  yt-dlp -x --audio-format wav --audio-quality 0 -o "$OUTPUT_DIR/%(id)s.%(ext)s" "$url"
done

echo "✅ All audio downloaded to $OUTPUT_DIR/"
