## Setup

python -m venv .venv    

source .venv/bin/activate      # Linux  
source .venv/bin/activate.fish # If using fish terminal

.venv\Scripts\activate         # Windows

pip install openai-whisper argostranslate srt

# NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Any CPU-only machine:
pip install torch --index-url https://download.pytorch.org/whl/cpu

Requires ffmpeg on PATH (winget install ffmpeg / sudo pacman -S ffmpeg)

## Usage

python subtitle_pipeline.py video.mp4 --to-lang el  
python subtitle_pipeline.py video.mp4 --to-lang el --hardcode

The script auto-detects CUDA and picks a sensible default Whisper model
(medium on GPU, small on CPU)
