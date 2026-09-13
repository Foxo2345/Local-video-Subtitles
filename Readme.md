## Setup
Requires ffmpeg on PATH (winget install ffmpeg / sudo pacman -S ffmpeg)

python -m venv .venv    

source .venv/bin/activate      # Linux  
source .venv/bin/activate.fish # If using fish terminal

.venv\Scripts\activate         # Windows

# NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu121    
pip install openai-whisper argostranslate srt

# Any CPU-only machine:
pip install torch --index-url https://download.pytorch.org/whl/cpu  
pip install openai-whisper argostranslate srt


## Usage

python subtitle_pipeline.py video.mp4 --to-lang el  
python subtitle_pipeline.py video.mp4 --to-lang el --hardcode

The script auto-detects CUDA and picks a sensible default Whisper model
(medium on GPU, small on CPU)
