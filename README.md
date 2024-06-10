# Video Transcription and Diarization Tool

## Description
This project provides a comprehensive tool for transcribing and diarizing video files. The tool converts video to audio, transcribes the audio using the Whisper model, and then diarizes the transcription with PyAnnote. 

## Features
- **Video to Audio Conversion**: Converts video files to audio files.
- **Audio Transcription**: Uses the Whisper model to transcribe audio files.
- **Speaker Diarization**: Diarizes the transcription to identify different speakers.

## Installation

### Prerequisites
- Python 3.x
- [Whisper](https://github.com/openai/whisper)
- [PyAnnote](https://github.com/pyannote/pyannote-audio)
- [MoviePy](https://github.com/Zulko/moviepy)
- [TQDM](https://github.com/tqdm/tqdm)
- [Torch](https://pytorch.org/)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository

2. **Create and activate a virtual environment**:
   ```bash
    python -m venv myenv
    source myenv/bin/activate

3. **Set the Hugging Face authentication token**:
    ```bash
    export HUGGING_FACE_AUTH_TOKEN="your_auth_token"

4. **Run script**:
    ```bash
    python transcribe.py