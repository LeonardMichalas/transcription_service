import moviepy.editor as mp
import whisper
from pyannote.audio import Pipeline
from tqdm import tqdm
import torch
import os

# Define the video file to transcribe and the output file name
VIDEO_FILE = "interview_files/test.mp4"
ALIGNED_TRANSCRIPTION_FILE = "transcribed_files/test.txt"

# Retrieve the Hugging Face authentication token from environment variables
AUTH_TOKEN = os.getenv("HUGGING_FACE_AUTH_TOKEN")
if AUTH_TOKEN is None:
    raise ValueError("Please set the HUGGING_FACE_AUTH_TOKEN environment variable.")

# Function to convert video to audio
def video_to_audio(video_file, audio_file):
    video = mp.VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)
    print(f"Converted {video_file} to {audio_file}")

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("medium")
    result = model.transcribe(audio_file)
    if result and 'text' in result:
        print(f"Transcription completed for {audio_file}")
        return result['text'], result['segments']
    else:
        raise ValueError("Failed to transcribe audio")

# Function to diarize transcription using PyAnnote
def diarize_audio(audio_file, auth_token):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth_token)
    diarization = pipeline(audio_file)
    return diarization

# Function to align transcription with diarization
def align_transcription_diarization(transcription_segments, diarization):
    aligned_transcription = []
    included_segments = set()  # Track included transcription segments

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turn_text = ''
        for segment in transcription_segments:
            segment_id = (segment['start'], segment['end'])
            if segment_id in included_segments:
                continue  # Skip already included segments
            segment_start = segment['start']
            segment_end = segment['end']
            if segment_start >= turn.start and segment_end <= turn.end:
                turn_text += segment['text'] + ' '
                included_segments.add(segment_id)
            elif segment_start < turn.end and segment_end > turn.start:
                turn_text += segment['text'] + ' '
                included_segments.add(segment_id)
        if turn_text:
            aligned_transcription.append((speaker, turn.start, turn.end, turn_text.strip()))
    return aligned_transcription

# Function to save aligned transcription to a text file
def save_aligned_transcription(aligned_transcription, file_path):
    with open(file_path, 'w') as file:
        for speaker, start, end, text in aligned_transcription:
            file.write(f"Speaker {speaker} ({start:.1f}s - {end:.1f}s): {text}\n")
    print(f"Aligned transcription saved to {file_path}")

# Function to clean up temporary files
def clean_up(audio_file):
    try:
        os.remove(audio_file)
        print(f"Deleted temporary file: {audio_file}")
    except OSError as e:
        print(f"Error deleting file {audio_file}: {e}")

# Main function to process video to diarized transcription
def process_video(video_file, aligned_transcription_file, auth_token):
    audio_file = "audio.wav"
    
    # Step 1: Convert video to audio
    video_to_audio(video_file, audio_file)
    
    # Step 2: Transcribe audio using Whisper
    transcription, transcription_segments = transcribe_audio(audio_file)
    print("Transcription:\n", transcription)
    
    # Step 3: Diarize audio using PyAnnote
    diarization = diarize_audio(audio_file, auth_token)
    print("Diarization completed for", audio_file)
    
    # Step 4: Align transcription with diarization
    aligned_transcription = align_transcription_diarization(transcription_segments, diarization)
    
    # Step 5: Save aligned transcription to a text file
    save_aligned_transcription(aligned_transcription, ALIGNED_TRANSCRIPTION_FILE)
    
    # Step 6: Clean up temporary files
    clean_up(audio_file)
    

# Run the process with your defined variables
process_video(VIDEO_FILE, ALIGNED_TRANSCRIPTION_FILE, AUTH_TOKEN)

