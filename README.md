# Transcription Service

[![CI](https://github.com/LeonardMichalas/transcription_service/actions/workflows/ci.yml/badge.svg)](https://github.com/LeonardMichalas/transcription_service/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A command-line tool that transcribes an audio or video file and labels each line with its speaker. Speech-to-text comes from [Whisper](https://github.com/openai/whisper), and the speaker labels from [pyannote](https://github.com/pyannote/pyannote-audio). Everything runs locally: after the models are downloaded once, the recording never leaves your machine.

```console
$ transcribe interview.mp4
Extracting audio from interview.mp4
Transcribing with Whisper medium on cuda
Identifying speakers
Wrote 42 lines from 2 speaker(s) to interview.txt
```

```text
[00:00:00.000 - 00:00:06.480] SPEAKER_00: Thanks for taking the time. Could you start by describing your role?
[00:00:06.900 - 00:00:15.220] SPEAKER_01: Sure. I run a small bakery together with my sister.
```

## Requirements

- Python 3.10 or newer
- [ffmpeg](https://ffmpeg.org/download.html) on your `PATH`
- A free [Hugging Face](https://huggingface.co) access token. The speaker model is gated, so accept the terms on [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) with the same account first.

A GPU is optional. On a CPU, use a smaller model such as `small` or `turbo` for long recordings.

## Install

```bash
uv tool install git+https://github.com/LeonardMichalas/transcription_service
# or: pip install git+https://github.com/LeonardMichalas/transcription_service
```

## Usage

```bash
export HF_TOKEN=hf_...

transcribe interview.mp4                          # writes interview.txt
transcribe meeting.m4a -f srt -l de -s 3          # German, three speakers, subtitles
transcribe talk.mkv -m turbo -o transcripts/talk.json -f json
```

| Option | Meaning |
| --- | --- |
| `-o`, `--output` | Output file. Defaults to the input path with the format's extension. |
| `-f`, `--format` | `txt` (default), `srt` subtitles, or `json`. |
| `-m`, `--model` | Whisper model: `tiny`, `base`, `small`, `medium` (default), `large-v3`, `turbo`. |
| `-l`, `--language` | Spoken language, such as `en` or `de`. Detected automatically if left out. |
| `-s`, `--speakers` | Number of speakers, if known. Improves the speaker labels. |
| `--device` | `auto` (default), `cpu` or `cuda`. |

## How it works

1. **ffmpeg** extracts the audio track as 16 kHz mono WAV, the input both models expect.
2. **Whisper** turns the speech into text segments with start and end times.
3. **pyannote** splits the recording into speaker turns.
4. **Alignment** gives each text segment to the speaker whose turns overlap it most, then merges consecutive segments from the same speaker into one line. Every segment appears exactly once, so no text is lost or repeated at a speaker change.

## Development

```bash
uv sync --only-group dev                  # tools only, no model dependencies
uv run --only-group dev pytest            # alignment, output formats, CLI, ffmpeg step
uv run --only-group dev ruff check .
uv run --only-group dev mypy transcribe.py tests
```

The tests cover everything except the models themselves, so they run in seconds and need no GPU or token.

## License

MIT, see [LICENSE](LICENSE).

<sub>A personal side project, written in my own free time.</sub>
