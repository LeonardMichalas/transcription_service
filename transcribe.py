"""Transcribe a recording and label who said what.

The pipeline has four steps:

1. ffmpeg extracts the audio track as 16 kHz mono WAV.
2. Whisper turns the speech into timed text segments.
3. pyannote splits the recording into speaker turns.
4. Each text segment goes to the speaker it overlaps most, and consecutive
   segments from the same speaker are merged into one line.

The heavy libraries (Whisper, pyannote, torch) are imported only when they are
needed, so ``--help`` and the alignment logic work without them.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__version__ = "0.2.0"

log = logging.getLogger("transcribe")

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
FORMATS = ("txt", "srt", "json")


@dataclass(frozen=True)
class Segment:
    """A piece of transcribed text with its position in the recording."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Turn:
    """A stretch of time in which one speaker is talking."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Utterance:
    """What one speaker said in one go: the final unit of output."""

    speaker: str
    start: float
    end: float
    text: str


# --- Alignment ---------------------------------------------------------------


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(segment: Segment, turns: Sequence[Turn]) -> str:
    """Return the speaker whose turns overlap the segment the most.

    A segment that overlaps no turn at all, which happens in pauses the
    diarization model skipped, goes to the speaker of the nearest turn.
    """
    if not turns:
        return "UNKNOWN"

    totals: dict[str, float] = {}
    for turn in turns:
        shared = _overlap(segment.start, segment.end, turn.start, turn.end)
        if shared > 0:
            totals[turn.speaker] = totals.get(turn.speaker, 0.0) + shared
    if totals:
        return max(totals, key=lambda speaker: totals[speaker])

    middle = (segment.start + segment.end) / 2
    nearest = min(turns, key=lambda turn: min(abs(turn.start - middle), abs(turn.end - middle)))
    return nearest.speaker


def align(segments: Iterable[Segment], turns: Sequence[Turn]) -> list[Utterance]:
    """Attach a speaker to every segment and merge runs from the same speaker.

    Every segment appears in the output exactly once, so no text is lost or
    repeated, however the speaker turns and the text segments line up.
    """
    utterances: list[Utterance] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        speaker = assign_speaker(segment, turns)
        if utterances and utterances[-1].speaker == speaker:
            previous = utterances[-1]
            utterances[-1] = Utterance(speaker, previous.start, segment.end, f"{previous.text} {text}")
        else:
            utterances.append(Utterance(speaker, segment.start, segment.end, text))
    return utterances


# --- Output ------------------------------------------------------------------


def timestamp(seconds: float, separator: str = ".") -> str:
    """Format seconds as ``HH:MM:SS.mmm``, or ``HH:MM:SS,mmm`` for SRT."""
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def render(utterances: Sequence[Utterance], fmt: str) -> str:
    """Render the utterances as plain text, SRT subtitles or JSON."""
    if fmt == "txt":
        return "".join(
            f"[{timestamp(u.start)} - {timestamp(u.end)}] {u.speaker}: {u.text}\n" for u in utterances
        )
    if fmt == "srt":
        blocks = [
            f"{index}\n{timestamp(u.start, ',')} --> {timestamp(u.end, ',')}\n{u.speaker}: {u.text}\n"
            for index, u in enumerate(utterances, start=1)
        ]
        return "\n".join(blocks)
    if fmt == "json":
        return json.dumps([asdict(u) for u in utterances], indent=2, ensure_ascii=False) + "\n"
    raise ValueError(f"unknown format {fmt!r}, expected one of {', '.join(FORMATS)}")


# --- Pipeline steps ----------------------------------------------------------


def extract_audio(source: Path, destination: Path) -> None:
    """Write the audio track of ``source`` as 16 kHz mono WAV, the input both models expect."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH; install it first")
    command = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(destination),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read {source}: {result.stderr.strip()}")


def pick_device(requested: str) -> str:
    """Resolve ``auto`` to ``cuda`` when a GPU is available, else ``cpu``."""
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def transcribe(audio: Path, model_name: str, language: str | None, device: str) -> list[Segment]:
    """Run Whisper and return its timed segments."""
    import whisper

    model = whisper.load_model(model_name, device=device)
    result: dict[str, Any] = model.transcribe(str(audio), language=language, fp16=device == "cuda")
    return [Segment(float(s["start"]), float(s["end"]), str(s["text"])) for s in result["segments"]]


def diarize(audio: Path, token: str, device: str, speakers: int | None) -> list[Turn]:
    """Run pyannote and return its speaker turns."""
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            f"could not load {DIARIZATION_MODEL}; accept its terms on Hugging Face "
            "and check that the token is valid"
        )
    pipeline.to(torch.device(device))
    options = {"num_speakers": speakers} if speakers else {}
    annotation = pipeline(str(audio), **options)
    return [
        Turn(float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


# --- Command line ------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe an audio or video file and label each line with its speaker.",
    )
    parser.add_argument("input", type=Path, help="audio or video file, anything ffmpeg can read")
    parser.add_argument(
        "-o", "--output", type=Path, help="output file (default: the input path with the format's extension)"
    )
    parser.add_argument("-f", "--format", choices=FORMATS, default="txt", help="output format (default: txt)")
    parser.add_argument(
        "-m", "--model", default="medium",
        help="Whisper model: tiny, base, small, medium, large-v3, turbo (default: medium)",
    )  # fmt: skip
    parser.add_argument("-l", "--language", help="spoken language, such as en or de (default: detect)")
    parser.add_argument("-s", "--speakers", type=int, help="number of speakers, if known")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="where to run the models (default: auto)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = args.input.with_suffix(f".{args.format}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    if not args.input.is_file():
        log.error("input file not found: %s", args.input)
        return 2
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("set HF_TOKEN to a Hugging Face access token; see the README")
        return 2

    try:
        device = pick_device(args.device)
        with tempfile.TemporaryDirectory(prefix="transcribe-") as workdir:
            audio = Path(workdir) / "audio.wav"
            log.info("Extracting audio from %s", args.input)
            extract_audio(args.input, audio)
            log.info("Transcribing with Whisper %s on %s", args.model, device)
            segments = transcribe(audio, args.model, args.language, device)
            log.info("Identifying speakers")
            turns = diarize(audio, token, device, args.speakers)
    except RuntimeError as error:
        log.error("%s", error)
        return 1

    utterances = align(segments, turns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(utterances, args.format), encoding="utf-8")
    speakers = len({u.speaker for u in utterances})
    log.info("Wrote %d lines from %d speaker(s) to %s", len(utterances), speakers, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
