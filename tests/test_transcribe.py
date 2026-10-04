"""Tests for the parts that do not need the models: alignment, output and the CLI."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from transcribe import (
    Segment,
    Turn,
    Utterance,
    align,
    assign_speaker,
    extract_audio,
    main,
    parse_args,
    render,
    timestamp,
)

TURNS = [Turn(0.0, 5.0, "SPEAKER_00"), Turn(5.0, 9.0, "SPEAKER_01"), Turn(9.0, 12.0, "SPEAKER_00")]


def test_segment_inside_one_turn_gets_that_speaker() -> None:
    assert assign_speaker(Segment(1.0, 3.0, "hi"), TURNS) == "SPEAKER_00"


def test_segment_spanning_two_turns_goes_to_the_larger_overlap() -> None:
    # 1 s with speaker 00, 3 s with speaker 01.
    assert assign_speaker(Segment(4.0, 8.0, "hi"), TURNS) == "SPEAKER_01"


def test_overlap_is_summed_per_speaker() -> None:
    """Two short turns from one speaker can outweigh one longer turn from another."""
    turns = [Turn(0.0, 1.5, "A"), Turn(1.5, 3.5, "B"), Turn(3.5, 5.0, "A")]
    assert assign_speaker(Segment(0.0, 5.0, "x"), turns) == "A"


def test_segment_in_a_gap_goes_to_the_nearest_turn() -> None:
    turns = [Turn(0.0, 2.0, "A"), Turn(10.0, 12.0, "B")]
    assert assign_speaker(Segment(8.0, 9.0, "x"), turns) == "B"


def test_no_turns_means_unknown_speaker() -> None:
    assert assign_speaker(Segment(0.0, 1.0, "x"), []) == "UNKNOWN"


def test_consecutive_segments_from_one_speaker_are_merged() -> None:
    segments = [Segment(0.0, 2.0, " Hello"), Segment(2.0, 4.0, " there."), Segment(5.5, 8.0, " Hi!")]
    assert align(segments, TURNS) == [
        Utterance("SPEAKER_00", 0.0, 4.0, "Hello there."),
        Utterance("SPEAKER_01", 5.5, 8.0, "Hi!"),
    ]


def test_every_segment_appears_exactly_once() -> None:
    """A segment straddling a turn boundary must not be dropped or repeated."""
    segments = [Segment(0.0, 4.0, "one"), Segment(4.0, 10.0, "two"), Segment(10.0, 12.0, "three")]
    words = " ".join(u.text for u in align(segments, TURNS)).split()
    assert sorted(words) == sorted(["one", "two", "three"])


def test_empty_segments_are_skipped() -> None:
    assert align([Segment(0.0, 1.0, "   ")], TURNS) == []


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "00:00:00.000"), (61.5, "00:01:01.500"), (3725.042, "01:02:05.042"), (59.9996, "00:01:00.000")],
)
def test_timestamp(seconds: float, expected: str) -> None:
    assert timestamp(seconds) == expected


UTTERANCES = [Utterance("SPEAKER_00", 0.0, 2.5, "Hello."), Utterance("SPEAKER_01", 3.0, 4.25, "Grüß dich.")]


def test_render_txt() -> None:
    assert render(UTTERANCES, "txt") == (
        "[00:00:00.000 - 00:00:02.500] SPEAKER_00: Hello.\n"
        "[00:00:03.000 - 00:00:04.250] SPEAKER_01: Grüß dich.\n"
    )


def test_render_srt() -> None:
    assert render(UTTERANCES, "srt") == (
        "1\n00:00:00,000 --> 00:00:02,500\nSPEAKER_00: Hello.\n"
        "\n"
        "2\n00:00:03,000 --> 00:00:04,250\nSPEAKER_01: Grüß dich.\n"
    )


def test_render_json_keeps_non_ascii_text() -> None:
    rendered = render(UTTERANCES, "json")
    assert "Grüß" in rendered
    assert json.loads(rendered)[1] == {
        "speaker": "SPEAKER_01",
        "start": 3.0,
        "end": 4.25,
        "text": "Grüß dich.",
    }


def test_render_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="unknown format"):
        render(UTTERANCES, "docx")


def test_default_output_follows_the_input_and_format() -> None:
    assert parse_args(["talks/interview.mp4"]).output == Path("talks/interview.txt")
    assert parse_args(["talks/interview.mp4", "-f", "srt"]).output == Path("talks/interview.srt")
    assert parse_args(["a.mp4", "-o", "out/b.txt"]).output == Path("out/b.txt")


def test_missing_input_fails_before_loading_anything(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing.mp4")]) == 2


def test_missing_token_fails_before_loading_anything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "talk.wav"
    source.write_bytes(b"")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert main([str(source)]) == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_extract_audio_writes_16khz_mono_wav(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    generate = (
        "ffmpeg -nostdin -loglevel error"
        " -f lavfi -i sine=frequency=440:duration=1:sample_rate=44100"
        " -f lavfi -i color=size=64x64:duration=1 -ac 2 -shortest"
    )
    subprocess.run([*generate.split(), str(video)], check=True)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"stale file from an earlier run")

    extract_audio(video, audio)

    probe_command = "ffprobe -v error -show_entries stream=sample_rate,channels -of csv=p=0"
    probe = subprocess.run([*probe_command.split(), str(audio)], capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "16000,1"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_extract_audio_reports_an_unreadable_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_text("not a video")
    with pytest.raises(RuntimeError, match="could not read"):
        extract_audio(broken, tmp_path / "audio.wav")
