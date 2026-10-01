from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility shim: patch av.open() to tolerate metadata_errors kwarg
# removed in av>=12, but still passed by faster-whisper 1.x
# ---------------------------------------------------------------------------
try:
    import av as _av_mod
    _orig_av_open = _av_mod.open

    def _av_open_compat(*args, **kwargs):
        kwargs.pop("metadata_errors", None)  # removed in av>=12
        return _orig_av_open(*args, **kwargs)

    _av_mod.open = _av_open_compat
except Exception:
    pass  # av not installed yet or already compatible – no action needed


# ---------------------------------------------------------------------------
# Data models (public API – unchanged)
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionSegment:
    """A single timed segment from Whisper."""
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Structured result returned by transcribe_file()."""
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: Optional[str] = None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings:
    whisper_model_default: str = "base"
    transcription_device: str = "auto"   # "auto" | "cpu" | "cuda"
    chunk_duration_seconds: int = 600    # 10-minute chunks by default
    chunk_overlap_seconds: int = 5       # 5-second overlap to avoid cut-off words
    compute_type_cpu: str = "int8"       # int8 quantisation → much lower RAM on CPU
    compute_type_gpu: str = "float16"

settings = Settings()


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _ffmpeg_bin() -> str:
    """Return the ffmpeg binary path (handles imageio-ffmpeg bundle)."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(
        "FFmpeg not found. Install it system-wide or add imageio-ffmpeg to requirements."
    )


def _check_ffmpeg_available() -> bool:
    """Check whether ffmpeg is available on this system."""
    try:
        ffmpeg = _ffmpeg_bin()
        subprocess.run([ffmpeg, "-version"], capture_output=True, timeout=5, check=True)
        return True
    except Exception as exc:
        logger.warning(f"FFmpeg check failed: {exc}")
        return False


def _probe_duration(media_path: Path) -> float:
    """Return media duration in seconds using ffprobe (or ffmpeg -i)."""
    try:
        ffmpeg = _ffmpeg_bin()
        ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
        if not shutil.which(ffprobe) and not Path(ffprobe).exists():
            ffprobe = ffmpeg  # fall back to ffmpeg -i stderr
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_format", str(media_path)],
            capture_output=True, text=True, timeout=30,
        )
        import json
        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0))
    except Exception:
        # Fallback: parse ffmpeg -i stderr
        try:
            ffmpeg = _ffmpeg_bin()
            r = subprocess.run(
                [ffmpeg, "-i", str(media_path)],
                capture_output=True, text=True, timeout=30,
            )
            import re
            m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
            if m:
                h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                return h * 3600 + mn * 60 + s
        except Exception:
            pass
    return 0.0


def _extract_audio_to_wav(media_path: Path, out_path: Path) -> Path:
    """Extract audio from a media file to a 16 kHz mono WAV at out_path."""
    ffmpeg = _ffmpeg_bin()

    if out_path.exists():
        out_path.unlink()

    common_tail = [
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(out_path),
    ]

    strategies = [
        ([ffmpeg, "-i", str(media_path), "-map", "0:a:0"] + common_tail,
         "Explicit audio map"),
        ([ffmpeg, "-i", str(media_path), "-vn"] + common_tail,
         "Strip video"),
    ]

    last_error: Optional[str] = None
    for cmd, name in strategies:
        try:
            logger.info(f"Audio extraction strategy: {name}")
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 44:
                logger.info(f"Audio extracted → {out_path}")
                return out_path
            last_error = (r.stderr or "")[-400:]
            logger.warning(f"Strategy '{name}' failed: {last_error}")
            if out_path.exists():
                out_path.unlink()
        except subprocess.TimeoutExpired:
            logger.warning(f"Strategy '{name}' timed out")
            last_error = "timeout"
            if out_path.exists():
                out_path.unlink()
        except Exception as exc:
            logger.warning(f"Strategy '{name}' raised: {exc}")
            last_error = str(exc)
            if out_path.exists():
                out_path.unlink()

    raise RuntimeError(f"Audio extraction failed. Last error: {last_error}")


def _extract_audio_chunk(
    audio_path: Path,
    chunk_path: Path,
    start_sec: float,
    duration_sec: float,
) -> Path:
    """
    Extract a sub-segment of an already-extracted WAV file to chunk_path.
    Uses FFmpeg seeking on the audio file directly (fast).
    """
    ffmpeg = _ffmpeg_bin()
    cmd = [
        ffmpeg,
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", str(audio_path),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(chunk_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not chunk_path.exists() or chunk_path.stat().st_size <= 44:
        raise RuntimeError(
            f"Chunk extraction failed (start={start_sec}s): {(r.stderr or '')[-300:]}"
        )
    return chunk_path


def _plan_chunks(
    total_duration: float,
    chunk_duration: int,
    overlap: int,
) -> List[Tuple[float, float, float]]:
    """
    Return list of (start, end, offset) tuples for each chunk.
    overlap: seconds of overlap between adjacent chunks (to avoid word cut-off).
    """
    chunks = []
    pos = 0.0
    while pos < total_duration:
        end = min(pos + chunk_duration + overlap, total_duration)
        chunks.append((pos, end, pos))   # (chunk_start, chunk_end, time_offset)
        if end >= total_duration:
            break
        pos += chunk_duration  # next chunk starts at pos + chunk_duration (no overlap advance)
    return chunks


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _detect_device() -> Tuple[str, str]:
    """Return (device, compute_type) appropriate for this machine."""
    force = settings.transcription_device
    try:
        import torch  # type: ignore
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if force == "cuda":
        device = "cuda" if cuda_ok else "cpu"
    elif force == "cpu":
        device = "cpu"
    else:  # auto
        device = "cuda" if cuda_ok else "cpu"

    compute_type = (
        settings.compute_type_gpu if device == "cuda" else settings.compute_type_cpu
    )
    return device, compute_type


# ---------------------------------------------------------------------------
# Model loader — cached at Streamlit layer (see transcribe_file wrapper below)
# ---------------------------------------------------------------------------

def _load_faster_whisper_model(model_name: str, device: str, compute_type: str):
    """Load a faster-whisper WhisperModel. Called once and cached by the caller."""
    from faster_whisper import WhisperModel  # type: ignore

    logger.info(f"Loading faster-whisper model='{model_name}' device={device} compute={compute_type}")
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=2,          # Streamlit Community Cloud has 2 cores
        num_workers=1,
        download_root=os.path.expanduser("~/.cache/faster_whisper"),
    )
    logger.info("faster-whisper model loaded OK")
    return model


# ---------------------------------------------------------------------------
# Chunk transcription
# ---------------------------------------------------------------------------

def _transcribe_chunk_faster_whisper(
    model,
    chunk_path: Path,
    time_offset: float,
    language: Optional[str] = None,
) -> Tuple[List[TranscriptionSegment], Optional[str]]:
    """
    Transcribe a single audio chunk with faster-whisper.
    Returns (segments_with_offset, detected_language).
    """
    segments_gen, info = model.transcribe(
        str(chunk_path),
        language=language,
        beam_size=5,
        word_timestamps=False,
        vad_filter=True,        # skip silence → faster, less hallucination
        vad_parameters={"min_silence_duration_ms": 500},
    )

    detected_lang = info.language if hasattr(info, "language") else None
    segments: List[TranscriptionSegment] = []

    for seg in segments_gen:
        segments.append(TranscriptionSegment(
            start=seg.start + time_offset,
            end=seg.end + time_offset,
            text=seg.text.strip(),
        ))

    return segments, detected_lang


# ---------------------------------------------------------------------------
# Overlap deduplication
# ---------------------------------------------------------------------------

def _deduplicate_segments(
    segments: List[TranscriptionSegment],
    overlap: int,
) -> List[TranscriptionSegment]:
    """
    Remove duplicate/overlapping segments that arise at chunk boundaries.
    Strategy: if a segment's start time falls inside the previous chunk's
    overlap zone AND its text is identical (or very similar) to an already-
    recorded segment, drop it.
    """
    if not segments or overlap == 0:
        return segments

    deduped: List[TranscriptionSegment] = []
    seen_texts: set[str] = set()

    for seg in segments:
        key = seg.text.lower().strip()
        if key and key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append(seg)

    return deduped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_file(
    media_path: Path,
    model_name: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    chunk_minutes: int = 10,
) -> TranscriptionResult:
    """
    Transcribe a media file to text using faster-whisper, processing audio
    in chunks to keep RAM usage low for long videos.

    Parameters
    ----------
    media_path : Path
        Path to the video/audio file on disk.
    model_name : str, optional
        Whisper model size: "tiny", "base", "small", "medium". Defaults to "base".
    progress_callback : callable, optional
        Called with (percent: int, message: str) during processing.
    chunk_minutes : int, optional
        Chunk duration in minutes. Default 10.

    Returns
    -------
    TranscriptionResult
    """
    if model_name is None:
        model_name = settings.whisper_model_default

    chunk_sec = chunk_minutes * 60
    overlap_sec = settings.chunk_overlap_seconds

    def _cb(pct: int, msg: str) -> None:
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    _cb(0, "Checking FFmpeg…")
    if not _check_ffmpeg_available():
        raise RuntimeError(
            "FFmpeg is required but was not found. "
            "Add imageio-ffmpeg to requirements.txt or install FFmpeg system-wide."
        )

    device, compute_type = _detect_device()
    _cb(5, f"Detected device: {device} | compute: {compute_type}")

    # We use a single temp directory for all intermediate files.
    # Everything is cleaned up in the finally block.
    tmp_dir = Path(tempfile.mkdtemp(prefix="transcriber_"))

    try:
        # ------------------------------------------------------------------ #
        # Step 1: Extract full audio to WAV (disk only, not loaded into RAM)  #
        # ------------------------------------------------------------------ #
        _cb(8, "Extracting audio from video…")
        wav_path = tmp_dir / "full_audio.wav"
        _extract_audio_to_wav(media_path, wav_path)

        # ------------------------------------------------------------------ #
        # Step 2: Probe duration and plan chunks                              #
        # ------------------------------------------------------------------ #
        _cb(12, "Analysing audio duration…")
        duration = _probe_duration(wav_path)
        if duration <= 0:
            duration = _probe_duration(media_path)
        if duration <= 0:
            # Last resort: assume at most 3 hours and let processing stop naturally
            logger.warning("Could not determine duration; assuming 10800s")
            duration = 10800.0

        chunks = _plan_chunks(duration, chunk_sec, overlap_sec)
        n_chunks = len(chunks)
        logger.info(f"Total duration: {duration:.1f}s → {n_chunks} chunk(s) of {chunk_sec}s")

        # ------------------------------------------------------------------ #
        # Step 3: Load model ONCE                                             #
        # ------------------------------------------------------------------ #
        _cb(15, f"Loading AI model ({model_name})…")

        # Try to get a cached model from Streamlit session (if available).
        # If called outside Streamlit (e.g., tests), just load directly.
        model = _get_cached_model(model_name, device, compute_type)

        # ------------------------------------------------------------------ #
        # Step 4: Process each chunk                                          #
        # ------------------------------------------------------------------ #
        all_segments: List[TranscriptionSegment] = []
        detected_language: Optional[str] = None

        # Reserve progress range 20–90 for chunks
        progress_start = 20
        progress_end = 90
        progress_per_chunk = (progress_end - progress_start) / max(n_chunks, 1)

        for idx, (start, end, offset) in enumerate(chunks):
            chunk_label = f"chunk {idx + 1}/{n_chunks}"
            _cb(
                int(progress_start + idx * progress_per_chunk),
                f"Transcribing {chunk_label} "
                f"({_fmt_time(start)} → {_fmt_time(end)})…",
            )

            chunk_path = tmp_dir / f"chunk_{idx:04d}.wav"
            try:
                _extract_audio_chunk(wav_path, chunk_path, start, end - start)

                segs, lang = _transcribe_chunk_faster_whisper(
                    model, chunk_path, offset, language=detected_language
                )

                if lang and not detected_language:
                    detected_language = lang

                all_segments.extend(segs)
                logger.info(f"{chunk_label}: {len(segs)} segment(s)")

            except Exception as chunk_exc:
                logger.error(f"{chunk_label} failed: {chunk_exc}")
                # Don't crash entirely — add an error placeholder segment
                all_segments.append(TranscriptionSegment(
                    start=start,
                    end=end,
                    text=f"[Transcription error for segment {idx + 1}: {chunk_exc}]",
                ))
            finally:
                if chunk_path.exists():
                    chunk_path.unlink()
                gc.collect()

        # ------------------------------------------------------------------ #
        # Step 5: Deduplicate overlap boundaries & combine                    #
        # ------------------------------------------------------------------ #
        _cb(91, "Combining transcript…")
        all_segments = _deduplicate_segments(all_segments, overlap_sec)
        all_segments.sort(key=lambda s: s.start)

        full_text = " ".join(s.text for s in all_segments if s.text).strip()

        _cb(95, "Generating subtitle files…")

        _cb(100, "Transcription complete!")
        return TranscriptionResult(
            text=full_text,
            segments=all_segments,
            language=detected_language,
        )

    finally:
        # Always clean up temp directory
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as cleanup_exc:
            logger.warning(f"Temp cleanup failed: {cleanup_exc}")


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------

def _get_cached_model(model_name: str, device: str, compute_type: str):
    """
    Return a faster-whisper model, using st.cache_resource when inside
    Streamlit, or a plain load otherwise.
    """
    try:
        import streamlit as st  # type: ignore

        # Build a unique cache key so switching models reloads correctly.
        cache_key = f"fw_model_{model_name}_{device}_{compute_type}"

        # st.cache_resource caches by function identity; we use a nested
        # function keyed on the parameters to get per-model caching.
        @st.cache_resource(show_spinner=False)
        def _cached(key: str, _model_name: str, _device: str, _compute_type: str):
            return _load_faster_whisper_model(_model_name, _device, _compute_type)

        return _cached(cache_key, model_name, device, compute_type)

    except Exception:
        # Outside Streamlit (tests / CLI) — load without caching
        return _load_faster_whisper_model(model_name, device, compute_type)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"