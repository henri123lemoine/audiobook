"""Throughput helpers for estimation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.audio.pipeline import preprocess_segments
from src.book.catalog import load_book
from src.book.types import Segment


def collect_processed_segments(book_id: str, limit: int | None = None) -> list[Segment]:
    """Collect preprocessed segments in reading order."""
    book = load_book(book_id)
    segments: list[Segment] = []
    for part in book.parts:
        for chapter in part.chapters:
            processed = preprocess_segments(chapter.segments)
            for segment in processed:
                segments.append(segment)
                if limit is not None and len(segments) >= limit:
                    return segments
    return segments


def normalize_gpu_type(name: str) -> str:
    cleaned = name.upper()
    for token in ("NVIDIA", "GEFORCE", "TESLA"):
        cleaned = cleaned.replace(token, "")
    cleaned = re.sub(r"[^A-Z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "UNKNOWN"


def detect_gpu_type(device: str) -> tuple[str, str]:
    """Return (normalized_gpu_type, raw_gpu_name)."""
    if device == "cpu":
        return "CPU", "CPU"
    if device != "cuda":
        return normalize_gpu_type(device), device

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    raw_name = torch.cuda.get_device_name()
    return normalize_gpu_type(raw_name), raw_name


def _load_manifest_entries(segments_dir: Path) -> dict[int, dict]:
    manifest_files = sorted(segments_dir.glob("manifest_*.json"))
    if not manifest_files:
        raise FileNotFoundError(f"No manifest files found in {segments_dir}")

    entries: dict[int, dict] = {}
    for manifest_path in manifest_files:
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON: {manifest_path}") from exc

        for entry in manifest:
            idx = entry.get("global_index")
            if idx is None:
                continue
            entries[idx] = entry

    if not entries:
        raise FileNotFoundError(f"No valid manifest entries in {segments_dir}")
    return entries


def summarize_throughput(
    book_id: str,
    gpu_type: str,
    segments_dir: Path | None = None,
) -> dict:
    """Summarize throughput from manifest metrics."""
    segments_dir = segments_dir or Path("books") / book_id / "audio" / "segments"
    entries = _load_manifest_entries(segments_dir)

    normalized_input = normalize_gpu_type(gpu_type)
    total_chars = 0
    total_audio_seconds = 0.0
    total_wall_seconds = 0.0
    sample_count = 0
    raw_names: set[str] = set()

    for entry in entries.values():
        entry_gpu = entry.get("gpu_type")
        raw_name = entry.get("gpu_name_raw")
        entry_norm = normalize_gpu_type(entry_gpu) if entry_gpu else None
        raw_norm = normalize_gpu_type(raw_name) if raw_name else None
        if entry_norm and entry_norm != normalized_input and raw_norm != normalized_input:
            continue
        if entry_norm is None and raw_norm is not None and raw_norm != normalized_input:
            continue
        if entry_norm is None and raw_norm is None:
            continue

        chars = entry.get("chars")
        audio_seconds = entry.get("audio_seconds")
        wall_seconds = entry.get("wall_seconds")
        if chars is None or audio_seconds is None or wall_seconds is None:
            continue

        if chars <= 0 or audio_seconds <= 0 or wall_seconds <= 0:
            continue

        total_chars += int(chars)
        total_audio_seconds += float(audio_seconds)
        total_wall_seconds += float(wall_seconds)
        sample_count += 1

        if raw_name:
            raw_names.add(raw_name)

    if sample_count == 0:
        raise RuntimeError(
            f"No throughput samples found for {gpu_type} in {segments_dir}. "
            "Run a short generate on that GPU (e.g. --limit 10) to collect metrics."
        )

    if total_chars <= 0 or total_audio_seconds <= 0 or total_wall_seconds <= 0:
        raise RuntimeError("Throughput samples are invalid")

    return {
        "gpu_type": gpu_type,
        "gpu_name_raw": sorted(raw_names) if raw_names else [],
        "segments": sample_count,
        "total_chars": total_chars,
        "total_audio_seconds": total_audio_seconds,
        "total_wall_seconds": total_wall_seconds,
        "audio_seconds_per_char": total_audio_seconds / total_chars,
        "audio_seconds_per_wall_second": total_audio_seconds / total_wall_seconds,
        "chars_per_wall_second": total_chars / total_wall_seconds,
        "segments_dir": str(segments_dir),
        "book_id": book_id,
    }
