"""Combine generated segments into chapters, parts, and a final audiobook."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from pydub import AudioSegment

SEGMENT_SILENCE_MS = 500
CHAPTER_SILENCE_MS = 1000
PART_SILENCE_MS = 1500


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
            if idx in entries and entries[idx] != entry:
                logger.warning(
                    f"Duplicate manifest entry for {idx} in {manifest_path.name}; using latest"
                )
            entries[idx] = entry

    if not entries:
        raise FileNotFoundError(f"No valid manifest entries in {segments_dir}")
    return entries


def _combine_audio_files(files: list[Path], output_path: Path, silence_ms: int) -> None:
    if not files:
        raise FileNotFoundError(f"No audio files to combine for {output_path}")

    silence = AudioSegment.silent(duration=silence_ms)
    combined = AudioSegment.from_file(files[0])
    for audio_path in files[1:]:
        combined += silence + AudioSegment.from_file(audio_path)
    combined.export(output_path, format="mp3")


def combine_from_segments(book_id: str, output_dir: Path | None = None) -> Path:
    """Combine segments from manifest files into final audiobook output."""
    output_dir = output_dir or Path("books") / book_id / "audio"
    segments_dir = output_dir / "segments"

    entries = _load_manifest_entries(segments_dir)

    chapters: dict[tuple[int, int], list[tuple[int, Path]]] = {}
    missing_files: list[Path] = []

    for idx, entry in entries.items():
        file_name = entry.get("file")
        if not file_name:
            continue

        part = int(entry.get("part", 1))
        chapter = int(entry.get("chapter", 1))
        file_path = segments_dir / file_name

        if not file_path.exists():
            missing_files.append(file_path)
            continue

        chapters.setdefault((part, chapter), []).append((idx, file_path))

    if missing_files:
        sample = ", ".join(p.name for p in missing_files[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_files)} segment files in {segments_dir}: {sample}"
        )

    if not chapters:
        raise FileNotFoundError(f"No chapter data found in manifests at {segments_dir}")

    part_numbers = sorted({part for part, _ in chapters.keys()})
    for part_num in part_numbers:
        part_dir = output_dir / f"partie_{part_num}"
        part_dir.mkdir(parents=True, exist_ok=True)

        chapter_numbers = sorted({ch for p, ch in chapters.keys() if p == part_num})
        for chapter_num in chapter_numbers:
            files = chapters[(part_num, chapter_num)]
            ordered_files = [path for _, path in sorted(files, key=lambda item: item[0])]
            chapter_path = part_dir / f"chapitre_{chapter_num}_full.mp3"
            _combine_audio_files(ordered_files, chapter_path, SEGMENT_SILENCE_MS)

    part_paths: list[Path] = []
    for part_num in part_numbers:
        part_dir = output_dir / f"partie_{part_num}"
        chapter_files = sorted(
            part_dir.glob("chapitre_*_full.mp3"),
            key=lambda f: int(f.stem.split("_")[1]),
        )
        if not chapter_files:
            continue
        part_path = output_dir / f"partie_{part_num}_complete.mp3"
        _combine_audio_files(chapter_files, part_path, CHAPTER_SILENCE_MS)
        part_paths.append(part_path)

    if not part_paths:
        raise FileNotFoundError(f"No part audio files found in {output_dir}")

    final_path = output_dir / "audiobook_complete.mp3"
    _combine_audio_files(part_paths, final_path, PART_SILENCE_MS)
    return final_path
