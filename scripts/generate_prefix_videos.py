#!/usr/bin/env python3

"""Generate prefix videos (long/medium/short) for occlusion query frames."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


DEFAULT_BOUNDARIES = Path("/opt/data/private/dataset/mask_results/occlusion_boundaries_original_frames_nonzero.json")
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/dataset/prefix_videos_from_start")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundaries-json",
        type=Path,
        default=DEFAULT_BOUNDARIES,
        help="Path to occlusion_boundaries_original_frames.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory for generated prefix videos and metadata",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rendering when output clip already exists",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise errors for missing inputs or invalid frame ranges",
    )
    return parser.parse_args()


def frame_base_start(start: int) -> int:
    """Convert annotation start into 0-based start frame used by extraction."""

    if start <= 0:
        return start
    return start - 1


def resolve_original_video_path(dataset_root: Path, original_video: str) -> Path:
    video_path = Path(original_video)
    if video_path.is_absolute():
        return video_path
    return (dataset_root / video_path).resolve()


def build_ffmpeg_cmd(input_video: Path, output_video: Path, start_frame: int, end_frame_inclusive: int) -> list[str]:
    # ffmpeg trim uses an exclusive end frame.
    end_frame_exclusive = end_frame_inclusive + 1
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame_exclusive},setpts=PTS-STARTPTS"
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]


def safe_stem(name: str) -> str:
    # Keep filenames deterministic and shell-friendly.
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "segment_name",
        "segment_video",
        "original_video",
        "frame_kind",
        "question_field",
        "question_frame_original",
        "segment_start_raw",
        "segment_start_0based",
        "segment_end_raw",
        "clip_type",
        "clip_start_frame",
        "clip_end_frame_inclusive",
        "output_video",
        "status",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def generate_for_question_frame(
    *,
    ffmpeg_input: Path,
    out_base_dir: Path,
    file_stem: str,
    question_frame: int,
    subtask_start_0based: int,
    row_common: dict,
    skip_existing: bool,
    strict: bool,
) -> list[dict]:
    long_start = 0
    short_start = subtask_start_0based
    medium_start = (long_start + short_start) // 2

    clip_specs = [
        ("long", long_start),
        ("medium", medium_start),
        ("short", short_start),
    ]

    rows: list[dict] = []
    for clip_type, clip_start in clip_specs:
        output_path = out_base_dir / clip_type / f"{file_stem}__{clip_type}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        row = {
            **row_common,
            "clip_type": clip_type,
            "clip_start_frame": clip_start,
            "clip_end_frame_inclusive": question_frame,
            "output_video": str(output_path),
            "status": "pending",
            "error": "",
        }

        if clip_start < 0:
            row["status"] = "skipped"
            row["error"] = f"negative start frame {clip_start}"
            if strict:
                raise ValueError(f"Negative start frame for {row_common['segment_name']} ({clip_type})")
            rows.append(row)
            continue

        if question_frame < clip_start:
            row["status"] = "skipped"
            row["error"] = f"question frame {question_frame} earlier than start {clip_start}"
            if strict:
                raise ValueError(f"Invalid range for {row_common['segment_name']} ({clip_type})")
            rows.append(row)
            continue

        if skip_existing and output_path.exists():
            row["status"] = "skipped_existing"
            rows.append(row)
            continue

        cmd = build_ffmpeg_cmd(ffmpeg_input, output_path, clip_start, question_frame)
        try:
            subprocess.run(cmd, check=True)
            row["status"] = "ok"
        except subprocess.CalledProcessError as exc:
            row["status"] = "failed"
            row["error"] = f"ffmpeg exit code {exc.returncode}"
            if strict:
                raise

        rows.append(row)

    return rows


def main() -> None:
    args = parse_args()

    with args.boundaries_json.open("r", encoding="utf-8") as handle:
        boundaries = json.load(handle)

    dataset_root = args.boundaries_json.parent.parent
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for segment_name, item in boundaries.items():
        original_video_path = resolve_original_video_path(dataset_root, item["original_video"])
        if not original_video_path.exists():
            message = f"missing original video: {original_video_path}"
            if args.strict:
                raise FileNotFoundError(message)

            # Record skipped entries for both frame kinds when input video is absent.
            for frame_kind, question_field in (
                ("unoccluded", "last_unoccluded_frame_original"),
                ("occluded", "last_occluded_frame_original"),
            ):
                question_frame = item.get(question_field)
                if question_frame is None and frame_kind == "occluded":
                    question_frame = item.get("last_occluded_frame_origiinal")
                if question_frame is None:
                    continue

                all_rows.append(
                    {
                        "segment_name": segment_name,
                        "segment_video": item.get("segment_video"),
                        "original_video": str(original_video_path),
                        "frame_kind": frame_kind,
                        "question_field": question_field,
                        "question_frame_original": question_frame,
                        "segment_start_raw": item.get("start"),
                        "segment_start_0based": frame_base_start(int(item["start"])),
                        "segment_end_raw": item.get("end"),
                        "clip_type": "",
                        "clip_start_frame": "",
                        "clip_end_frame_inclusive": "",
                        "output_video": "",
                        "status": "skipped",
                        "error": message,
                    }
                )
            continue

        segment_start_raw = int(item["start"])
        segment_start_0based = frame_base_start(segment_start_raw)

        frame_jobs = [
            ("unoccluded", "last_unoccluded_frame_original", item.get("last_unoccluded_frame_original")),
            (
                "occluded",
                "last_occluded_frame_original",
                item.get("last_occluded_frame_original", item.get("last_occluded_frame_origiinal")),
            ),
        ]

        for frame_kind, question_field, question_frame in frame_jobs:
            if question_frame is None:
                continue

            question_frame_int = int(question_frame)
            frame_output_root = args.output_root / frame_kind
            file_stem = safe_stem(Path(segment_name).stem) + f"__{frame_kind}"

            row_common = {
                "segment_name": segment_name,
                "segment_video": item.get("segment_video"),
                "original_video": str(original_video_path),
                "frame_kind": frame_kind,
                "question_field": question_field,
                "question_frame_original": question_frame_int,
                "segment_start_raw": segment_start_raw,
                "segment_start_0based": segment_start_0based,
                "segment_end_raw": item.get("end"),
            }

            rows = generate_for_question_frame(
                ffmpeg_input=original_video_path,
                out_base_dir=frame_output_root,
                file_stem=file_stem,
                question_frame=question_frame_int,
                subtask_start_0based=segment_start_0based,
                row_common=row_common,
                skip_existing=args.skip_existing,
                strict=args.strict,
            )
            all_rows.extend(rows)

    meta_json = args.output_root / "prefix_videos_meta.json"
    meta_csv = args.output_root / "prefix_videos_meta.csv"

    with meta_json.open("w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, ensure_ascii=False, indent=2)

    write_csv(meta_csv, all_rows)

    total = len(all_rows)
    ok = sum(1 for row in all_rows if row["status"] == "ok")
    skipped = sum(1 for row in all_rows if row["status"].startswith("skipped"))
    failed = sum(1 for row in all_rows if row["status"] == "failed")

    print(f"Total clip records: {total}")
    print(f"OK: {ok} | Skipped: {skipped} | Failed: {failed}")
    print(f"Metadata JSON: {meta_json}")
    print(f"Metadata CSV: {meta_csv}")


if __name__ == "__main__":
    main()
