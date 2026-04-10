#!/usr/bin/env python3

"""Generate prefix videos from frame 0 to occlusion query frames."""

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
        help="Skip rendering if output clip already exists",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise errors for missing inputs or invalid frame values",
    )
    return parser.parse_args()


def resolve_original_video_path(dataset_root: Path, original_video: str) -> Path:
    video_path = Path(original_video)
    if video_path.is_absolute():
        return video_path
    return (dataset_root / video_path).resolve()


def build_ffmpeg_cmd(input_video: Path, output_video: Path, end_frame_inclusive: int) -> list[str]:
    # ffmpeg trim uses an exclusive end frame.
    end_frame_exclusive = end_frame_inclusive + 1
    vf = f"trim=start_frame=0:end_frame={end_frame_exclusive},setpts=PTS-STARTPTS"
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
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "segment_name",
        "segment_video",
        "original_video",
        "frame_kind",
        "question_field",
        "question_frame_original",
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


def render_one_prefix(
    *,
    original_video_path: Path,
    output_path: Path,
    question_frame_original: int,
    skip_existing: bool,
    strict: bool,
    row: dict,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if question_frame_original < 0:
        row["status"] = "skipped"
        row["error"] = f"negative question frame {question_frame_original}"
        if strict:
            raise ValueError(f"Negative question frame: {question_frame_original}")
        return row

    if skip_existing and output_path.exists():
        row["status"] = "skipped_existing"
        return row

    cmd = build_ffmpeg_cmd(original_video_path, output_path, question_frame_original)
    try:
        subprocess.run(cmd, check=True)
        row["status"] = "ok"
    except subprocess.CalledProcessError as exc:
        row["status"] = "failed"
        row["error"] = f"ffmpeg exit code {exc.returncode}"
        if strict:
            raise

    return row


def main() -> None:
    args = parse_args()

    with args.boundaries_json.open("r", encoding="utf-8") as handle:
        boundaries = json.load(handle)

    dataset_root = args.boundaries_json.parent.parent
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for segment_name, item in boundaries.items():
        original_video_path = resolve_original_video_path(dataset_root, str(item["original_video"]))
        if not original_video_path.exists():
            message = f"missing original video: {original_video_path}"
            if args.strict:
                raise FileNotFoundError(message)

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
                        "question_frame_original": int(question_frame),
                        "clip_start_frame": 0,
                        "clip_end_frame_inclusive": int(question_frame),
                        "output_video": "",
                        "status": "skipped",
                        "error": message,
                    }
                )
            continue

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
            frame_dir = args.output_root / frame_kind
            file_stem = safe_stem(Path(segment_name).stem)
            output_path = frame_dir / f"{file_stem}__{frame_kind}.mp4"

            row = {
                "segment_name": segment_name,
                "segment_video": item.get("segment_video"),
                "original_video": str(original_video_path),
                "frame_kind": frame_kind,
                "question_field": question_field,
                "question_frame_original": question_frame_int,
                "clip_start_frame": 0,
                "clip_end_frame_inclusive": question_frame_int,
                "output_video": str(output_path),
                "status": "pending",
                "error": "",
            }

            row = render_one_prefix(
                original_video_path=original_video_path,
                output_path=output_path,
                question_frame_original=question_frame_int,
                skip_existing=args.skip_existing,
                strict=args.strict,
                row=row,
            )
            all_rows.append(row)

    meta_json = args.output_root / "prefix_videos_from_start_meta.json"
    meta_csv = args.output_root / "prefix_videos_from_start_meta.csv"

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
