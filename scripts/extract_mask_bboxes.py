#!/usr/bin/env python3

"""Extract per-frame bounding boxes from binary mask videos.

The script scans mask videos in an input directory, computes the tight
bounding box of all non-zero mask pixels for every frame, and writes:

1. one JSON file per video with full per-frame details
2. one CSV summary for quick inspection
3. one JSON index that points to all per-video result files
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT_DIR = Path("/opt/data/private/dataset/masks")
DEFAULT_OUTPUT_DIR = Path("/opt/data/private/dataset/mask_results/bboxes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing mask videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write bbox results",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Pixel threshold used to binarize mask frames",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for videos recursively under the input directory",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose result JSON already exists",
    )
    return parser.parse_args()


def iter_video_paths(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(input_dir.rglob("*.mp4"))
    return sorted(input_dir.glob("*.mp4"))


def sanitize_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def frame_to_binary_mask(frame: np.ndarray, threshold: int) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return (frame > threshold).astype(np.uint8)


def compute_bbox(mask: np.ndarray) -> dict | None:
    points = cv2.findNonZero(mask)
    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    return {
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "bbox_xyxy": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
        "area_pixels": int(w * h),
        "mask_pixels": int(cv2.countNonZero(mask)),
    }


def analyze_video(video_path: Path, threshold: int) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frames: list[dict] = []
        frame_index = 0
        non_empty_frames = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            mask = frame_to_binary_mask(frame, threshold)
            bbox = compute_bbox(mask)

            if bbox is not None:
                non_empty_frames += 1

            frames.append(
                {
                    "frame_index": frame_index,
                    "has_mask": bbox is not None,
                    "bbox_xywh": None if bbox is None else bbox["bbox_xywh"],
                    "bbox_xyxy": None if bbox is None else bbox["bbox_xyxy"],
                    "area_pixels": 0 if bbox is None else bbox["area_pixels"],
                    "mask_pixels": 0 if bbox is None else bbox["mask_pixels"],
                }
            )
            frame_index += 1

        return {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": len(frames),
            "non_empty_frames": non_empty_frames,
            "empty_frames": len(frames) - non_empty_frames,
            "threshold": threshold,
            "frames": frames,
        }
    finally:
        capture.release()


def write_video_result(output_path: Path, result: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def write_summary_csv(output_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "video_name",
        "video_path",
        "result_json",
        "total_frames",
        "non_empty_frames",
        "empty_frames",
        "fps",
        "width",
        "height",
        "threshold",
        "status",
        "error",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = iter_video_paths(input_dir, args.recursive)
    summary_rows: list[dict] = []

    if not video_paths:
        print(f"No mp4 files found in {input_dir}")
        return

    for video_path in video_paths:
        video_stem = sanitize_stem(video_path.stem)
        result_json = output_dir / f"{video_stem}.json"

        row = {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "result_json": str(result_json),
            "total_frames": 0,
            "non_empty_frames": 0,
            "empty_frames": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "threshold": args.threshold,
            "status": "pending",
            "error": "",
        }

        if args.skip_existing and result_json.exists():
            row["status"] = "skipped_existing"
            summary_rows.append(row)
            continue

        try:
            result = analyze_video(video_path, args.threshold)
            write_video_result(result_json, result)

            row.update(
                {
                    "total_frames": result["total_frames"],
                    "non_empty_frames": result["non_empty_frames"],
                    "empty_frames": result["empty_frames"],
                    "fps": result["fps"],
                    "width": result["width"],
                    "height": result["height"],
                    "status": "ok",
                }
            )
        except Exception as exc:  # pragma: no cover - keep batch processing going
            row["status"] = "failed"
            row["error"] = str(exc)

        summary_rows.append(row)
        print(
            f"{row['status']}: {video_path.name} | frames={row['total_frames']} | "
            f"non_empty={row['non_empty_frames']}"
        )

    summary_json = output_dir / "mask_bboxes_index.json"
    summary_csv = output_dir / "mask_bboxes_summary.csv"

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, ensure_ascii=False, indent=2)

    write_summary_csv(summary_csv, summary_rows)

    total = len(summary_rows)
    ok = sum(1 for row in summary_rows if row["status"] == "ok")
    skipped = sum(1 for row in summary_rows if row["status"].startswith("skipped"))
    failed = sum(1 for row in summary_rows if row["status"] == "failed")

    print(f"Total videos: {total}")
    print(f"OK: {ok} | Skipped: {skipped} | Failed: {failed}")
    print(f"Index JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Per-video JSON directory: {output_dir}")


if __name__ == "__main__":
    main()