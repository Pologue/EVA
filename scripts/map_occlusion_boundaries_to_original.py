#!/usr/bin/env python3

"""Map segment-level occlusion boundary frames back to original video frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_ACTION_META = Path("/opt/data/private/dataset/action_videos.jsonl")
DEFAULT_OCCLUSION_JSON = Path("/opt/data/private/dataset/mask_results/occlusion_boundaries_nonzero.json")
DEFAULT_OUTPUT_JSON = Path("/opt/data/private/dataset/mask_results/occlusion_boundaries_original_frames_nonzero.json")
DEFAULT_OUTPUT_CSV = Path("/opt/data/private/dataset/mask_results/occlusion_boundaries_original_frames_nonzero.csv")


def load_action_meta(meta_path: Path) -> dict[str, dict]:
    meta_by_segment: dict[str, dict] = {}
    duplicates: set[str] = set()

    with meta_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            item = json.loads(line)
            segment_name = Path(item["segment_video"]).name

            if segment_name in meta_by_segment:
                duplicates.add(segment_name)

            item["_meta_line"] = line_no
            meta_by_segment[segment_name] = item

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate segment_video entries found in action meta: {duplicate_list}")

    return meta_by_segment


def segment_local_to_original(start: int, local_frame: int) -> int:
    """Convert a segment-local 0-based frame index to an original-video frame index."""

    if start <= 0:
        return start + local_frame

    return start - 1 + local_frame


def convert_boundaries(meta: dict[str, dict], boundaries: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    results: dict[str, dict] = {}
    missing_segments: list[str] = []

    for segment_name, boundary in boundaries.items():
        segment_meta = meta.get(segment_name)
        if segment_meta is None:
            missing_segments.append(segment_name)
            continue

        start = int(segment_meta["start"])
        end = int(segment_meta["end"])

        last_unoccluded = boundary.get("last_unoccluded_frame")
        last_occluded = boundary.get("last_occluded_frame")

        converted = {
            "original_video": segment_meta["original_video"],
            "segment_video": segment_meta["segment_video"],
            "segment_index": segment_meta.get("segment_index"),
            "total_segments": segment_meta.get("total_segments"),
            "start": start,
            "end": end,
            "last_unoccluded_frame": last_unoccluded,
            "last_occluded_frame": last_occluded,
            "last_unoccluded_frame_original": None,
            "last_occluded_frame_original": None,
            "frame_offset_rule": "start - 1 for positive starts; start for zero-or-negative starts",
        }

        if last_unoccluded is not None:
            converted["last_unoccluded_frame_original"] = segment_local_to_original(start, int(last_unoccluded))

        if last_occluded is not None:
            converted["last_occluded_frame_original"] = segment_local_to_original(start, int(last_occluded))

        results[segment_name] = converted

    return results, missing_segments


def write_csv(output_path: Path, rows: dict[str, dict]) -> None:
    fieldnames = [
        "segment_video",
        "original_video",
        "segment_index",
        "total_segments",
        "start",
        "end",
        "last_unoccluded_frame",
        "last_unoccluded_frame_original",
        "last_occluded_frame",
        "last_occluded_frame_original",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for segment_name in sorted(rows):
            row = rows[segment_name]
            writer.writerow({
                "segment_video": row["segment_video"],
                "original_video": row["original_video"],
                "segment_index": row["segment_index"],
                "total_segments": row["total_segments"],
                "start": row["start"],
                "end": row["end"],
                "last_unoccluded_frame": row["last_unoccluded_frame"],
                "last_unoccluded_frame_original": row["last_unoccluded_frame_original"],
                "last_occluded_frame": row["last_occluded_frame"],
                "last_occluded_frame_original": row["last_occluded_frame_original"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-meta", type=Path, default=DEFAULT_ACTION_META, help="Path to action_videos.jsonl")
    parser.add_argument("--occlusion-json", type=Path, default=DEFAULT_OCCLUSION_JSON, help="Path to occlusion_boundaries.json")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Path to the converted JSON output")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Path to the converted CSV output")
    parser.add_argument("--strict", action="store_true", help="Fail if a segment appears in only one input file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    meta = load_action_meta(args.action_meta)
    with args.occlusion_json.open("r", encoding="utf-8") as handle:
        boundaries = json.load(handle)

    converted, missing_segments = convert_boundaries(meta, boundaries)

    missing_in_boundaries = sorted(set(meta) - set(boundaries))
    if args.strict and (missing_segments or missing_in_boundaries):
        problems = []
        if missing_segments:
            problems.append(f"missing meta for {len(missing_segments)} segments")
        if missing_in_boundaries:
            problems.append(f"missing boundaries for {len(missing_in_boundaries)} segments")
        raise ValueError("; ".join(problems))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, indent=2, ensure_ascii=False)

    write_csv(args.output_csv, converted)

    print(f"Converted {len(converted)} segment videos.")
    print(f"Missing meta entries: {len(missing_segments)}")
    print(f"Missing boundary entries: {len(missing_in_boundaries)}")
    print(f"JSON written to: {args.output_json}")
    print(f"CSV written to: {args.output_csv}")


if __name__ == "__main__":
    main()