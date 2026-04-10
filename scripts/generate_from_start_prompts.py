#!/usr/bin/env python3

"""Generate from_start prompts for unoccluded prefix videos.

The script joins:
1) prefix metadata from dataset/prefix_videos_from_start/prefix_videos_from_start_meta.json
2) RoboCasa episodes metadata under robocasa/datasets/v1.0/pretrain/composite/<task>/<date>/lerobot/meta/episodes.jsonl

It then writes one concrete prompt .txt file per unoccluded prefix video.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/opt/data/private/robocasa/datasets/v1.0/pretrain/composite")
DEFAULT_PREFIX_META = Path("/opt/data/private/dataset/prefix_videos_from_start/prefix_videos_from_start_meta.json")
DEFAULT_TEMPLATE_PATH = Path("/opt/data/private/dataset/prompt_templates/from_start_unoccluded.txt")
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/dataset/generated_prompts/from_start")

EPISODE_PATTERN = re.compile(r"__ep(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="RoboCasa composite dataset root",
    )
    parser.add_argument(
        "--prefix-meta",
        type=Path,
        default=DEFAULT_PREFIX_META,
        help="Path to prefix_videos_from_start_meta.json",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help="Prompt template path",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for generated prompts",
    )
    parser.add_argument(
        "--frame-kind",
        default="unoccluded",
        help="Only process rows whose frame_kind matches this value",
    )
    parser.add_argument(
        "--status",
        default="ok",
        help="Only process rows whose status matches this value",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prompt files",
    )
    return parser.parse_args()


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_template(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_task_name(segment_name: str) -> str:
    return segment_name.split("__", 1)[0]


def parse_episode_index(segment_name: str) -> int:
    match = EPISODE_PATTERN.search(segment_name)
    if not match:
        raise ValueError(f"Cannot parse episode index from segment name: {segment_name}")
    return int(match.group(1))


def normalize_task_description(episode: dict) -> str:
    if "task_description" in episode and str(episode["task_description"]).strip():
        return str(episode["task_description"]).strip()

    tasks = episode.get("tasks")
    if isinstance(tasks, list):
        items = [str(item).strip() for item in tasks if str(item).strip()]
        return " ".join(items)
    if tasks is None:
        return ""
    return str(tasks).strip()


def find_single_episode_path(task_dir: Path) -> tuple[Path, str]:
    candidates = sorted(task_dir.glob("*/lerobot/meta/episodes.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No episodes.jsonl found under {task_dir}")
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one date directory under {task_dir}, got {len(candidates)}"
        )

    episode_path = candidates[0]
    date_dir = episode_path.relative_to(task_dir).parts[0]
    return episode_path, date_dir


def load_episodes_index(episode_path: Path) -> dict[int, dict]:
    by_index: dict[int, dict] = {}
    with episode_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            episode = json.loads(line)
            episode_idx = episode.get("episode_index")
            if episode_idx is None:
                continue
            by_index[int(episode_idx)] = episode
    return by_index


def build_prompt(template: str, task_description: str) -> str:
    return template.replace("{{task_description}}", task_description)


def main() -> None:
    args = parse_args()

    template = load_template(args.template_path)
    rows = load_json(args.prefix_meta)
    if not isinstance(rows, list):
        raise TypeError(f"Expected list in {args.prefix_meta}, got {type(rows)}")

    selected_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("frame_kind") == args.frame_kind
        and row.get("status") == args.status
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)

    task_cache: dict[str, tuple[Path, str, dict[int, dict]]] = {}
    records: list[dict] = []

    for row in selected_rows:
        segment_name = str(row.get("segment_name", "")).strip()
        if not segment_name:
            continue

        task_name = parse_task_name(segment_name)
        episode_index = parse_episode_index(segment_name)

        if task_name not in task_cache:
            task_dir = args.dataset_root / task_name
            episode_path, date_dir = find_single_episode_path(task_dir)
            episodes_by_index = load_episodes_index(episode_path)
            task_cache[task_name] = (episode_path, date_dir, episodes_by_index)

        episode_path, date_dir, episodes_by_index = task_cache[task_name]
        episode = episodes_by_index.get(episode_index)

        if episode is None:
            records.append(
                {
                    "status": "missing_episode",
                    "segment_name": segment_name,
                    "task_name": task_name,
                    "episode_index": episode_index,
                    "source_episode_path": str(episode_path),
                }
            )
            continue

        task_description = normalize_task_description(episode)
        prompt_text = build_prompt(template, task_description)

        clip_path = Path(str(row.get("output_video", "")))
        clip_stem = clip_path.stem if clip_path.stem else Path(segment_name).stem

        out_dir = args.output_root / args.frame_kind / task_name / date_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{clip_stem}__from_start_prompt.txt"

        if out_path.exists() and not args.overwrite:
            status = "skipped_existing"
        else:
            out_path.write_text(prompt_text + "\n", encoding="utf-8")
            status = "ok"

        records.append(
            {
                "status": status,
                "frame_kind": args.frame_kind,
                "segment_name": segment_name,
                "segment_video": row.get("segment_video"),
                "prefix_video": row.get("output_video"),
                "task_name": task_name,
                "date_dir": date_dir,
                "episode_index": episode_index,
                "episode_length": episode.get("length"),
                "source_episode_path": str(episode_path),
                "task_description": task_description,
                "prompt_path": str(out_path),
            }
        )

    output_jsonl = args.output_root / args.frame_kind / "prompts_from_start.jsonl"
    output_json = args.output_root / args.frame_kind / "prompts_from_start.json"
    write_jsonl(output_jsonl, records)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    total = len(records)
    ok = sum(1 for item in records if item.get("status") == "ok")
    skipped = sum(1 for item in records if str(item.get("status", "")).startswith("skipped"))
    missing = sum(1 for item in records if item.get("status") == "missing_episode")

    print(f"Selected rows: {len(selected_rows)}")
    print(f"Generated records: {total}")
    print(f"OK: {ok} | Skipped: {skipped} | Missing episode: {missing}")
    print(f"JSONL: {output_jsonl}")
    print(f"JSON: {output_json}")


if __name__ == "__main__":
    main()