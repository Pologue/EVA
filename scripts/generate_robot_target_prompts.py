#!/usr/bin/env python3

"""Generate actual RoboCasa target-location prompts for occluded and unoccluded scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/opt/data/private/robocasa/datasets/v1.0/pretrain/composite")
DEFAULT_TEMPLATE_ROOT = Path("/opt/data/private/dataset/prompt_templates")
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/dataset/generated_prompts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing composite task datasets",
    )
    parser.add_argument(
        "--template-root",
        type=Path,
        default=DEFAULT_TEMPLATE_ROOT,
        help="Directory containing occluded.txt and unoccluded.txt templates",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to write generated prompts and metadata",
    )
    return parser.parse_args()


def load_template(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def normalize_task_description(tasks: object) -> str:
    if isinstance(tasks, list):
        items = [str(item).strip() for item in tasks if str(item).strip()]
        return " ".join(items)
    if tasks is None:
        return ""
    return str(tasks).strip()


def list_episode_files(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("*/[0-9]*/lerobot/meta/episodes.jsonl"))


def infer_task_and_date(episode_path: Path, dataset_root: Path) -> tuple[str, str]:
    relative = episode_path.relative_to(dataset_root)
    if len(relative.parts) < 5:
        raise ValueError(f"Unexpected episode path layout: {episode_path}")
    task_name = relative.parts[0]
    date_dir = relative.parts[1]
    return task_name, date_dir


def build_prompt(template: str, task_description: str) -> str:
    return template.format(task_description=task_description)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    occluded_template = load_template(args.template_root / "occluded.txt")
    unoccluded_template = load_template(args.template_root / "unoccluded.txt")

    prompt_specs = {
        "occluded": occluded_template,
        "unoccluded": unoccluded_template,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    records_by_type: dict[str, list[dict]] = {key: [] for key in prompt_specs}

    episode_files = list_episode_files(args.dataset_root)
    if not episode_files:
        raise FileNotFoundError(f"No episodes.jsonl files found under {args.dataset_root}")

    for episode_path in episode_files:
        task_name, date_dir = infer_task_and_date(episode_path, args.dataset_root)

        with episode_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                episode = json.loads(line)
                task_description = normalize_task_description(episode.get("tasks"))
                episode_index = episode.get("episode_index")
                episode_length = episode.get("length")

                for prompt_type, template in prompt_specs.items():
                    prompt_text = build_prompt(template, task_description)

                    record = {
                        "prompt_type": prompt_type,
                        "task_name": task_name,
                        "date_dir": date_dir,
                        "episode_index": episode_index,
                        "episode_length": episode_length,
                        "source_episode_path": str(episode_path),
                        "task_description": task_description,
                        "prompt": prompt_text,
                    }

                    output_dir = args.output_root / prompt_type / task_name / date_dir
                    output_dir.mkdir(parents=True, exist_ok=True)

                    txt_path = output_dir / f"episode_{episode_index:06d}.txt"
                    txt_path.write_text(prompt_text + "\n", encoding="utf-8")
                    record["prompt_path"] = str(txt_path)

                    records_by_type[prompt_type].append(record)
                    all_records.append(record)

    combined_path = args.output_root / "all_prompts.jsonl"
    write_jsonl(combined_path, all_records)

    for prompt_type, rows in records_by_type.items():
        write_jsonl(args.output_root / prompt_type / "prompts.jsonl", rows)

    print(f"Generated {len(all_records)} prompts from {len(episode_files)} episode files.")
    print(f"Combined metadata: {combined_path}")
    for prompt_type, rows in records_by_type.items():
        print(f"{prompt_type}: {len(rows)} prompts")


if __name__ == "__main__":
    main()
