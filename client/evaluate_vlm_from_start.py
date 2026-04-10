#!/usr/bin/env python3

"""Evaluate VLMs on from_start unoccluded prefix videos.

Features:
1. Uses all videos under dataset/prefix_videos_from_start/unoccluded by default.
2. Loads the corresponding prompt text generated under generated_prompt(s)/from_start/unoccluded.
3. Samples frames from each video at a configurable FPS and always appends the last frame.
4. Sends a set of images + prompt text to VLMs (Qwen via vLLM, ChatGPT-4o, Gemini).
5. Records per-request statistics: token usage, latency, and optional GPU metrics.
6. Saves machine-readable outputs for downstream analysis.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import datetime as dt
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import OpenAI


DEFAULT_VIDEO_DIR = Path("/opt/data/private/dataset/prefix_videos_from_start/unoccluded")
DEFAULT_PROMPT_META_CANDIDATES = [
    Path("/opt/data/private/dataset/generated_prompt/from_start/unoccluded/prompts_from_start.jsonl"),
    Path("/opt/data/private/dataset/generated_prompts/from_start/unoccluded/prompts_from_start.jsonl"),
]
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/dataset/eval_results/from_start")


@dataclass
class Sample:
    sample_id: str
    video_path: Path
    prompt_path: Path
    prompt_text: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument(
        "--prompt-meta",
        type=Path,
        default=None,
        help="Path to prompts_from_start.jsonl. If omitted, auto-detects generated_prompt(s).",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen", "chatgpt", "gemini"],
        choices=["qwen", "chatgpt", "gemini"],
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Sample frames at this FPS, then always append the last frame",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument(
        "--attempt-timeout",
        type=int,
        default=240,
        help="Hard timeout (seconds) for each single request attempt",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--chatgpt-base-url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="Optional base URL for OpenAI-compatible ChatGPT provider",
    )
    parser.add_argument(
        "--gemini-base-url",
        type=str,
        default=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
        help="Gemini API base URL or endpoint for third-party providers",
    )
    parser.add_argument("--gpu-poll-seconds", type=float, default=0.5)
    parser.add_argument(
        "--qwen-gpu-indices",
        type=str,
        default="",
        help="Comma-separated GPU indices to monitor for Qwen, e.g. 0,1. Empty means all visible GPUs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only prepare inputs, do not call APIs")
    return parser.parse_args()


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve_prompt_meta_path(cli_path: Path | None) -> Path:
    if cli_path is not None:
        if not cli_path.exists():
            raise FileNotFoundError(f"Prompt metadata file not found: {cli_path}")
        return cli_path

    for path in DEFAULT_PROMPT_META_CANDIDATES:
        if path.exists():
            return path

    candidate_str = "\n".join(str(p) for p in DEFAULT_PROMPT_META_CANDIDATES)
    raise FileNotFoundError(f"No prompt metadata file found. Checked:\n{candidate_str}")


def parse_gpu_indices(text: str) -> set[int] | None:
    if not text.strip():
        return None
    values = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.add(int(chunk))
    return values or None


def normalize_base_url(url: str) -> str:
    return url.strip().rstrip("/")


def load_prompt_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
    return records


def build_samples(video_dir: Path, prompt_meta_path: Path) -> list[Sample]:
    prompt_records = load_prompt_records(prompt_meta_path)
    by_video_name: dict[str, dict[str, Any]] = {}
    for row in prompt_records:
        prefix_video = str(row.get("prefix_video", "")).strip()
        prompt_path = str(row.get("prompt_path", "")).strip()
        if row.get("status") != "ok" or not prefix_video or not prompt_path:
            continue
        by_video_name[Path(prefix_video).name] = row

    samples: list[Sample] = []
    for video_path in sorted(video_dir.glob("*.mp4")):
        row = by_video_name.get(video_path.name)
        if row is None:
            continue

        prompt_path = Path(str(row["prompt_path"]))
        if not prompt_path.exists():
            continue

        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        samples.append(
            Sample(
                sample_id=video_path.stem,
                video_path=video_path,
                prompt_path=prompt_path,
                prompt_text=prompt_text,
                metadata=row,
            )
        )

    return samples


def extract_frames(video_path: Path, sample_fps: float) -> tuple[list[np.ndarray], list[int], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 0:
        video_fps = 1.0

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    total = len(frames)
    if total == 0:
        raise RuntimeError(f"Empty video: {video_path}")

    sample_fps = max(float(sample_fps), 0.0)
    if sample_fps <= 0:
        sample_fps = 1.0

    if total == 1:
        indices = [0]
    else:
        frame_step = max(video_fps / sample_fps, 1e-6)
        sampled = np.arange(0.0, float(total - 1), frame_step)
        indices = [int(round(value)) for value in sampled.tolist()]
        indices.append(total - 1)

    deduped: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)

    return [frames[i] for i in deduped], deduped, total


def encode_image_to_data_url(frame_bgr: np.ndarray, jpeg_quality: int) -> str:
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 1, 100))],
    )
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def encode_image_to_base64(frame_bgr: np.ndarray, jpeg_quality: int) -> str:
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 1, 100))],
    )
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def to_plain_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def extract_text_from_openai_response(resp: Any) -> str:
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def extract_usage_from_openai_response(resp: Any) -> dict[str, Any]:
    usage = to_plain_dict(getattr(resp, "usage", None))
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def query_nvidia_smi() -> list[dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    rows: list[dict[str, int]] = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "utilization_gpu": int(parts[1]),
                "memory_used_mb": int(parts[2]),
                "memory_total_mb": int(parts[3]),
            }
        )
    return rows


class GPUMonitor:
    def __init__(self, poll_seconds: float, gpu_indices: set[int] | None) -> None:
        self.poll_seconds = max(0.1, poll_seconds)
        self.gpu_indices = gpu_indices
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = True

    def _filter_rows(self, rows: list[dict[str, int]]) -> list[dict[str, int]]:
        if self.gpu_indices is None:
            return rows
        return [row for row in rows if row["index"] in self.gpu_indices]

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                rows = self._filter_rows(query_nvidia_smi())
                self.samples.append({"time": time.time(), "gpus": rows})
            except Exception:
                self.available = False
                break
            self._stop_event.wait(self.poll_seconds)

    def start(self) -> None:
        try:
            _ = self._filter_rows(query_nvidia_smi())
        except Exception:
            self.available = False
            return

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        if not self.available or not self.samples:
            return {"available": False}

        per_gpu: dict[int, dict[str, int]] = {}
        for sample in self.samples:
            for row in sample.get("gpus", []):
                idx = int(row["index"])
                state = per_gpu.setdefault(
                    idx,
                    {
                        "max_utilization_gpu": 0,
                        "max_memory_used_mb": 0,
                        "memory_total_mb": int(row.get("memory_total_mb", 0)),
                    },
                )
                state["max_utilization_gpu"] = max(state["max_utilization_gpu"], int(row["utilization_gpu"]))
                state["max_memory_used_mb"] = max(state["max_memory_used_mb"], int(row["memory_used_mb"]))

        overall_max_util = 0
        overall_max_mem = 0
        for state in per_gpu.values():
            overall_max_util = max(overall_max_util, state["max_utilization_gpu"])
            overall_max_mem = max(overall_max_mem, state["max_memory_used_mb"])

        return {
            "available": True,
            "sample_count": len(self.samples),
            "overall_max_utilization_gpu": overall_max_util,
            "overall_max_memory_used_mb": overall_max_mem,
            "per_gpu": per_gpu,
        }


def call_qwen(
    sample: Sample,
    image_data_urls: list[str],
    args: argparse.Namespace,
    timeout: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    base_url = os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.getenv("QWEN_API_KEY", "EMPTY")
    model_name = os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-VL-32B-Instruct")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    content: list[dict[str, Any]] = [
        {"type": "text", "text": sample.prompt_text}
    ]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_data_urls)

    messages = [{"role": "user", "content": content}]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )

    usage = extract_usage_from_openai_response(response)
    meta = {
        "provider": "qwen_vllm",
        "model": model_name,
        "base_url": base_url,
    }
    return extract_text_from_openai_response(response), usage, meta


def call_chatgpt(
    sample: Sample,
    image_data_urls: list[str],
    args: argparse.Namespace,
    timeout: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    base_url = normalize_base_url(args.chatgpt_base_url)
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    else:
        client = OpenAI(api_key=api_key, timeout=timeout)

    content: list[dict[str, Any]] = [
        {"type": "text", "text": sample.prompt_text}
    ]
    content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_data_urls)

    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )

    usage = extract_usage_from_openai_response(response)
    meta = {
        "provider": "openai",
        "model": model_name,
        "base_url": base_url or None,
    }
    return extract_text_from_openai_response(response), usage, meta


def build_gemini_request(
    model_name: str,
    base_url: str,
    api_key: str,
    body: bytes,
) -> urllib.request.Request:
    base = normalize_base_url(base_url)
    if "{model}" in base:
        url = base.format(model=model_name)
    elif base.endswith(":generateContent"):
        url = base
    else:
        url = f"{base}/models/{model_name}:generateContent"

    headers = {"Content-Type": "application/json"}

    if "googleapis.com" in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}key={api_key}"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )


def call_gemini(
    sample: Sample,
    image_base64_list: list[str],
    args: argparse.Namespace,
    timeout: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    parts = [{"text": sample.prompt_text}]
    parts.extend(
        {
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": b64,
            }
        }
        for b64 in image_base64_list
    )

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": args.temperature,
            "maxOutputTokens": args.max_output_tokens,
        },
    }
    body = json.dumps(payload).encode("utf-8")

    req = build_gemini_request(
        model_name=model_name,
        base_url=args.gemini_base_url,
        api_key=api_key,
        body=body,
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTPError {exc.code}: {err}") from exc

    data = json.loads(raw)
    text = ""
    candidates = data.get("candidates") or []
    if candidates:
        parts_out = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        text = "".join(str(item.get("text", "")) for item in parts_out)

    usage_meta = data.get("usageMetadata") or {}
    usage = {
        "prompt_tokens": usage_meta.get("promptTokenCount"),
        "completion_tokens": usage_meta.get("candidatesTokenCount"),
        "total_tokens": usage_meta.get("totalTokenCount"),
    }
    meta = {
        "provider": "gemini",
        "model": model_name,
        "base_url": normalize_base_url(args.gemini_base_url),
        "endpoint": "generateContent",
    }
    return text, usage, meta


def safe_json_loads(text: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except Exception:
        return False, None


def run_with_hard_timeout(func: Any, timeout_seconds: int) -> Any:
    """Run a blocking callable with a hard timeout guard."""
    timeout_seconds = max(1, int(timeout_seconds))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"request timed out after {timeout_seconds}s") from exc


def run_single_request(
    model_key: str,
    sample: Sample,
    image_data_urls: list[str],
    image_base64_list: list[str],
    frame_indices: list[int],
    total_frames: int,
    args: argparse.Namespace,
    qwen_gpu_indices: set[int] | None,
) -> dict[str, Any]:
    started_iso = utc_now_iso()
    started_ts = time.time()

    gpu_stats: dict[str, Any] = {"available": False}
    monitor: GPUMonitor | None = None
    if model_key == "qwen":
        monitor = GPUMonitor(args.gpu_poll_seconds, qwen_gpu_indices)
        monitor.start()

    error_message = ""
    response_text = ""
    usage: dict[str, Any] = {}
    model_meta: dict[str, Any] = {}
    status = "ok"
    timed_out = False

    for attempt in range(args.max_retries + 1):
        try:
            if args.dry_run:
                response_text = '{"dry_run": true}'
                usage = {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                }
                model_meta = {"provider": model_key, "model": "dry-run"}
            else:
                if model_key == "qwen":
                    invoke = lambda: call_qwen(sample, image_data_urls, args, args.request_timeout)
                elif model_key == "chatgpt":
                    invoke = lambda: call_chatgpt(sample, image_data_urls, args, args.request_timeout)
                elif model_key == "gemini":
                    invoke = lambda: call_gemini(sample, image_base64_list, args, args.request_timeout)
                else:
                    raise RuntimeError(f"Unsupported model key: {model_key}")

                response_text, usage, model_meta = run_with_hard_timeout(invoke, args.attempt_timeout)

            model_meta["request_timeout_seconds"] = int(args.request_timeout)
            model_meta["attempt_timeout_seconds"] = int(args.attempt_timeout)
            break
        except Exception as exc:
            error_message = str(exc)
            timed_out = isinstance(exc, TimeoutError) or "timed out" in error_message.lower()
            if attempt >= args.max_retries:
                status = "failed"
            else:
                time.sleep(min(2.0, 0.5 * (attempt + 1)))

    if monitor is not None:
        gpu_stats = monitor.stop()

    ended_ts = time.time()
    ended_iso = utc_now_iso()
    elapsed = ended_ts - started_ts

    is_json, parsed_json = safe_json_loads(response_text)

    return {
        "status": status,
        "timed_out": timed_out,
        "error": error_message,
        "model_key": model_key,
        "provider": model_meta.get("provider"),
        "model": model_meta.get("model"),
        "model_meta": model_meta,
        "sample_id": sample.sample_id,
        "video_path": str(sample.video_path),
        "prompt_path": str(sample.prompt_path),
        "frame_indices": frame_indices,
        "num_input_images": len(frame_indices),
        "total_video_frames": total_frames,
        "request_started_at": started_iso,
        "request_ended_at": ended_iso,
        "latency_seconds": elapsed,
        "usage": usage,
        "context_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "gpu_stats": gpu_stats,
        "response_text": response_text,
        "response_is_json": is_json,
        "response_json": parsed_json if is_json else None,
        "prompt_metadata": sample.metadata,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "status",
        "model_key",
        "provider",
        "model",
        "sample_id",
        "video_path",
        "prompt_path",
        "num_input_images",
        "total_video_frames",
        "latency_seconds",
        "context_tokens",
        "completion_tokens",
        "total_tokens",
        "response_is_json",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("model_key", "unknown")
        agg = by_model.setdefault(
            key,
            {
                "requests": 0,
                "ok": 0,
                "failed": 0,
                "json_ok": 0,
                "sum_latency": 0.0,
                "sum_context_tokens": 0,
                "sum_total_tokens": 0,
                "context_token_records": 0,
                "total_token_records": 0,
            },
        )
        agg["requests"] += 1
        if row.get("status") == "ok":
            agg["ok"] += 1
        else:
            agg["failed"] += 1
        if row.get("response_is_json"):
            agg["json_ok"] += 1

        latency = row.get("latency_seconds")
        if isinstance(latency, (int, float)):
            agg["sum_latency"] += float(latency)

        context_tokens = row.get("context_tokens")
        if isinstance(context_tokens, (int, float)):
            agg["sum_context_tokens"] += int(context_tokens)
            agg["context_token_records"] += 1

        total_tokens = row.get("total_tokens")
        if isinstance(total_tokens, (int, float)):
            agg["sum_total_tokens"] += int(total_tokens)
            agg["total_token_records"] += 1

    summary = {"overall": {"records": len(rows)}, "by_model": {}}
    for key, agg in by_model.items():
        requests = max(1, agg["requests"])
        ctx_recs = max(1, agg["context_token_records"])
        tok_recs = max(1, agg["total_token_records"])
        summary["by_model"][key] = {
            "requests": agg["requests"],
            "ok": agg["ok"],
            "failed": agg["failed"],
            "json_ok": agg["json_ok"],
            "avg_latency_seconds": agg["sum_latency"] / requests,
            "avg_context_tokens": agg["sum_context_tokens"] / ctx_recs,
            "avg_total_tokens": agg["sum_total_tokens"] / tok_recs,
        }
    return summary


def main() -> None:
    args = parse_args()

    prompt_meta_path = resolve_prompt_meta_path(args.prompt_meta)
    samples = build_samples(args.video_dir, prompt_meta_path)

    if not samples:
        raise RuntimeError("No matched (video, prompt) samples found.")

    start = max(0, args.start_index)
    if start >= len(samples):
        raise ValueError(f"start-index {start} >= sample count {len(samples)}")

    selected = samples[start:]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "run_config.json"
    config = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "video_dir": str(args.video_dir),
        "prompt_meta_path": str(prompt_meta_path),
        "models": args.models,
        "selected_samples": len(selected),
        "sample_fps": args.sample_fps,
        "jpeg_quality": args.jpeg_quality,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "chatgpt_base_url": normalize_base_url(args.chatgpt_base_url) or None,
        "gemini_base_url": normalize_base_url(args.gemini_base_url),
        "max_retries": args.max_retries,
        "request_timeout": args.request_timeout,
        "dry_run": args.dry_run,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    qwen_gpu_indices = parse_gpu_indices(args.qwen_gpu_indices)

    all_rows: list[dict[str, Any]] = []

    total_requests = len(selected) * len(args.models)
    request_counter = 0

    for sample in selected:
        frames, frame_indices, total_frames = extract_frames(sample.video_path, args.sample_fps)
        image_data_urls = [encode_image_to_data_url(f, args.jpeg_quality) for f in frames]
        image_base64_list = [encode_image_to_base64(f, args.jpeg_quality) for f in frames]

        for model_key in args.models:
            request_counter += 1
            print(
                f"[{request_counter}/{total_requests}] model={model_key} sample={sample.sample_id} "
                f"images={len(frames)}"
            )
            row = run_single_request(
                model_key=model_key,
                sample=sample,
                image_data_urls=image_data_urls,
                image_base64_list=image_base64_list,
                frame_indices=frame_indices,
                total_frames=total_frames,
                args=args,
                qwen_gpu_indices=qwen_gpu_indices,
            )
            all_rows.append(row)

    detail_jsonl = run_dir / "details.jsonl"
    detail_json = run_dir / "details.json"
    summary_csv = run_dir / "summary.csv"
    metrics_json = run_dir / "metrics.json"

    write_jsonl(detail_jsonl, all_rows)
    detail_json.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(summary_csv, all_rows)

    metrics = aggregate_metrics(all_rows)
    metrics["run_id"] = run_id
    metrics["created_at"] = utc_now_iso()
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Run directory: {run_dir}")
    print(f"Detail JSONL: {detail_jsonl}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Metrics JSON: {metrics_json}")


if __name__ == "__main__":
    main()