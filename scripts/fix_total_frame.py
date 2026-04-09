import json
import subprocess
from pathlib import Path
import sys

ANNOTATION_PATH = "annotations.json"
OUTPUT_PATH = "annotations_fixed.json"


def get_ffmpeg_frames(video_path):
    """用 ffprobe 获取真实帧数"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.stdout.strip())


def find_start_index(data, start_video_name):
    for i, item in enumerate(data):
        if start_video_name in item["file_upload"]:
            return i
    raise ValueError(f"找不到起始视频: {start_video_name}")


def main(start_video_name, txt_path):
    with open(ANNOTATION_PATH, "r") as f:
        data = json.load(f)

    with open(txt_path, "r") as f:
        ls_frames_list = [int(line.strip()) for line in f if line.strip()]

    start_idx = find_start_index(data, start_video_name)

    print(f"Start index: {start_idx}")

    for offset, ls_total_frames in enumerate(ls_frames_list):
        idx = start_idx + offset

        if idx >= len(data):
            print("⚠️ 超出 annotations 范围，停止")
            break

        item = data[idx]
        video_path = item["data"]["video"]

        # 获取真实帧数
        real_frames = get_ffmpeg_frames(video_path)

        ratio = real_frames / ls_total_frames

        print(f"\nProcessing {video_path}")
        print(f"LS frames: {ls_total_frames}, Real frames: {real_frames}, ratio: {ratio:.4f}")

        annotations = item.get("annotations", [])
        if not annotations:
            continue

        results = annotations[0]["result"]

        for res in results:
            r = res["value"]["ranges"][0]

            old_start = r["start"]
            old_end = r["end"]

            # 缩放
            new_start = int(round(old_start * ratio))
            new_end = int(round(old_end * ratio))

            # 防越界
            new_start = max(0, min(new_start, real_frames - 1))
            new_end = max(new_start + 1, min(new_end, real_frames))

            r["start"] = new_start
            r["end"] = new_end

    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ 已输出到 {OUTPUT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python fix_frame_mismatch.py <start_video_name> <txt_path>")
        sys.exit(1)

    start_video_name = sys.argv[1]
    txt_path = sys.argv[2]

    main(start_video_name, txt_path)