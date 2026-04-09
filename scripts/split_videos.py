import json
import os
import re
import subprocess
from pathlib import Path
from tqdm import tqdm

INPUT_JSON = "/home/pologue/Desktop/academic/Senior2/dataset/annotations.json"
OUTPUT_VIDEO_DIR = "./action_videos"
OUTPUT_META = "/home/pologue/Desktop/academic/Senior2/dataset/action_videos.jsonl"

FPS = 20.0

os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)


def safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)


def run_ffmpeg(input_path, start, end, output_path):
    start_sec = start / FPS
    end_sec = end / FPS

    duration = end - start
    duration_sec = end_sec - start_sec

    # cmd = [
    #     "ffmpeg",
    #     "-y",
    #     "-i", input_path,
    #     "-ss", str(start_sec),
    #     "-t", str(duration_sec),
    #     "-c:v", "libx264",
    #     "-frames:v", f"{duration}",
    #     "-preset", "fast",
    #     "-crf", "23",
    #     "-an",
    #     output_path
    # ]

    # ffmpeg -i input.mp4 -vf "select='between(n,100,200)',setpts=N/FRAME_RATE/TB" -c:v libx264 output.mp4
    # cmd = [
    #     "ffmpeg",
    #     "-i", input_path,
    #     "-vf", f"select='between(n,{start},{end})',setpts=N/FRAME_RATE/TB",
    #     "-c:v", "libx264",
    #     "-frames:v", f"{duration}",
    #     "-crf", "18",
    #     output_path
    # ]

    # ffmpeg -i input.mp4 -vf "trim=start_frame=100:end_frame=200,setpts=PTS-STARTPTS" -af "atrim=start_sample=44100:end_sample=88200,asetpts=PTS-STARTPTS" output.mp4
    cmd = [
        "ffmpeg",
        "-i", input_path,
        # "-vf", f"trim=start_frame={start-1}:end_frame={end+1},setpts=PTS-STARTPTS,drawtext=text='%{{n}}':x=10:y=10:fontsize=30:fontcolor=red",
        "-vf", f"trim=start_frame={start-1}:end_frame={end+1},setpts=PTS-STARTPTS",
        "-c:v", "libx264",
        # "-crf", "18",
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    # subprocess.run(cmd, check=True)


def process():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    meta_out = []

    for item in tqdm(data):
        video_path = item["data"]["video"]
        annotations = item.get("annotations", [])
        file_name = item["file_upload"]

        if not annotations:
            continue

        results = annotations[0]["result"]
        total_segments = len(results)

        for idx, res in enumerate(results):
            ranges = res["value"]["ranges"][0]
            label = res["value"]["timelinelabels"][0]

            start = ranges["start"]
            end = ranges["end"]

            safe_label = safe_filename(label)

            segment_name = f"{Path(file_name).stem}_seg{idx}_{safe_label}.mp4"
            segment_path = os.path.join(OUTPUT_VIDEO_DIR, segment_name)

            # 切视频
            run_ffmpeg(video_path, start, end, segment_path)

            # 写 meta
            meta = {
                "original_video": video_path,
                "segment_video": segment_path,
                "label": label,
                "start": start,
                "end": end,
                "segment_index": idx,
                "total_segments": total_segments
            }

            meta_out.append(meta)

    # 保存 JSONL
    with open(OUTPUT_META, "w") as f:
        for m in meta_out:
            f.write(json.dumps(m) + "\n")


if __name__ == "__main__":
    process()