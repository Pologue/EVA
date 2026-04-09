import json
import os
import cv2
import numpy as np
from tqdm import tqdm

import sam3
import torch

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# use all available GPUs on the machine
gpus_to_use = range(torch.cuda.device_count())
# # use only a single GPU
# gpus_to_use = [torch.cuda.current_device()]


# ==== 需要你改路径 ====
META_FILE = "action_videos.jsonl"
OUTPUT_DIR = "./masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====== 伪接口（你需要替换成真实模型调用） ======
def detect_bbox_with_text(image, text):
    """
    用 GroundingDINO
    return: [x1, y1, x2, y2]
    """
    h, w, _ = image.shape
    return [0, 0, w, h]  # TODO: 替换


def sam_segment(image, bbox):
    """
    用 SAM
    return mask (H, W)
    """
    h, w, _ = image.shape
    return np.ones((h, w), dtype=np.uint8) * 255  # TODO


# ==============================================

def process_video(meta):
    video_path = meta["segment_video"]
    label = meta["label"]

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(
        OUTPUT_DIR,
        os.path.basename(video_path).replace(".mp4", "_mask.mp4")
    )

    ret, frame = cap.read()
    if not ret:
        return

    h, w = frame.shape[:2]
    out = cv2.VideoWriter(out_path, fourcc, 20, (w, h), isColor=False)

    # 第1帧：检测 + 分割
    bbox = detect_bbox_with_text(frame, label)

    while True:
        if frame is None:
            break

        mask = sam_segment(frame, bbox)

        out.write(mask)

        ret, frame = cap.read()
        if not ret:
            break

    cap.release()
    out.release()


def main():
    with open(META_FILE, "r") as f:
        metas = [json.loads(line) for line in f]

    for meta in tqdm(metas):
        process_video(meta)


if __name__ == "__main__":
    main()