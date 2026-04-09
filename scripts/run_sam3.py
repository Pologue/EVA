import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

import sam3
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization

# ===== 路径 =====
META_FILE = "action_videos.jsonl"
MASK_DIR = "./masks"
DEBUG_DIR = "./debug_masks"

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# ===== SAM3 初始化 =====
ckpt = "/opt/data/private/.cache/modelscope/hub/models/facebook/sam3/sam3.pt"

gpus_to_use = range(torch.cuda.device_count())

predictor = build_sam3_video_predictor(
    ckpt,
    gpus_to_use=gpus_to_use,
)


# ===== 工具函数 =====

def clean_label(label):
    return label.lower().replace("_", " ").strip()


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}

    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def merge_frame_masks(frame_out):
    """将单帧输出统一转成一个二值 mask（多实例取并集）。"""
    if frame_out is None:
        return None

    # 官方 prepare_masks_for_visualization 之后的格式: {obj_id: mask}
    if isinstance(frame_out, dict):
        if not frame_out:
            return None

        # 原始输出格式: {'out_binary_masks': ..., ...}
        if "out_binary_masks" in frame_out:
            masks = to_numpy(frame_out["out_binary_masks"])
            if masks.size == 0:
                return None
            if masks.ndim == 2:
                return (masks > 0).astype(np.uint8)
            if masks.ndim == 3:
                return (np.any(masks > 0, axis=0)).astype(np.uint8)
            raise ValueError(f"Unexpected out_binary_masks shape: {masks.shape}")

        # prepare 后: 每个 value 是单实例 mask，空 dict 表示无目标
        merged = None
        for mask in frame_out.values():
            arr = to_numpy(mask)
            if arr.size == 0:
                continue
            if arr.ndim > 2:
                arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Unexpected mask shape: {arr.shape}")
            cur = arr > 0
            merged = cur if merged is None else (merged | cur)

        if merged is None:
            return None
        return merged.astype(np.uint8)

    # 兼容 list/ndarray/tensor
    arr = to_numpy(frame_out)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        return (arr > 0).astype(np.uint8)
    if arr.ndim == 3:
        return (np.any(arr > 0, axis=0)).astype(np.uint8)
    raise ValueError(f"Unknown output format: type={type(frame_out)}, shape={arr.shape}")


def save_mask_video(mask_dict, video_path, out_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)

    frame_idx = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break

        if frame_idx in mask_dict:
            mask = mask_dict[frame_idx]
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        out.write(mask)
        frame_idx += 1

    cap.release()
    out.release()


def save_overlay_video(mask_dict, video_path, out_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in mask_dict:
            mask = mask_dict[frame_idx]
            mask = (mask > 0.5).astype(np.uint8)

            color_mask = np.zeros_like(frame)
            color_mask[:, :, 2] = mask * 255  # 红色 overlay

            frame = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


# ===== 主流程 =====

def main():
    with open(META_FILE, "r") as f:
        metas = [json.loads(line) for line in f]

    for meta in tqdm(metas):
        video_path = meta["segment_video"]
        label = clean_label(meta["label"])

        video_name = os.path.basename(video_path)
        mask_path = os.path.join(MASK_DIR, video_name)
        debug_path = os.path.join(DEBUG_DIR, video_name)

        try:
            # ===== 1️⃣ 创建 session =====
            response = predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=video_path,
                )
            )
            session_id = response["session_id"]

            # ===== 2️⃣ 添加文本 prompt =====
            predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=label,
                )
            )

            # ===== 3️⃣ 视频传播 =====
            outputs = propagate_in_video(predictor, session_id)

            # ===== 4️⃣ 转成 mask =====
            outputs = prepare_masks_for_visualization(outputs)

            # outputs（官方 notebook 格式）: {frame_idx: {obj_id: binary_mask}}
            mask_dict = {}

            for k, v in outputs.items():
                try:
                    mask = merge_frame_masks(v)
                    if mask is None:
                        continue
                    mask_dict[k] = mask
                except Exception as e:
                    print(f"⚠️ frame {k} parse failed: {e}")

            # finally, close the inference session to free its GPU resources
            # (you may start a new session on another video)
            _ = predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )

            # ===== 5️⃣ 保存 =====
            save_mask_video(mask_dict, video_path, mask_path)
            save_overlay_video(mask_dict, video_path, debug_path)

        except Exception as e:
            print(f"❌ Failed on {video_path}: {e}")


if __name__ == "__main__":
    main()