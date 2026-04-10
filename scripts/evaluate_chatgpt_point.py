import json
import re
import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# INPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_070057/details.json"
# INPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/details.json"
INPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_140446/details.json"
MAPPING_FILE = "/opt/data/private/dataset/action_videos.jsonl"
MASK_VIDEO_DIR = "/opt/data/private/dataset/masks/" # 存放掩码视频的目录

# OUTPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_070057/details_points_evaluated.json"
# OUTPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/details_points_evaluated.json"
OUTPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_140446/details_points_evaluated.json"
# OUTPUT_METRICS = "/opt/data/private/dataset/eval_results/from_start/20260410_070057/points_metrics.json"
# OUTPUT_METRICS = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/points_metrics.json"
OUTPUT_METRICS = "/opt/data/private/dataset/eval_results/from_start/20260410_140446/points_metrics.json"
# ===========================================

def load_segment_mappings(mapping_path):
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            seg_name = os.path.basename(data['segment_video'])
            mapping[seg_name] = data['start']
    return mapping

def extract_points_from_response(text):
    """从 ChatGPT 的 JSON 回复中提取 points: [[x1, y1], [x2, y2]]"""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("points") # 格式通常是 [[x, y], [x, y]]
    except: return None
    return None

def get_mask_pixel_value(mask_video_path, local_frame_idx, points_1000):
    """
    检查归一化的点是否落在 Mask 像素上
    """
    if not os.path.exists(mask_video_path):
        return None, "Mask file not found"

    cap = cv2.VideoCapture(mask_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, f"Frame {local_frame_idx} not found in mask"

    # 转换为单通道二值掩码
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    h, w = mask.shape

    results = []
    for pt in points_1000:
        # 1. 坐标转换：从 1000 映射到实际分辨率 (e.g., 256)
        # 注意：Prompt 要求是 [x, y]，对应矩阵是 mask[y, x]
        px = int(pt[0] * w / 1000)
        py = int(pt[1] * h / 1000)
        
        # 边界检查
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))

        # 2. 检查像素值
        is_hit = 1 if mask[py, px] > 0 else 0
        results.append(is_hit)
    
    return results, "success"

def main():
    seg_start_map = load_segment_mappings(MAPPING_FILE)
    with open(INPUT_DETAILS, 'r') as f:
        details_list = json.load(f)

    evaluated_items = []
    point_hits = 0
    total_points = 0
    sample_hits = 0
    valid_samples = 0

    print("Evaluating points against SAM3 masks...")

    for item in tqdm(details_list):
        sample_id = item["sample_id"]
        abs_frame = item["frame_indices"][-1]
        
        # 1. 获取对应的掩码视频路径和局部帧索引
        seg_base_name = sample_id.replace("__unoccluded", "").replace("__occluded", "")
        seg_video_name = seg_base_name + ".mp4"
        mask_path = os.path.join(MASK_VIDEO_DIR, seg_video_name)
        
        if seg_video_name not in seg_start_map: continue
        local_idx = abs_frame - seg_start_map[seg_video_name]

        # 2. 提取预测点
        pred_points = extract_points_from_response(item.get("response_text", ""))
        
        if not pred_points or not isinstance(pred_points, list):
            item['point_eval'] = {"status": "no_points_predicted"}
            evaluated_items.append(item)
            continue

        # 3. 像素级核对
        hit_results, status = get_mask_pixel_value(mask_path, local_idx, pred_points)
        
        if hit_results is not None:
            hit_count = sum(hit_results)
            point_hits += hit_count
            total_points += len(pred_points)
            
            # 如果有一个点中了，该样本就算 Hit
            is_sample_hit = 1 if hit_count > 0 else 0
            sample_hits += is_sample_hit
            valid_samples += 1
            
            item['point_eval'] = {
                "status": status,
                "pred_points": pred_points,
                "hit_results": hit_results,
                "hit_count": hit_count,
                "is_hit": bool(is_sample_hit)
            }
        else:
            item['point_eval'] = {"status": status}
            
        evaluated_items.append(item)

    # 汇总结果
    metrics = {
        "point_wise_accuracy": point_hits / total_points if total_points > 0 else 0,
        "sample_hit_rate": sample_hits / valid_samples if valid_samples > 0 else 0,
        "valid_samples": valid_samples,
        "total_points_evaluated": total_points
    }

    with open(OUTPUT_DETAILS, 'w') as f:
        json.dump(evaluated_items, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_METRICS, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*40)
    print(f"点坐标评估完成！")
    print(f"点级准确率 (Point-wise Acc): {metrics['point_wise_accuracy']:.2%}")
    print(f"样本击中率 (Hit Rate @ 1): {metrics['sample_hit_rate']:.2%}")
    print(f"平均每个视频预测点数: {total_points/valid_samples:.1f}" if valid_samples > 0 else "")
    print("="*40)

if __name__ == "__main__":
    main()