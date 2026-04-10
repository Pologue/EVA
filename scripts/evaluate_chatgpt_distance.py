import json
import re
import os
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# INPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_070057/details.json"
INPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/details.json"
MAPPING_FILE = "/opt/data/private/dataset/action_videos.jsonl"
BBOX_GT_DIR = "/opt/data/private/dataset/mask_results/bboxes/"

# OUTPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_070057/details_evaluated.json"
OUTPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/details_evaluated.json"
# OUTPUT_METRICS = "/opt/data/private/dataset/eval_results/from_start/20260410_070057/evaluation_metrics.json"
OUTPUT_METRICS = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/evaluation_metrics.json"
# ===========================================

def calculate_center_distance(boxA, boxB):
    """
    计算两个矩形框中心点的归一化欧氏距离 (Normalized Euclidean Distance)
    box 格式均为 [xmin, ymin, xmax, ymax]
    """
    if not boxA or not boxB: return None

    # 计算 boxA 中心点
    centerA_x = (boxA[0] + boxA[2]) / 2
    centerA_y = (boxA[1] + boxA[3]) / 2

    # 计算 boxB 中心点
    centerB_x = (boxB[0] + boxB[2]) / 2
    centerB_y = (boxB[1] + boxB[3]) / 2

    # 计算欧氏距离
    distance = np.sqrt((centerA_x - centerB_x)**2 + (centerA_y - centerB_y)**2)

    # 归一化到 0-1 之间 (假设最远距离是 1000)
    normalized_distance = distance / 1000
    return normalized_distance

def extract_pred_bbox(text):
    """从 ChatGPT 文本中提取坐标 [xmin, ymin, xmax, ymax]"""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("bbox")
    except: return None
    return None

def load_segment_mappings(mapping_path):
    """加载 action_videos.jsonl，建立文件名到 start 帧的映射"""
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            seg_name = os.path.basename(data['segment_video'])
            mapping[seg_name] = data['start']
    return mapping

def get_gt_bbox_from_json(sample_id, target_frame):
    """从 GT JSON 提取坐标并作归一化"""
    base_name = sample_id.replace("__unoccluded", "").replace("__occluded", "")
    gt_json_path = os.path.join(BBOX_GT_DIR, f"{base_name}.json")
    
    if not os.path.exists(gt_json_path):
        return None

    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    width, height = gt_data.get("width", 256), gt_data.get("height", 256)
    for frame in gt_data.get("frames", []):
        if frame["frame_index"] == target_frame:
            if frame["has_mask"] and frame["bbox_xyxy"]:
                xmin, ymin, xmax, ymax = frame["bbox_xyxy"]
                return [
                    int(xmin * 1000 / width),
                    int(ymin * 1000 / height),
                    int(xmax * 1000 / width),
                    int(ymax * 1000 / height)
                ]
    return None

def main():
    # 1. 加载映射表
    print("Loading frame mappings...")
    seg_start_map = load_segment_mappings(MAPPING_FILE)

    # 2. 读取模型结果
    with open(INPUT_DETAILS, 'r') as f:
        details_list = json.load(f)

    evaluated_items = []
    all_distances = []
    
    print(f"Evaluating {len(details_list)} samples based on center point distance...")

    for item in tqdm(details_list):
        sample_id = item["sample_id"]
        abs_last_frame = item["frame_indices"][-1]
        
        # 3. 计算 Local 帧索引
        seg_base_name = sample_id.replace("__unoccluded", "").replace("__occluded", "")
        seg_video_name = seg_base_name + ".mp4"
        if seg_video_name not in seg_start_map:
            print(f"Mapping not found for {seg_video_name}")
            continue
        start_frame = seg_start_map[seg_video_name]
        local_frame_idx = abs_last_frame - start_frame
        
        # 4. 获取 GT 坐标
        gt_json_path = os.path.join(BBOX_GT_DIR, f"{seg_base_name}.json")
        gt_bbox = None
        if os.path.exists(gt_json_path):
            with open(gt_json_path, 'r') as f:
                gt_data = json.load(f)
            width, height = gt_data.get("width", 256), gt_data.get("height", 256)
            for frame in gt_data.get("frames", []):
                if frame["frame_index"] == local_frame_idx:
                    if frame["has_mask"] and frame["bbox_xyxy"]:
                        xmin, ymin, xmax, ymax = frame["bbox_xyxy"]
                        gt_bbox = [
                            int(xmin * 1000 / width),
                            int(ymin * 1000 / height),
                            int(xmax * 1000 / width),
                            int(ymax * 1000 / height)
                        ]
                    break
        
        # 5. 提取 Pred 并计算 Distance
        pred_bbox = extract_pred_bbox(item.get("response_text", ""))
        distance = None
        if pred_bbox and gt_bbox:
            distance = calculate_center_distance(pred_bbox, gt_bbox)
            if distance is not None:
                all_distances.append(distance)
        
        # 6. 记录
        item['evaluation'] = {
            "abs_frame": abs_last_frame,
            "local_frame": local_frame_idx,
            "gt_bbox": gt_bbox,
            "pred_bbox": pred_bbox,
            "center_distance": distance # 新增：中心点距离
        }
        evaluated_items.append(item)

    # 统计指标
    metrics = {
        "mean_distance": np.mean(all_distances) if all_distances else None,
        "accuracy_100": sum(1 for d in all_distances if d is not None and d <= 0.1) / len(all_distances) if all_distances else 0, # 距离小于 100 像素
        "accuracy_200": sum(1 for d in all_distances if d is not None and d <= 0.2) / len(all_distances) if all_distances else 0, # 距离小于 200 像素
        "valid_samples": len(all_distances), # 有效样本的数量
        "total_samples": len(details_list) # 总样本的数量
    }

    # 保存
    with open(OUTPUT_DETAILS, 'w') as f:
        json.dump(evaluated_items, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_METRICS, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*40)
    print(f"评估完成 (基于中心点距离)！")
    print(f"平均中心点距离 (归一化): {metrics['mean_distance']:.4f}")
    print(f"准确率 (距离 < 100): {metrics['accuracy_100']:.2%}")
    print(f"有效样本数: {metrics['valid_samples']} / {metrics['total_samples']}")
    print("="*40)

if __name__ == "__main__":
    main()