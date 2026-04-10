import json
import re
import os
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# INPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_070057/details.json"
# INPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/details.json"
INPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_140446/details.json"
MAPPING_FILE = "/opt/data/private/dataset/action_videos.jsonl"
BBOX_GT_DIR = "/opt/data/private/dataset/mask_results/bboxes/"

# OUTPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_070057/details_evaluated.json"
# OUTPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/details_evaluated.json"
OUTPUT_DETAILS = "/opt/data/private/dataset/eval_results/from_start/20260410_140446/details_evaluated.json"
# OUTPUT_METRICS = "/opt/data/private/dataset/eval_results/from_start/20260410_070057/evaluation_metrics.json"
# OUTPUT_METRICS = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/evaluation_metrics.json"
OUTPUT_METRICS = "/opt/data/private/dataset/eval_results/from_start/20260410_140446/evaluation_metrics.json"
# ===========================================

def calculate_iou(boxA, boxB):
    """[xmin, ymin, xmax, ymax]"""
    if not boxA or not boxB: return 0
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea <= 0: return 0
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

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
            # 获取文件名，例如: AddIceCubes...seg0_Ice_Cubes.mp4
            seg_name = os.path.basename(data['segment_video'])
            mapping[seg_name] = data['start']
    return mapping

def main():
    # 1. 加载映射表
    print("Loading frame mappings...")
    seg_start_map = load_segment_mappings(MAPPING_FILE)

    # 2. 读取模型结果
    with open(INPUT_DETAILS, 'r') as f:
        details_list = json.load(f)

    evaluated_items = []
    all_ious = []
    log_errors = []

    print(f"Evaluating {len(details_list)} samples with frame offset calibration...")

    for item in tqdm(details_list):
        sample_id = item["sample_id"]
        # 获取 ChatGPT 输入的绝对帧索引 (e.g., 146)
        abs_last_frame = item["frame_indices"][-1]
        
        # 解析出对应的子视频文件名
        # AddIceCubes..._unoccluded -> AddIceCubes...Ice_Cubes.mp4
        seg_base_name = sample_id.replace("__unoccluded", "").replace("__occluded", "")
        seg_video_name = seg_base_name + ".mp4"
        
        # 3. 计算 Mask 侧的 Local 帧索引
        if seg_video_name not in seg_start_map:
            log_errors.append(f"Mapping not found for {seg_video_name}")
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
        
        # 5. 提取 Pred 并计算 IoU
        pred_bbox = extract_pred_bbox(item.get("response_text", ""))
        iou = 0
        if pred_bbox and gt_bbox:
            iou = calculate_iou(pred_bbox, gt_bbox)
            all_ious.append(iou)
        
        # 6. 记录
        item['evaluation'] = {
            "abs_frame": abs_last_frame,
            "local_frame": local_frame_idx,
            "gt_bbox": gt_bbox,
            "pred_bbox": pred_bbox,
            "iou": iou
        }
        evaluated_items.append(item)

    # 统计指标
    metrics = {
        "mean_iou": np.mean(all_ious) if all_ious else 0,
        "accuracy_05": sum(1 for i in all_ious if i >= 0.5) / len(details_list) if details_list else 0,
        "accuracy_07": sum(1 for i in all_ious if i >= 0.7) / len(details_list) if details_list else 0,
        "valid_samples": len(all_ious),
        "total_samples": len(details_list)
    }

    # 保存
    with open(OUTPUT_DETAILS, 'w') as f:
        json.dump(evaluated_items, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_METRICS, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*40)
    print(f"评估完成 (已校准帧偏移)！")
    print(f"平均 IoU: {metrics['mean_iou']:.4f}")
    print(f"有效评估样本数: {metrics['valid_samples']}")
    print(f"Acc@0.5: {metrics['accuracy_05']:.2%}")
    print("="*40)

if __name__ == "__main__":
    main()