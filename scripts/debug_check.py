import json

EVAL_JSON = "/opt/data/private/dataset/eval_results/from_start/20260410_070057/details_evaluated.json"

with open(EVAL_JSON, 'r') as f:
    data = json.load(f)

print(f"{'Sample ID':<40} | {'GT (XYXY)':<20} | {'Pred (XYXY)':<20} | {'IoU'}")
print("-" * 100)

for item in data[:10]: # 查看前10个
    eval_res = item.get('evaluation', {})
    gt = eval_res.get('gt_bbox_normalized_xyxy')
    pred = eval_res.get('pred_bbox_normalized_xyxy')
    iou = eval_res.get('iou', 0)
    sid = item.get('sample_id', 'N/A')[:38]
    
    print(f"{sid:<40} | {str(gt):<20} | {str(pred):<20} | {iou:.4f}")