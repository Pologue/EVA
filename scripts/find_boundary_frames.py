import json
import os
import csv

# 1. 配置文件路径
INPUT_JSON = "/opt/data/private/mask_results/detailed_occlusion_metrics.json"
OUTPUT_DIR = "/opt/data/private/mask_results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "occlusion_boundaries.csv")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "occlusion_boundaries.json")

# 阈值设定
THRESHOLD_UNOCCLUDED = 0.6  # 大于此值视为未遮挡
THRESHOLD_OCCLUDED = 0.4    # 小于此值视为被遮挡

print("📂 正在读取之前生成的详细数据...")
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = {}
# 准备 CSV 的表头
csv_rows = [["Video Name", f"Last Unoccluded Frame (> {THRESHOLD_UNOCCLUDED})", f"Last Occluded Frame (< {THRESHOLD_OCCLUDED})"]]

for video_name, metrics in data.items():
    visibility_ratios = metrics.get("visibility_ratios",[])
    
    last_unoccluded = -1
    last_occluded = -1
    
    # 逆序遍历每一帧的比例，找到“最后一个”满足条件的帧
    for i in range(len(visibility_ratios) - 1, -1, -1):
        ratio = visibility_ratios[i]
        
        # 寻找最后一个未遮挡的帧 (Ratio > 0.6)
        if last_unoccluded == -1 and ratio > THRESHOLD_UNOCCLUDED:
            last_unoccluded = i
            
        # 寻找最后一个被遮挡的帧 (Ratio < 0.4)
        if last_occluded == -1 and ratio < THRESHOLD_OCCLUDED:
            last_occluded = i
            
        # 如果两个都找到了，就可以提前结束这个视频的循环
        if last_unoccluded != -1 and last_occluded != -1:
            break
            
    # 处理特殊情况（比如视频全称都被遮挡，或者全程都没被遮挡）
    last_unoccluded_str = last_unoccluded if last_unoccluded != -1 else "N/A"
    last_occluded_str = last_occluded if last_occluded != -1 else "N/A"
    
    # 存入字典
    results[video_name] = {
        "last_unoccluded_frame": last_unoccluded if last_unoccluded != -1 else None,
        "last_occluded_frame": last_occluded if last_occluded != -1 else None
    }
    
    # 存入 CSV 行
    csv_rows.append([video_name, last_unoccluded_str, last_occluded_str])

# 2. 将结果保存到私有目录
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print(f"✅ 处理完成！共分析了 {len(data)} 个视频。")
print(f"📊 边界帧数据(CSV)已保存至: {OUTPUT_CSV}")
print(f"📦 边界帧数据(JSON)已保存至: {OUTPUT_JSON}")