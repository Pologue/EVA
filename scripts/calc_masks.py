import cv2
import numpy as np
import os
import glob
import json
import csv

# 1. 定义 PlatforMax 平台上的路径
INPUT_DIR = "/opt/data/share/dataset/masks"  # 共享数据中的视频目录
OUTPUT_DIR = "/opt/data/private/mask_results"  # 保存结果的私有目录

# 确保私有输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 获取所有视频文件（假设是 mp4 格式，如果是 avi 请修改后缀）
video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
print(f"🧐 共找到 {len(video_files)} 个 Mask 视频，开始处理...")

# 用于保存所有视频的综合结果
all_results = {}
csv_data =[["Video Name", "Total Frames", "Max Area", "Max Frame Index"]]

for video_path in video_files:
    video_name = os.path.basename(video_path)
    
    cap = cv2.VideoCapture(video_path)
    frame_areas =[]
    
    # 逐帧读取视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕
            
        # 将视频帧转为单通道灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理：确保背景为 0，物体为 255（以防视频压缩带来灰度噪点）
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 计算白色像素的数量（即 Mask 面积）
        white_area = cv2.countNonZero(binary)
        frame_areas.append(white_area)
        
    cap.release()
    
    if not frame_areas:
        print(f"⚠️ 警告: 视频 {video_name} 无法读取帧，跳过。")
        continue

    # 2. 核心逻辑计算
    # 找到白色面积最大的值，以及它所在的帧索引
    max_area = max(frame_areas)
    max_frame_idx = frame_areas.index(max_area)
    
    visibility_ratios =[] # 不被遮挡比例 (当前面积 / 最大面积)
    occlusion_ratios =[]  # 遮挡比例 (1 - 不被遮挡比例)
    
    for area in frame_areas:
        if max_area == 0:
            visibility = 0.0
            occlusion = 1.0
        else:
            visibility = round(area / max_area, 4)
            occlusion = round(1.0 - visibility, 4)
            
        visibility_ratios.append(visibility)
        occlusion_ratios.append(occlusion)
    
    # 记录该视频的结果
    all_results[video_name] = {
        "total_frames": len(frame_areas),
        "max_area_pixels": max_area,
        "max_frame_index": max_frame_idx,
        "frame_areas": frame_areas,
        "visibility_ratios": visibility_ratios, # 重点：不被遮挡比例
        "occlusion_ratios": occlusion_ratios    # 重点：被遮挡比例
    }
    
    csv_data.append([video_name, len(frame_areas), max_area, max_frame_idx])
    print(f"✅ 完成: {video_name} | 最大面积帧: 第{max_frame_idx}帧 | 最大面积: {max_area}像素")

# 3. 将结果持久化保存到私有目录 (/opt/data/private)
# 保存为详细的 JSON 文件，包含每一帧的数据
json_path = os.path.join(OUTPUT_DIR, "detailed_occlusion_metrics.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4)

# 保存为一个简单的 CSV 汇总表，方便查阅每个视频的"最清晰一帧"在哪
csv_path = os.path.join(OUTPUT_DIR, "video_max_frames_summary.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print(f"\n🎉 任务圆满完成！")
print(f"📂 详细的每一帧遮挡比例数据已保存至: {json_path}")
print(f"📊 视频最大帧汇总表已保存至: {csv_path}")