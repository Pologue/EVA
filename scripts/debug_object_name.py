import json
import os

# ================= 配置区域 =================
# 使用你之前生成的评估结果文件
EVAL_DETAILS_FILE = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/details_evaluated.json"
# 输出分析结果的文件
OUTPUT_ANALYSIS = "/opt/data/private/dataset/eval_results/from_start/20260410_080023/semantic_consistency_report.txt"
# ===========================================

def extract_predicted_object(response_text):
    """从 response_text 中提取 target_object 字段"""
    try:
        # 寻找 JSON 块
        import re
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            res_json = json.loads(match.group())
            return res_json.get("target_object", "N/A").lower()
    except:
        return "ERROR_PARSING"
    return "N/A"

def main():
    if not os.path.exists(EVAL_DETAILS_FILE):
        print(f"找不到文件: {EVAL_DETAILS_FILE}")
        return

    with open(EVAL_DETAILS_FILE, 'r') as f:
        data = json.load(f)

    report_lines = []
    mismatch_count = 0
    total_count = len(data)

    header = f"{'Index':<5} | {'Expected (GT Label)':<25} | {'Model Predicted':<25} | {'Match?'}"
    report_lines.append(header)
    report_lines.append("-" * 80)

    for i, item in enumerate(data):
        # 1. 获取期望的 Label (真值)
        # 注意：这里我们从 prompt_metadata 里的 segment_name 提取，或者直接看你的 action_videos 标签
        # 假设你的 sample_id 结构中最后一部分就是目标物，或者从 metadata 获取
        expected_label = item.get('prompt_metadata', {}).get('label', "Unknown")
        # 如果 metadata 里没有 label 字段，我们从 segment_name 尝试提取
        if expected_label == "Unknown":
            seg_name = item.get('prompt_metadata', {}).get('segment_name', "")
            # 提取 segX_ 之后的部分作为 label
            if "_seg" in seg_name:
                expected_label = seg_name.split("_seg")[1].split(".mp4")[0][2:].replace("_", " ")

        # 2. 获取模型预测的物体
        predicted_obj = extract_predicted_object(item.get("response_text", ""))

        # 3. 简单逻辑判断是否匹配
        # 如果真值包含在预测值里，或者预测包含在真值里，视为语义匹配
        is_match = (expected_label.lower() in predicted_obj.lower()) or \
                   (predicted_obj.lower() in expected_label.lower())
        
        match_str = "YES" if is_match else "NO"
        if not is_match:
            mismatch_count += 1

        line = f"{i:<5} | {expected_label[:25]:<25} | {predicted_obj[:25]:<25} | {match_str}"
        report_lines.append(line)

    # 4. 汇总
    summary = f"\n全量语义核对汇总:"
    summary += f"\n总样本数: {total_count}"
    summary += f"\n语义不匹配数: {mismatch_count}"
    summary += f"\n语义匹配率: {((total_count - mismatch_count) / total_count):.2%}"
    
    report_lines.append(summary)

    # 5. 写入文件
    with open(OUTPUT_ANALYSIS, 'w') as f:
        f.write("\n".join(report_lines))

    print(f"全量语义一致性报告已生成: {OUTPUT_ANALYSIS}")
    print(f"估算的语义匹配率: {((total_count - mismatch_count) / total_count):.2%}")

if __name__ == "__main__":
    main()