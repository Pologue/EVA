import json
import re

def clean_filename(filename):
    """
    去掉开头的 8位哈希 + '-'
    例如: 8ae65a99-xxx.mp4 -> xxx.mp4
    """
    return re.sub(r'^[0-9a-fA-F]{8}-', '', filename)

def process_item(item):
    # 处理 file_upload
    if "file_upload" in item:
        original_name = item["file_upload"]
        cleaned_name = clean_filename(original_name)
        item["file_upload"] = cleaned_name

        # 同步修改 data.video
        if "data" in item and "video" in item["data"]:
            item["data"]["video"] = f"./videos/{cleaned_name}"

    return item


def main(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果是数组（你说是整个 json 文件的数组）
    if isinstance(data, list):
        data = [process_item(item) for item in data]
    else:
        data = process_item(data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main("/home/pologue/Desktop/academic/Senior2/dataset/annotations.json", "/home/pologue/Desktop/academic/Senior2/dataset/annotations_new.json")