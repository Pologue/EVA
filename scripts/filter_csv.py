import pandas as pd


INPUT_CSV_FILE = "/opt/data/private/dataset/mask_results/video_max_frames_summary.csv"
OUTPUT_CSV_FILE = "/opt/data/private/dataset/mask_results/video_max_frames_nonzero.csv"

# 1. 读取CSV文件
df = pd.read_csv(INPUT_CSV_FILE) # 请将 'your_file.csv' 替换为你的文件名

# 2. 筛选第三列不为0的数据
# 方法一：如果知道第三列的列名（例如 'ColumnC'），这是最推荐的方式
result = df[df['Max Area'] != 0]

# 方法二：如果不知道列名，可以通过位置索引来获取（索引从0开始，所以第三列是2）
# result = df[df.iloc[:, 2] != 0].iloc[:, 2]

# 3. 打印结果
print(result)

result.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"nonzero summary saved to {OUTPUT_CSV_FILE}")