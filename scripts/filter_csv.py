import csv
import json


INPUT_CSV_FILE = "/opt/data/private/dataset/mask_results/video_max_frames_summary.csv"
INPUT_JSON_FILE = "/opt/data/private/mask_results/detailed_occlusion_metrics.json"
OUTPUT_CSV_FILE = "/opt/data/private/dataset/mask_results/video_max_frames_nonzero.csv"
OUTPUT_JSON_FILE = "/opt/data/private/mask_results/detailed_occlusion_metrics_nonzero.json"


def main() -> None:
	with open(INPUT_CSV_FILE, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)

	filtered_rows = [row for row in rows if float(row["Max Area"]) != 0]
	filtered_videos = {row["Video Name"] for row in filtered_rows}

	with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
		all_results = json.load(f)

	filtered_results = {
		video_name: video_data
		for video_name, video_data in all_results.items()
		if video_name in filtered_videos
	}

	fieldnames = reader.fieldnames or []
	with open(OUTPUT_CSV_FILE, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(filtered_rows)

	with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
		json.dump(filtered_results, f, ensure_ascii=False, indent=4)

	print(f"kept {len(filtered_rows)} rows from {len(rows)} total rows")
	print(f"nonzero summary saved to {OUTPUT_CSV_FILE}")
	print(f"filtered json saved to {OUTPUT_JSON_FILE}")


if __name__ == "__main__":
	main()