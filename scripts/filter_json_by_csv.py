import csv
import json


INPUT_CSV_FILE = "/opt/data/private/dataset/mask_results/video_max_frames_nonzero.csv"
INPUT_JSON_FILE = "/opt/data/private/dataset/mask_results/detailed_occlusion_metrics.json"
OUTPUT_JSON_FILE = "/opt/data/private/dataset/mask_results/detailed_occlusion_metrics_nonzero.json"


def main() -> None:
	with open(INPUT_CSV_FILE, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		filtered_videos = {row["Video Name"] for row in reader}

	with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
		all_results = json.load(f)

	filtered_results = {
		video_name: video_data
		for video_name, video_data in all_results.items()
		if video_name in filtered_videos
	}

	with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
		json.dump(filtered_results, f, ensure_ascii=False, indent=4)

	print(f"kept {len(filtered_results)} videos from {len(all_results)} total videos")
	print(f"filtered json saved to {OUTPUT_JSON_FILE}")


if __name__ == "__main__":
	main()