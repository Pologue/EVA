import csv


INPUT_CSV_FILE = "/opt/data/private/dataset/mask_results/video_max_frames_summary.csv"
OUTPUT_CSV_FILE = "/opt/data/private/dataset/mask_results/video_max_frames_nonzero.csv"


def main() -> None:
	with open(INPUT_CSV_FILE, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)

	filtered_rows = [row for row in rows if float(row["Max Area"]) != 0]

	fieldnames = reader.fieldnames or []
	with open(OUTPUT_CSV_FILE, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(filtered_rows)

	print(f"kept {len(filtered_rows)} rows from {len(rows)} total rows")
	print(f"nonzero summary saved to {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
	main()