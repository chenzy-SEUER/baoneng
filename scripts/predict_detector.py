from pathlib import Path

from pipline import data, detector

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "output/yolo_detect/train/weights/best.pt"
MANIFEST_PATH = ROOT / "output/dataset/manifest.json"
PREDICTIONS_DIR = ROOT / "output/predictions"


def main():
    for path in (WEIGHTS_PATH, MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    records = []
    for split in ("train", "val"):
        records.extend(data.load_dataset_from_manifest(MANIFEST_PATH, split))
    predictions = detector.predict_records(WEIGHTS_PATH, records, PREDICTIONS_DIR / "sample.json")
    for record in records:
        save_path = PREDICTIONS_DIR / "vis" / f"{record['id']}.jpg"
        detector.draw_prediction(record, predictions, save_path)


if __name__ == "__main__":
    main()
