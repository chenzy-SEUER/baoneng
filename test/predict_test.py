from pathlib import Path

from pipline import data, detector

ROOT = Path(__file__).resolve().parents[1]


best_weights_path = ROOT / "output/yolo_detect/train/weights/best.pt"
manifest_path = ROOT / "output/dataset/manifest.json"
sample_json = ROOT / "output/predictions/sample.json"
predict_sample = ROOT / "output/predictions"

assert best_weights_path.is_file(), f"权重不存在：{best_weights_path}"
assert manifest_path.is_file(), f"清单不存在：{manifest_path}"

records = []
for split in ('train', 'val'):
    records.extend(data.load_dataset_from_manifest(manifest_path,split))
predictions = detector.predict_records(best_weights_path, records, sample_json)


for record in records:
    save_path = ROOT / "output/predictions/vis" / f"{record['id']}.jpg"
    detector.draw_prediction(record, predictions, save_path)



















