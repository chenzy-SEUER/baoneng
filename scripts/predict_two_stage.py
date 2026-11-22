import json
from pathlib import Path

import torch
from ultralytics import YOLO

from pipline import classifier, data, detector, two_stage

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "output/dataset/manifest.json"
DETECTOR_WEIGHTS = ROOT / "output/yolo_detect_50ep/train2/weights/best.pt"
CLASSIFIER_WEIGHTS = ROOT / "output/classifier_runs/run02_merged/last.pt"
CANDIDATE_DIR = ROOT / "output/two_stage_predictions/pred/sample05"
CLASSIFIER_DIR = ROOT / "output/two_stage_predictions/classifier/sample05"


def main():
    for path in (MANIFEST, DETECTOR_WEIGHTS, CLASSIFIER_WEIGHTS):
        if not path.is_file():
            raise FileNotFoundError(path)

    test_records = [record for record in data.load_manifest(MANIFEST) if record["split"] == "test"]
    records = []
    for dataset in ("Data10", "Data11"):
        records.extend([record for record in test_records if record["dataset"] == dataset][:5])
    if not records:
        raise ValueError("manifest 中没有 Data10 或 Data11 的 test 记录")
    for record in records:
        path = ROOT / record["image_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        record["image_path"] = str(path)

    for directory in (CANDIDATE_DIR, CLASSIFIER_DIR):
        if directory.exists():
            raise FileExistsError(directory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_model, transform, checkpoint = classifier.build_classifier_with_weights(
        CLASSIFIER_WEIGHTS, device=device,
    )
    idx_to_class = {idx: name for name, idx in checkpoint["class_to_idx"].items()}
    detector_model = YOLO(str(DETECTOR_WEIGHTS))
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=False)
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=False)

    candidates, kept, rejected = {}, {}, {}
    for record in records:
        image_id = record["id"]
        _, candidates[image_id], kept[image_id], rejected[image_id] = two_stage.predict_image(
            record, detector_model, classifier_model, transform, idx_to_class, device=device,
        )

    (CANDIDATE_DIR / "candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for name, results in (("kept", kept), ("rejected", rejected)):
        (CLASSIFIER_DIR / f"{name}.json").write_text(json.dumps(results), encoding="utf-8")
    for record in records:
        detector.draw_classified_prediction(record, kept, CLASSIFIER_DIR / "vis" / f"{record['id']}.jpg")

    candidate_count = sum(map(len, candidates.values()))
    kept_count = sum(map(len, kept.values()))
    rejected_count = sum(map(len, rejected.values()))
    skipped_count = candidate_count - kept_count - rejected_count
    print(
        f"图片数={len(records)}, 候选数={candidate_count}, "
        f"保留数={kept_count}, 剔除数={rejected_count}, "
        f"无效框跳过数={skipped_count}"
    )


if __name__ == "__main__":
    main()
