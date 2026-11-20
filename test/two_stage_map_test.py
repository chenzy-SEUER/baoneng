"""完整 test 集：检测、减少候选框、分类、AP50。根目录执行 python -m test.two_stage_map_test。"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time

import torch
from ultralytics import YOLO

from pipline import classifier, data
from pipline.detection_metrics import DEFECT_CLASSES, evaluate_map50


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2, help="CPU 线程数")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads 必须大于 0")
    torch.set_num_threads(args.threads)

    manifest = ROOT / "output/dataset/manifest.json"
    detector_weights = ROOT / "output/yolo_detect_50ep/train2/weights/best.pt"
    classifier_weights = ROOT / "output/classifier_runs/run02_merged/last.pt"
    for path in (manifest, detector_weights, classifier_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    records = [r for r in data.load_manifest(manifest) if r["split"] == "test"]
    if not records:
        raise ValueError("manifest 中没有 test 数据")
    for record in records:
        path = Path(record["image_path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        record["image_path"] = str(path)

    device = torch.device(args.device)
    model, transform, idx_to_class = classifier.build_classifier_with_weights(classifier_weights)
    model.to(device)
    if set(idx_to_class.values()) != set(DEFECT_CLASSES) | {"non_defect"}:
        raise ValueError(f"分类权重的类别不符合当前评估口径：{idx_to_class}")
    detector = YOLO(str(detector_weights))
    out_dir = args.out_dir or ROOT / "output/two_stage_predictions" / datetime.now().strftime("test_rmhalf_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "split": "test", "image_count": len(records),
        "datasets": dict(Counter(r["dataset"] for r in records)),
        "manifest": str(manifest), "detector_weights": str(detector_weights),
        "classifier_weights": str(classifier_weights), "device": str(device),
        "conf": 0.03, "nms_iou": 0.7, "imgsz": 640,
        "candidate_rule": "if count > 10: sort by detection score descending and keep count // 2",
        "image_ids": [r["id"] for r in records],
    }
    write_json(out_dir / "settings.json", settings)
    print(f"输出目录：{out_dir}", flush=True)
    print(f"完整 test 集：{settings['datasets']}，设备：{device}", flush=True)
    candidates, kept, rejected = {}, {}, {}
    raw_count = 0
    start = time.monotonic()
    for index, record in enumerate(records, 1):
        result = detector.predict(
            source=record["image_path"], conf=0.03, iou=0.7,
            imgsz=640, device=str(device), verbose=False,
        )[0]
        boxes = [
            {"box": box, "score": score}
            for box, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())
        ]
        raw_count += len(boxes)
        if len(boxes) > 10:
            boxes.sort(key=lambda p: p["score"], reverse=True)
            boxes = boxes[:len(boxes) // 2]
        image_id = record["id"]
        candidates[image_id] = boxes
        accepted, removed = classifier.classify_candidates(
            model, [record], {image_id: boxes}, transform, idx_to_class, device
        )
        kept.update(accepted)
        rejected.update(removed)
        # 每张图保存一行，可追查未完成运行，不将部分结果当完整 test 结果。
        with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": image_id, "candidates": boxes,
                                     "kept": accepted[image_id], "rejected": removed[image_id]}) + "\n")
        if index % 25 == 0 or index == len(records):
            print(f"进度 {index}/{len(records)}，用时 {time.monotonic() - start:.1f}s", flush=True)

    write_json(out_dir / "candidates.json", candidates)
    write_json(out_dir / "kept.json", kept)
    write_json(out_dir / "rejected.json", rejected)
    candidate_count = sum(map(len, candidates.values()))
    kept_count = sum(map(len, kept.values()))
    rejected_count = sum(map(len, rejected.values()))
    metrics = evaluate_map50(records, kept)
    report = {
        "settings": settings, "raw_candidate_count": raw_count,
        "candidate_count": candidate_count, "kept_count": kept_count,
        "rejected_count": rejected_count,
        "invalid_box_count": candidate_count - kept_count - rejected_count,
        "elapsed_seconds": time.monotonic() - start,
        "metrics": metrics,
        "detection_score_metrics": evaluate_map50(records, kept, score_mode="detection"),
    }
    write_json(out_dir / "report.json", report)
    print("类别          GT     预测     TP     FP     FN     AP50", flush=True)
    for label, values in metrics["per_class"].items():
        ap_text = f"{values['ap50']:.4f}" if values["ap50"] is not None else "N/A"
        print(f"{label:<12} {values['gt_count']:5d} {values['prediction_count']:7d} "
              f"{values['tp']:6d} {values['fp']:6d} {values['fn']:6d} {ap_text}")
    print(f"mAP@0.5（检测分数×分类分数）：{metrics['map50']:.4f}")
    print(f"mAP@0.5（仅检测分数排序）：{report['detection_score_metrics']['map50']:.4f}")
    print(f"原始候选={raw_count}，减少后={candidate_count}，保留={kept_count}，剔除={rejected_count}")
    return report


if __name__ == "__main__":
    main()
