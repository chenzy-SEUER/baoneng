"""完整 test 集：检测、减少候选框、分类、AP50。根目录执行 python -m scripts.evaluate_two_stage_map。"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time

import torch
from ultralytics import YOLO

from pipline import classifier, data, two_stage
from pipline.detection_metrics import DEFECT_CLASSES, evaluate_map50


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2, help="CPU 线程数")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--test-data", type=Path, nargs="+", help="独立的 VOC 测试数据集目录")
    parser.add_argument("--detector-weights", type=Path, default=ROOT / "output/yolo_detect_50ep/train2/weights/best.pt")
    parser.add_argument("--classifier-weights", type=Path, default=ROOT / "output/classifier_runs/run02_merged/last.pt")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads 必须大于 0")
    torch.set_num_threads(args.threads)

    detector_weights = args.detector_weights.resolve()
    classifier_weights = args.classifier_weights.resolve()
    for path in (detector_weights, classifier_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.test_data:
        test_dirs = [path.resolve() for path in args.test_data]
        records = []
        for path in test_dirs:
            loaded, summary = data.load_dataset(path)
            if not loaded or summary["missing_annotations"] or summary["errors"]:
                raise ValueError(f"测试数据为空或图片标注不完整：{path}，{summary}")
            records.extend(sorted(loaded, key=lambda record: record["id"]))
        if len({record["id"] for record in records}) != len(records):
            raise ValueError("测试数据中存在重复图片 ID")
        test_source = {"test_data": [str(path) for path in test_dirs]}
    else:
        manifest = ROOT / "output/dataset/manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        records = [r for r in data.load_manifest(manifest) if r["split"] == "test"]
        if not records:
            raise ValueError("manifest 中没有 test 数据")
        test_source = {"manifest": str(manifest)}
    for record in records:
        path = Path(record["image_path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        record["image_path"] = str(path)

    device = torch.device(args.device)
    model, transform, checkpoint = classifier.build_classifier_with_weights(classifier_weights, device=device)
    idx_to_class = {
        idx: name for name, idx in checkpoint["class_to_idx"].items()
    }
    if set(idx_to_class.values()) != set(DEFECT_CLASSES) | {"non_defect"}:
        raise ValueError(f"分类权重的类别不符合当前评估口径：{idx_to_class}")
    detector_model = YOLO(str(detector_weights))
    conf, nms_iou, imgsz = 0.03, 0.7, 640
    out_dir = args.out_dir or ROOT / "output/two_stage_predictions" / datetime.now().strftime("test_rmhalf_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "split": "test", "image_count": len(records),
        "datasets": dict(Counter(r["dataset"] for r in records)),
        **test_source, "detector_weights": str(detector_weights),
        "classifier_weights": str(classifier_weights), "device": str(device),
        "conf": conf, "nms_iou": nms_iou, "imgsz": imgsz,
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
        raw_image_count, boxes, accepted, removed = two_stage.predict_image(
            record, detector_model, model, transform, idx_to_class,
            device=device, conf=conf, nms_iou=nms_iou, imgsz=imgsz,
            verbose=False, reduce_candidates=True,
        )
        raw_count += raw_image_count
        image_id = record["id"]
        candidates[image_id] = boxes
        kept[image_id] = accepted
        rejected[image_id] = removed
        # 每张图保存一行，保留中断前完成的结果。
        with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "id": image_id, "candidates": boxes,
                "kept": accepted, "rejected": removed,
            }) + "\n")
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
