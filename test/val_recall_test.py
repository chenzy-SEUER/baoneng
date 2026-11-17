"""使用已有权重评估完整 val 集：python -m test.val_recall_test。"""

import json
from pathlib import Path

from ultralytics import YOLO

from pipline import data
from pipline.evaluate import evaluate_recall


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "output/dataset/manifest.json"

WEIGHTS_PATH = ROOT / "output/yolo_detect_50ep/train2/weights/best.pt"
PREDICTIONS_PATH = ROOT / "output/predictions/val_50ep.json"
REPORT_PATH = ROOT / "output/predictions/val_recall_50ep.json"

# 推理参数；NMS_IOU 与下面的评估 IoU 是不同的参数。
CONF = 0.03
NMS_IOU = 0.7
IMGSZ = 640
DEVICE = "0"
SMALL_GT_AREA = 12 * 12
MIN_GT_COVERAGE = 0.9


def main():
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"权重不存在：{WEIGHTS_PATH}")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"清单不存在：{MANIFEST_PATH}")

    records = [
        record for record in data.load_manifest(MANIFEST_PATH)
        if record["split"] == "val"
    ]
    if not records:
        raise ValueError("manifest 中没有 val 记录")

    # 先检查图片路径，避免工作目录不同或缺图时触发自动下载。
    image_paths = []
    for record in records:
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"图片不存在：{image_path}")
        record["image_path"] = str(image_path)
        image_paths.append(image_path)

    print(f"验证集图片数：{len(records)}", flush=True)
    print(f"conf={CONF}, NMS IoU={NMS_IOU}, imgsz={IMGSZ}, device={DEVICE}")
    model = YOLO(str(WEIGHTS_PATH))
    predictions = {}
    for index, (record, image_path) in enumerate(zip(records, image_paths), start=1):
        # 当前 predict_records 没有 NMS IoU 参数，此处显式指定以固定实验设置。
        result = model.predict(
            source=str(image_path), conf=CONF, iou=NMS_IOU,
            imgsz=IMGSZ, device=DEVICE, verbose=False,
        )[0]
        boxes = result.boxes
        predictions[record["id"]] = [
            {"box": box, "score": score}
            for box, score in zip(
                boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()
            )
        ]
        if index % 50 == 0 or index == len(records):
            print(f"推理进度：{index}/{len(records)}", flush=True)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 两个评估阈值复用同一份预测，不重复推理。
    results = {}
    small_object_results = {}
    for threshold in (0.5, 0.3):
        result = evaluate_recall(records, predictions, iou_threshold=threshold,vis_dir = ROOT / f"output/predictions/missed_50ep/iou_{threshold}")
        results[str(threshold)] = result
        small_result = evaluate_recall(
            records, predictions, iou_threshold=threshold,
            small_gt_area=SMALL_GT_AREA, min_gt_coverage=MIN_GT_COVERAGE,
            vis_dir=ROOT / f"output/predictions/missed_50ep_small/iou_{threshold}",
        )
        small_object_results[str(threshold)] = small_result
        print(
            f"评估 IoU={threshold}：GT={result['gt_count']}，"
            f"匹配={result['matched_count']}，漏检={result['missed_count']}，"
            f"召回率={result['recall']:.2%}"
            if result["recall"] is not None
            else f"评估 IoU={threshold}：没有真实框，召回率未定义"
        )
        print(f"小框覆盖补充规则 IoU={threshold}：{small_result}")

    report = {
        "weights_path": str(WEIGHTS_PATH),
        "manifest_path": str(MANIFEST_PATH),
        "predictions_path": str(PREDICTIONS_PATH),
        "split": "val",
        "image_count": len(records),
        "prediction_count": sum(len(boxes) for boxes in predictions.values()),
        "inference": {
            "conf": CONF, "nms_iou": NMS_IOU, "imgsz": IMGSZ, "device": DEVICE,
        },
        "results": results,
        "small_object_rule": {
            "max_gt_area": SMALL_GT_AREA,
            "min_gt_coverage": MIN_GT_COVERAGE,
            "coverage_definition": "intersection_area / gt_area",
        },
        "small_object_results": small_object_results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"预测已保存：{PREDICTIONS_PATH}")
    print(f"评估已保存：{REPORT_PATH}")
    return report


if __name__ == "__main__":
    main()
