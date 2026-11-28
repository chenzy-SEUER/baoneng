"""按模态评估验证集，并按数据来源和标注类别统计漏检。"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS

from pipline.data import load_manifest
from pipline.evaluate import box_iou

#new
EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "output/restart_80_20"
MODALITIES = ("infrared", "visible")
LABEL_ALIASES = {
    "热斑": "hotspot",
    "二极管故障": "diode",
    "二极管": "diode",
    "PID效应": "PID",
    "PID 效应": "PID",
    "鸟屎": "鸟粪",
    "破裂": "crack",
    "破裂/裂缝": "crack",
    "遮挡": "zhedang",
}


def load_validation_data(dataset_dir):
    manifest = dataset_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"清单不存在：{manifest}")
    records = {}
    for record in load_manifest(manifest):
        if record["split"] != "val":
            continue
        image_id = record["id"]
        if image_id in records:
            raise ValueError(f"验证集图片 ID 重复：{image_id}")
        records[image_id] = record
    if not records:
        raise ValueError(f"清单中没有验证集记录：{manifest}")

    # 使用 YOLO 导出的图片，避免依赖清单中原图的机器专属路径。
    images_dir = dataset_dir / "images/val"
    images = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS
    ]
    image_ids = {path.stem for path in images}
    if len(image_ids) != len(images) or image_ids != records.keys():
        missing = sorted(records.keys() - image_ids)
        extra = sorted(image_ids - records.keys())
        raise ValueError(f"验证图片与清单不一致：{images_dir}，缺少 {missing}，额外 {extra}，"
                         f"同名图片 {len(images) - len(image_ids)} 张")
    return records, images_dir


def match_predictions(objects, predictions, conf, match_iou):
    """将通过置信度筛选的预测按分数降序与真值框一对一匹配。"""
    matched = set()
    prediction_count = 0
    for score, box in predictions:
        if score < conf:
            break
        prediction_count += 1
        overlap, index = max(
            ((box_iou(obj["box"], box), index)
             for index, obj in enumerate(objects) if index not in matched),
            default=(0.0, -1),
        )
        if overlap >= match_iou:
            matched.add(index)
    return matched, prediction_count


def recall_counts(total, matched):
    return {
        "gt_count": total,
        "matched_count": matched,
        "missed_count": total - matched,
        "recall": matched / total if total else None,
    }


def predict_validation(model, records, images_dir, args, tile_data):
    tile_records, inference_dir = tile_data if tile_data else (records, images_dir)
    seen = set()
    by_source = defaultdict(list)
    for result in model.predict(
        source=str(inference_dir), stream=True, batch=args.batch, device=args.device,
        imgsz=args.imgsz, conf=min(args.conf), iou=args.nms_iou, max_det=300,
        verbose=False,
    ):
        image_id = Path(result.path).stem
        if image_id in seen or image_id not in tile_records:
            raise ValueError(f"推理返回了重复或未知的图片：{result.path}")
        seen.add(image_id)
        boxes = result.boxes.xyxy.cpu()
        scores = result.boxes.conf.cpu()
        if tile_data:
            tile = tile_records[image_id]
            x, y = tile["offset"]
            by_source[tile["source_image_id"]].append((boxes + boxes.new_tensor([x, y, x, y]), scores))
        else:
            yield image_id, boxes.tolist(), scores.tolist()
        if len(seen) % 150 == 0 or len(seen) == len(tile_records):
            print(f"推理进度 {len(seen)}/{len(tile_records)}", flush=True)
    if seen != tile_records.keys():
        raise ValueError(f"部分验证图片没有推理结果：{sorted(tile_records.keys() - seen)}")
    if tile_data:
        for image_id, parts in by_source.items():
            boxes = torch.cat([part[0] for part in parts])
            scores = torch.cat([part[1] for part in parts])
            keep = nms(boxes, scores, args.nms_iou)
            yield image_id, boxes[keep].tolist(), scores[keep].tolist()


def evaluate(records, images_dir, weights, args, tile_data=None):
    model = YOLO(str(weights))
    totals = {"by_dataset": Counter(), "by_class": Counter()}
    sizes = defaultdict(list)
    for record in records.values():
        totals["by_dataset"][record["dataset"]] += len(record["objects"])
        scale = args.imgsz / max(record["width"], record["height"])
        for obj in record["objects"]:
            label = LABEL_ALIASES.get(obj["label"], obj["label"])
            totals["by_class"][label] += 1
            x1, y1, x2, y2 = obj["box"]
            if tile_data is None:
                sizes[label].append(min(x2 - x1, y2 - y1) * scale)

    hits = {conf: {group: Counter() for group in totals} for conf in args.conf}
    prediction_counts = Counter()
    for image_id, boxes, scores in predict_validation(model, records, images_dir, args, tile_data):
        record = records[image_id]
        objects = record["objects"]
        predictions = sorted(zip(scores, boxes), reverse=True)
        for conf in args.conf:
            matched, count = match_predictions(objects, predictions, conf, args.match_iou)
            prediction_counts[conf] += count
            hits[conf]["by_dataset"][record["dataset"]] += len(matched)
            for index in matched:
                label = objects[index]["label"]
                hits[conf]["by_class"][LABEL_ALIASES.get(label, label)] += 1

    thresholds = {}
    for conf in args.conf:
        groups = {
            group: {name: recall_counts(total, hits[conf][group][name])
                    for name, total in counts.most_common()}
            for group, counts in totals.items()
        }
        for name, edges in sizes.items():
            groups["by_class"][name]["median_short_side_px"] = round(median(edges), 1)
        gt_count = sum(totals["by_class"].values())
        matched_count = sum(hits[conf]["by_class"].values())
        prediction_count = prediction_counts[conf]
        thresholds[str(conf)] = {
            **recall_counts(gt_count, matched_count),
            "prediction_count": prediction_count,
            "precision": matched_count / prediction_count if prediction_count else None,
            **groups,
        }
    report = {
        "weights": str(weights),
        "manifest": str(images_dir.parent.parent / "manifest.json"),
        "image_count": len(records),
        "checkpoint_validation_metrics": (model.ckpt or {}).get("train_metrics"),
        "thresholds": thresholds,
    }
    if tile_data:
        report["tile_manifest"] = str(tile_data[1].parent.parent / "manifest.json")
        report["tile_count"] = len(tile_data[0])
    return report


def print_report(result):
    def percent(value):
        return f"{value:.2%}" if value is not None else "—"

    for conf, counts in result["thresholds"].items():
        print(f"置信度 ≥ {conf}，命中 {counts['matched_count']}/{counts['gt_count']}，"
              f"漏检 {counts['missed_count']}，召回率 {percent(counts['recall'])}，"
              f"精确率 {percent(counts['precision'])}")
        for group, title in (("by_dataset", "按数据来源"), ("by_class", "按标注类别")):
            print(title)
            for name, row in counts[group].items():
                print(f"  {name}: {row['matched_count']}/{row['gt_count']}，"
                      f"漏检 {row['missed_count']}，召回率 {percent(row['recall'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("both", *MODALITIES), default="both")
    parser.add_argument("--dataset-dir", type=Path, default=EXPERIMENT_DIR / "by_modality")
    parser.add_argument("--visible-tiles-dir", type=Path, help="可见光切片数据目录，预测合并回原图后评估")
    for modality in MODALITIES:
        parser.add_argument(
            f"--{modality}-weights", type=Path,
            default=EXPERIMENT_DIR / "detector_runs" / modality / "train/weights/best.pt",
        )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--conf", type=float, nargs="+", default=[0.10, 0.25])
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--report", type=Path, help="可选的 JSON 报告路径")
    args = parser.parse_args()
    if min(args.imgsz, args.batch, args.threads) < 1:
        parser.error("--imgsz、--batch、--threads 必须大于 0")
    if any(not 0 < value <= 1 for value in (*args.conf, args.match_iou, args.nms_iou)):
        parser.error("置信度和 IoU 阈值必须在 (0, 1] 内")
    args.conf = sorted(set(args.conf))
    torch.set_num_threads(args.threads)

    modalities = MODALITIES if args.modality == "both" else (args.modality,)
    inputs = {}
    for modality in modalities:
        weights = getattr(args, f"{modality}_weights").resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"权重不存在：{weights}")
        records, images_dir = load_validation_data((args.dataset_dir / modality).resolve())
        tile_data = None
        if modality == "visible" and args.visible_tiles_dir is not None:
            tiles, tile_images_dir = load_validation_data(args.visible_tiles_dir.resolve())
            if {tile["source_image_id"] for tile in tiles.values()} != records.keys():
                raise ValueError("切片清单的原图 ID 与可见光验证集不一致")
            for tile in tiles.values():
                original = records[tile["source_image_id"]]
                x, y = tile["offset"]
                if not (0 <= x < x + tile["width"] <= original["width"]
                        and 0 <= y < y + tile["height"] <= original["height"]):
                    raise ValueError(f"切片超出原图范围：{tile['id']}")
            tile_data = tiles, tile_images_dir
        inputs[modality] = records, images_dir, weights, tile_data

    report = {
        "inference": {
            "device": args.device, "imgsz": args.imgsz, "batch": args.batch,
            "conf": args.conf, "match_iou": args.match_iou, "nms_iou": args.nms_iou,
        },
        "label_aliases": LABEL_ALIASES,
        "modalities": {},
    }
    for modality, (records, images_dir, weights, tile_data) in inputs.items():
        print(f"评估 {modality} 验证集，共 {len(records)} 张图", flush=True)
        result = evaluate(records, images_dir, weights, args, tile_data)
        report["modalities"][modality] = result
        print_report(result)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"报告已保存：{args.report}")


if __name__ == "__main__":
    main()
