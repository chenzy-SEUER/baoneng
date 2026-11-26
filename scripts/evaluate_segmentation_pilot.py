"""CPU 对照：原始 GT 上评估检测框、分割框及掩膜外接框，并固定分类器。"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from math import floor, ceil
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

from pipline.classifier import build_classifier_with_weights
from pipline.evaluate import box_iou, gt_coverage

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def matched_indices(objects, predictions, threshold):
    matches = {}
    for j in sorted(range(len(predictions)), key=lambda j: predictions[j]["score"], reverse=True):
        candidates = [(box_iou(g["box"], predictions[j]["box"]), i)
                      for i, g in enumerate(objects) if i not in matches]
        if not candidates:
            break
        # Same GT tie order as the existing recall evaluator.
        overlap, i = max(candidates, key=lambda item: item[0])
        if overlap >= threshold:
            matches[i] = j
    return matches


def summarize(records, predictions):
    result = {}
    for filtering in ("all", "top_half_if_over_ten"):
        counts = defaultdict(Counter)
        matches_detail = []
        total_predictions = 0
        for r in records:
            rows = sorted(predictions[r["id"]], key=lambda p: p["score"], reverse=True)
            if filtering != "all" and len(rows) > 10:
                rows = rows[:len(rows)//2]
            total_predictions += len(rows)
            matches = {t: matched_indices(r["objects"], rows, t) for t in (.3, .5)}
            for i, g in enumerate(r["objects"]):
                c = counts[g["label"]]
                c["gt"] += 1
                expected = "crack" if g["label"] == "GB" else g["label"]
                for t, mapping in matches.items():
                    if i in mapping:
                        p = rows[mapping[i]]
                        c[f"matched_iou{t}"] += 1
                        c[f"classified_correct_iou{t}"] += int(p["label"] == expected)
                area = (g["box"][2]-g["box"][0])*(g["box"][3]-g["box"][1])
                eligible = [p for p in rows if
                            (p["box"][2]-p["box"][0])*(p["box"][3]-p["box"][1]) <= 4*area]
                coverage = max((gt_coverage(g["box"], p["box"]) for p in eligible), default=0.)
                c["oracle_coverage90_area_le4x"] += int(coverage >= .9)
                p = rows[matches[.3][i]] if i in matches[.3] else None
                matches_detail.append({"id": r["id"], "object_index": i, "gt_label": g["label"],
                                       "gt_box": g["box"], "matched_iou03": p,
                                       "best_coverage_area_le4x": coverage})
        # Explicit zero fields keep absent predictions distinguishable from missing data.
        fields = ["gt", "matched_iou0.3", "matched_iou0.5", "classified_correct_iou0.3",
                  "classified_correct_iou0.5", "oracle_coverage90_area_le4x"]
        per_class = {label: {k: c[k] for k in fields} for label, c in counts.items()}
        result[filtering] = {"prediction_count": total_predictions, "per_class": per_class,
                             "totals": {k: sum(c[k] for c in per_class.values()) for k in fields},
                             "details": matches_detail}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "output/segmentation_pilot_20261126_reviewed")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "output/segmentation_pilot_20261126_runs")
    args = parser.parse_args()
    dataset = args.dataset_dir.resolve()
    runs = args.runs_dir.resolve()
    weights = {k: runs / k / "weights/best.pt" for k in ("detect", "segment")}
    classifier_weights = ROOT / "output/classifier_runs/run02_merged/last.pt"
    records = [r for r in json.loads((dataset / "reference_manifest.json").read_text()) if r["split"] == "val"]
    provenance = {p["id"]: p for p in json.loads((dataset / "experiment.json").read_text())["annotations"]}
    for path in [*weights.values(), classifier_weights]:
        if not path.is_file():
            raise FileNotFoundError(path)
    image_paths = {}
    for r in records:
        images = list((dataset / "segment/images/val").glob(r["id"] + ".*"))
        if len(images) != 1:
            raise ValueError(f"图片缺失或重名：{r['id']}")
        image_paths[r["id"]] = images[0]
        assert hashlib.sha256(images[0].read_bytes()).hexdigest() == provenance[r["id"]]["image_sha256"]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    predictions = {k: {} for k in ("detect_box", "segment_box", "segment_mask_box")}
    empty_masks = 0
    for task, weight in weights.items():
        model = YOLO(str(weight))
        for number, r in enumerate(records, 1):
            result = model.predict(str(image_paths[r["id"]]), device="cpu", conf=.03, iou=.7,
                                   imgsz=640, max_det=300, retina_masks=True, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            key = "detect_box" if task == "detect" else "segment_box"
            predictions[key][r["id"]] = [{"box": b, "score": s, "candidate_index": i}
                                          for i, (b, s) in enumerate(zip(boxes, scores))]
            if task == "segment":
                mask_boxes = []
                if boxes:
                    assert result.masks is not None and len(result.masks.data) == len(boxes)
                    assert tuple(result.masks.data.shape[1:]) == tuple(result.orig_shape)
                    for i, (mask, score) in enumerate(zip(result.masks.data.cpu(), scores)):
                        ys, xs = torch.where(mask > .5)
                        if not len(xs):
                            empty_masks += 1
                            continue
                        mask_boxes.append({"box": [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
                                           "score": score, "candidate_index": i, "mask_area": len(xs)})
                predictions["segment_mask_box"][r["id"]] = mask_boxes
            print(f"{task} {number}/{len(records)}: {len(boxes)} candidates", flush=True)
        del model
    write_json(args.out_dir / "stage1_predictions.json", predictions)
    model, transform, checkpoint = build_classifier_with_weights(classifier_weights, torch.device("cpu"))
    model.eval()
    names = {v: k for k, v in checkpoint["class_to_idx"].items()}
    done = 0
    for r in records:
        jobs = []
        with Image.open(image_paths[r["id"]]) as source:
            image = source.convert("RGB")
            for kind in predictions:
                for p in predictions[kind][r["id"]]:
                    b = p["box"]
                    crop_box = [floor(max(0,b[0])), floor(max(0,b[1])),
                                ceil(min(image.width,b[2])), ceil(min(image.height,b[3]))]
                    assert crop_box[0] < crop_box[2] and crop_box[1] < crop_box[3]
                    p["crop_box"] = crop_box
                    jobs.append((p, transform(image.crop(tuple(crop_box)))))
        with torch.inference_mode():
            for start in range(0, len(jobs), 8):
                batch = jobs[start:start+8]
                probs = model(torch.stack([j[1] for j in batch])).softmax(dim=1).cpu()
                for (p, _), scores in zip(batch, probs):
                    p["label"] = names[scores.argmax().item()]
                    p["cls_score"] = scores.max().item()
        done += len(jobs)
        print(f"classified {r['id']}: {len(jobs)} crops; cumulative {done}", flush=True)
    summaries = {kind: summarize(records, rows) for kind, rows in predictions.items()}
    report = {
        "settings": {"images": len(records), "split": "val", "device": "cpu", "conf": .03,
                     "nms_iou": .7, "imgsz": 640, "max_det": 300,
                     "reference": "Original XML-derived GT boxes/classes, not the new polygon envelopes",
                     "mask_box": "Binary mask >0.5; original resolution; xyxy enclosing nonzero pixels; xmax/ymax exclusive",
                     "coverage_metric": "Oracle: any box covers >=90% GT, with box area <=4x GT; NOT one-to-one recall",
                     "classification": "Existing fixed classifier; exact RGB crop, existing stretch transform; GB maps to crack",
                     "matching": "Confidence descending, best unmatched GT across classes; class-agnostic; one-to-one",
                     "classifier_weights": str(classifier_weights.relative_to(ROOT)),
                     "dataset_dir": str(dataset), "runs_dir": str(runs),
                     "weights_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in weights.items()}},
        "empty_segment_masks": empty_masks, "summaries": summaries,
    }
    write_json(args.out_dir / "classified_predictions.json", predictions)
    write_json(args.out_dir / "report.json", report)
    lines = ["分割试验：原始GT上的CPU定位/分类对照", json.dumps(report["settings"], ensure_ascii=False), ""]
    for kind, summary in summaries.items():
        for filtering, metrics in summary.items():
            line = f"{kind} / {filtering}: candidates={metrics['prediction_count']}; totals={metrics['totals']}"
            lines.extend([line, "zhedang: " + str(metrics["per_class"].get("zhedang"))])
            print(line, flush=True)
            print("zhedang", metrics["per_class"].get("zhedang"), flush=True)
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
