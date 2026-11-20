"""对分类后的 xyxy 检测结果计算 AP50，不使用小框覆盖率补充规则。"""

import math

import numpy as np
from ultralytics.utils.metrics import compute_ap

from pipline.evaluate import box_iou


DEFECT_CLASSES = ("PID", "crack", "diode", "hotspot", "shelter", "zhedang")


def evaluate_map50(records, predictions, score_mode="product"):
    """按类别、置信度降序匹配同图中 IoU 最大的未匹配 GT。

    默认排序分数为 det_score * cls_score；也可指定 detection 使用 det_score。
    GT 的 GB 合并到 crack。AP 积分复用 Ultralytics 的 101 点插值实现。
    有 GT 的缺陷类别参与 mAP；non_defect 是被剔除的背景，不参与平均。
    """
    if score_mode not in ("product", "detection"):
        raise ValueError("score_mode 必须是 product 或 detection")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("评估记录中存在重复图片 ID")
    if set(ids) != set(predictions):
        raise ValueError("预测 JSON 的图片 ID 必须与评估数据完全一致，空预测也应保留键")

    ground_truth = {label: {} for label in DEFECT_CLASSES}
    ranked = {label: [] for label in DEFECT_CLASSES}
    for record in records:
        image_id = record["id"]
        for obj in record["objects"]:
            label = "crack" if obj["label"] == "GB" else obj["label"]
            if label not in ground_truth:
                raise ValueError(f"未知真实类别：{label}")
            ground_truth[label].setdefault(image_id, []).append(obj["box"])
        for pred in predictions[image_id]:
            label = pred["label"]
            if label not in ranked:
                raise ValueError(f"最终预测中存在未知类别或背景类别：{label}")
            box = pred["box"]
            if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError(f"无效预测框：{image_id}: {box}")
            score = float(pred["det_score"])
            if score_mode == "product":
                score *= float(pred["cls_score"])
            if not math.isfinite(score):
                raise ValueError(f"无效预测分数：{image_id}")
            ranked[label].append((score, image_id, box))

    per_class = {}
    for label in DEFECT_CLASSES:
        gt = ground_truth[label]
        gt_count = sum(len(boxes) for boxes in gt.values())
        matched = {image_id: set() for image_id in gt}
        detections = sorted(ranked[label], key=lambda p: p[0], reverse=True)
        hits = []
        for _, image_id, box in detections:
            best_index, best_iou = -1, 0.0
            for index, target in enumerate(gt.get(image_id, [])):
                if index in matched[image_id]:
                    continue
                overlap = box_iou(box, target)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            hit = best_index >= 0 and best_iou >= 0.5
            if hit:
                matched[image_id].add(best_index)
            hits.append(hit)
        tp = int(sum(hits))
        fp = len(hits) - tp
        ap = None
        if gt_count:
            if hits:
                cumulative_tp = np.cumsum(hits)
                recall = cumulative_tp / gt_count
                precision = cumulative_tp / np.arange(1, len(hits) + 1)
                ap = float(compute_ap(recall, precision)[0])
            else:
                ap = 0.0
        per_class[label] = {
            "gt_count": gt_count, "prediction_count": len(hits),
            "tp": tp, "fp": fp, "fn": gt_count - tp,
            "precision": tp / len(hits) if hits else 0.0,
            "recall": tp / gt_count if gt_count else None,
            "ap50": ap,
        }
    aps = [v["ap50"] for v in per_class.values() if v["ap50"] is not None]
    return {
        "map50": sum(aps) / len(aps) if aps else None,
        "iou_threshold": 0.5, "score_mode": score_mode,
        "ap_method": "Ultralytics compute_ap: 101-point interpolation + trapezoidal integration",
        "matching": "class-aware, confidence descending, best unmatched GT, one-to-one",
        "label_mapping": {"GB": "crack"}, "per_class": per_class,
    }
