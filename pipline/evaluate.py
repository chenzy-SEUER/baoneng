import statistics
from pathlib import Path

from pipline.detector import draw_prediction


def box_iou(box1, box2):
    """输入两个 xyxy 框，返回 IoU。"""
    # 交集的左上角、右下角
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 不相交时，交集宽或高为 0
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def box_iou2(box1, box2):
    """输入两个 xyxy 框(obj, pred)，返回 IoU。"""
    # 交集的左上角、右下角
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 不相交时，交集宽或高为 0
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def gt_coverage(gt_box, pred_box):
    """预测框覆盖真实框的面积比例：交集面积 / 真实框面积。"""
    area = max(0, gt_box[2] - gt_box[0]) * max(0, gt_box[3] - gt_box[1])
    intersection = (
        max(0, min(gt_box[2], pred_box[2]) - max(gt_box[0], pred_box[0]))
        * max(0, min(gt_box[3], pred_box[3]) - max(gt_box[1], pred_box[1]))
    )
    return intersection / area if area > 0 else 0.0


def evaluate_recall(records, predictions, iou_threshold=0.5, vis_dir=None,
                    small_gt_area=None, min_gt_coverage=0.9):
    """默认按 IoU 匹配；small_gt_area 非空时启用小框覆盖率补充规则。"""
    result = {}
    matched_count = 0
    missed_count = 0
    gt_count = 0
    coverage_matched_count = 0
    for record in records:
        objects = record['objects']
        matched_obj_idx = set()
        gt_count += len(objects)
        predict = sorted(predictions[record["id"]], key=lambda x: x["score"], reverse=True)
        for p in predict:
            best_idx, best_iou = -1, 0
            for idx, obj in enumerate(objects):
                if idx in matched_obj_idx:
                    continue
                iou = box_iou(obj['box'], p['box'])
                if best_iou < iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx != -1 and best_iou >= iou_threshold:
                matched_obj_idx.add(best_idx)
                continue
            # 没有满足 IoU 的目标时，才尝试覆盖率；仍保持一对一匹配。
            if small_gt_area is not None:
                coverage_idx, best_coverage = -1, 0.0
                for idx, obj in enumerate(objects):
                    if idx in matched_obj_idx:
                        continue
                    gt_box = obj['box']
                    area = max(0, gt_box[2] - gt_box[0]) * max(0, gt_box[3] - gt_box[1])
                    if not 0 < area <= small_gt_area:
                        continue
                    coverage = gt_coverage(gt_box, p['box'])
                    if coverage > best_coverage:
                        coverage_idx, best_coverage = idx, coverage
                if coverage_idx != -1 and best_coverage >= min_gt_coverage:
                    matched_obj_idx.add(coverage_idx)
                    coverage_matched_count += 1
        if len(objects) - len(matched_obj_idx)>0 and vis_dir is not None:
            save_path = Path(vis_dir) / f"{record['id']}.jpg"
            draw_prediction(record, predictions, save_path)
        matched_count += len(matched_obj_idx)
        missed_count += len(objects) - len(matched_obj_idx)
    result['gt_count'] = gt_count
    result['matched_count'] = matched_count
    result['missed_count'] = missed_count
    result['recall'] = matched_count /gt_count if gt_count > 0 else None
    if small_gt_area is not None:
        result['coverage_matched_count'] = coverage_matched_count

    return result




