from pipline import classifier, detector


def predict_image(
    record, detector_model, classifier_model, transform, idx_to_class, *,
    device, conf=0.03, imgsz=640, nms_iou=None, verbose=None,
    reduce_candidates=False,
):
    """逐图检测和分类，返回原始候选数、送分类的候选、保留框和剔除框。"""
    candidates = detector.predict_image(
        detector_model, record["image_path"], conf=conf, imgsz=imgsz,
        device=str(device), nms_iou=nms_iou, verbose=verbose,
    )
    raw_count = len(candidates)
    if reduce_candidates:
        candidates = detector.keep_top_half_if_over_ten(candidates)

    image_id = record["id"]
    kept, rejected = classifier.classify_candidates(
        classifier_model, [record], {image_id: candidates}, transform,
        idx_to_class, device,
    )
    return raw_count, candidates, kept[image_id], rejected[image_id]
