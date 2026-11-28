"""从已标注原图裁剪背景样本，生成独立的 non_defect 候选数据。"""

import argparse
from collections import Counter, defaultdict
import json
from math import ceil
from pathlib import Path
from random import Random
from statistics import median
from tempfile import TemporaryDirectory

from PIL import Image

from pipline.evaluate import box_iou
from scripts.prepare_classifier_candidates import EXPERIMENT_DIR, load_sources


def sample_background(record, fallback_size, count, rng):
    width, height = record["width"], record["height"]
    margin = max(8, ceil(max(width, height) * 0.015))
    objects = [obj["box"] for obj in record["objects"]]
    blocked = [(x1 - margin, y1 - margin, x2 + margin, y2 + margin) for x1, y1, x2, y2 in objects]
    boxes = []
    for _ in range(100 * count):
        if objects:
            x1, y1, x2, y2 = rng.choice(objects)
            anchor_x, anchor_y = (x1 + x2) / 2, (y1 + y2) / 2
            base_width, base_height = x2 - x1, y2 - y1
        else:
            anchor_x, anchor_y = width / 2, height / 2
            base_width, base_height = fallback_size
        scale = rng.uniform(0.8, 1.25)
        crop_width = min(width // 3, max(8, round(base_width * scale)))
        crop_height = min(height // 3, max(8, round(base_height * scale)))
        search_x = max(5 * crop_width, 6 * margin)
        search_y = max(5 * crop_height, 6 * margin)
        x = round(anchor_x + rng.uniform(-search_x, search_x) - crop_width / 2)
        y = round(anchor_y + rng.uniform(-search_y, search_y) - crop_height / 2)
        box = (x, y, x + crop_width, y + crop_height)
        if not (0 <= x < box[2] <= width and 0 <= y < box[3] <= height):
            continue
        if any(box_iou(box, other) > 0 for other in blocked + boxes):
            continue
        boxes.append(box)
        if len(boxes) == count:
            break
    return boxes, margin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=EXPERIMENT_DIR / "ir_no_data08_data09")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR / "classifier_background")
    parser.add_argument("--per-image", type=int, default=2, help="每张原图最多裁剪的背景块数")
    args = parser.parse_args()
    if args.per_image < 1:
        parser.error("--per-image 必须大于 0")
    source = args.dataset_dir.resolve()
    destination = args.out_dir.resolve()
    if destination.exists():
        raise FileExistsError(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    counts = Counter()
    rng = Random(42)
    with TemporaryDirectory(prefix=".classifier_background_", dir=destination.parent) as tmp:
        staged = Path(tmp) / "export"
        for modality in ("infrared", "visible"):
            records, _ = load_sources(source / modality)
            dimensions = defaultdict(list)
            for record in records:
                if record["split"] == "train":
                    for obj in record["objects"]:
                        x1, y1, x2, y2 = obj["box"]
                        dimensions[record["dataset"]].append((x2 - x1, y2 - y1))
            fallback = {dataset: tuple(map(median, zip(*sizes))) for dataset, sizes in dimensions.items()}
            for record in sorted(records, key=lambda item: item["id"]):
                dataset, split, image_id = record["dataset"], record["split"], record["id"]
                boxes, margin = sample_background(record, fallback.get(dataset, (32, 32)), args.per_image, rng)
                if not boxes:
                    continue
                image_path = Path(record["image_path"])
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                if rgb.size != (record["width"], record["height"]):
                    raise ValueError(f"图片尺寸与清单不一致：{image_path}")
                for index, box in enumerate(boxes):
                    crop = Path(modality) / split / "non_defect" / f"{image_id}_bg_{index}.jpg"
                    (staged / crop).parent.mkdir(parents=True, exist_ok=True)
                    rgb.crop(box).save(staged / crop, quality=95)
                    samples.append({
                        "crop": crop.as_posix(), "modality": modality, "split": split, "dataset": dataset,
                        "source_image_id": image_id, "source_image": image_path.relative_to(source).as_posix(),
                        "xyxy": box, "gt_margin_px": margin, "review_status": "sampled_from_annotations",
                    })
                    counts[modality, split] += 1
            print(f"{modality}：训练 {counts[modality, 'train']}，验证 {counts[modality, 'val']}", flush=True)
        (staged / "manifest.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        if destination.exists():
            raise FileExistsError(destination)
        staged.rename(destination)
    print(f"已裁剪 {len(samples)} 张背景样本：{destination}")


if __name__ == "__main__":
    main()
