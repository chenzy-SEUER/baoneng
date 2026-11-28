"""导出分类正样本和一阶段候选裁剪，供第二阶段训练和负样本审核。"""

import argparse
from collections import Counter, defaultdict
import json
from math import ceil, floor
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS

from pipline.data import export_gt_crops, load_manifest
from pipline.evaluate import box_iou, gt_coverage
from scripts.evaluate_detector_val import LABEL_ALIASES, predict_validation
from scripts.prepare_visible_tiles import TILE_SIZE, offsets


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "output/restart_80_20"
CLASS_ALIASES = {
    **LABEL_ALIASES,
    "GB": "crack",
    "破裂": "破裂",
    "破裂/裂缝": "破裂",
    "遮挡": "遮挡",
    "组件损毁": "组件损坏",
    "热板": "hotspot",
}
SPLITS = ("train", "val")


def load_sources(dataset):
    records = load_manifest(dataset / "manifest.json")
    if not records or len({record["id"] for record in records}) != len(records):
        raise ValueError(f"清单为空或包含重复 ID：{dataset}")
    images = {}
    for split in SPLITS:
        for path in (dataset / "images" / split).iterdir():
            if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS:
                key = (split, path.stem)
                if key in images:
                    raise ValueError(f"重复图片 ID：{path}")
                images[key] = path
    if images.keys() != {(r["split"], r["id"]) for r in records}:
        raise ValueError(f"图片与清单不一致：{dataset}")

    groups = defaultdict(list)
    for record in records:
        record["image_path"] = str(images[record["split"], record["id"]])
        for obj in record["objects"]:
            label = obj["label"]
            obj["label"] = CLASS_ALIASES.get(label, label)
        groups[record["split"], record["dataset"]].append(record)
    return records, groups


def candidate_inputs(records, folder, tiled):
    folder.mkdir()
    tile_records = {}
    for image_id, record in records.items():
        source = Path(record["image_path"])
        if not tiled:
            (folder / source.name).symlink_to(source)
            continue
        with Image.open(source) as image:
            rgb = image.convert("RGB")
        for y in offsets(record["height"]):
            for x in offsets(record["width"]):
                tile = f"{image_id}_x{x}_y{y}"
                rgb.crop((x, y, x + TILE_SIZE, y + TILE_SIZE)).save(folder / f"{tile}.jpg", quality=95)
                tile_records[tile] = {"source_image_id": image_id, "offset": (x, y)}
    return (tile_records, folder) if tiled else None


def export_candidates(model, selected, names, modality, args, out_dir):
    prediction_dir = out_dir.parent / f"{modality}_inference"
    tile_data = candidate_inputs(selected, prediction_dir, modality == "visible")
    pending_review = []
    positives = Counter()
    for image_id, boxes, scores in predict_validation(model, selected, prediction_dir, args, tile_data):
        record = selected[image_id]
        objects = record["objects"]
        split = record["split"]
        already_matched = set()
        review_count = 0
        source = Path(record["image_path"])
        with Image.open(source) as image:
            rgb = image.convert("RGB")
        for index, (score, box) in enumerate(sorted(zip(scores, boxes), reverse=True)):
            x1, y1, x2, y2 = box
            crop = [max(0, floor(x1)), max(0, floor(y1)), min(rgb.width, ceil(x2)), min(rgb.height, ceil(y2))]
            if crop[0] >= crop[2] or crop[1] >= crop[3]:
                continue
            overlap, gt_index = max(((box_iou(obj["box"], crop), i) for i, obj in enumerate(objects)), default=(0, -1))
            filename = f"{image_id}_pred_{index}.jpg"
            if overlap >= 0.5 and gt_index not in already_matched and gt_coverage(objects[gt_index]["box"], crop) >= 0.8:
                label = objects[gt_index]["label"]
                target = Path("candidate_positives") / modality / split / label / filename
                already_matched.add(gt_index)
                positives[split, label] += 1
            elif overlap == 0 and review_count < args.review_per_image:
                target = Path("review") / modality / split / filename
                pending_review.append({
                    "crop": target.as_posix(), "modality": modality, "split": split, "dataset": record["dataset"],
                    "source_image_id": image_id, "source_image": names[image_id], "xyxy": crop,
                    "score": round(score, 6), "review_status": "pending",
                })
                review_count += 1
            else:
                continue
            (out_dir / target).parent.mkdir(parents=True, exist_ok=True)
            rgb.crop(tuple(crop)).save(out_dir / target, quality=95)
    return pending_review, positives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=EXPERIMENT_DIR / "ir_no_data08_data09")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR / "classifier_preparation")
    parser.add_argument("--infrared-weights", type=Path, default=EXPERIMENT_DIR /
                        "detector_runs_ir_no_data08_data09/infrared/train/weights/best.pt")
    parser.add_argument("--visible-weights", type=Path, default=EXPERIMENT_DIR /
                        "detector_runs_highres_tiles/visible/train/weights/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--images-per-source", type=int, default=12, help="每个来源在 train 和 val 各抽多少张图")
    parser.add_argument("--review-per-image", type=int, default=4)
    args = parser.parse_args()
    if min(args.threads, args.batch, args.images_per_source, args.review_per_image) < 1:
        parser.error("threads、batch 和抽样数量必须大于 0")
    torch.set_num_threads(args.threads)
    args.conf = [0.10]
    args.nms_iou = 0.70

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    source = args.dataset_dir.resolve()
    profiles = (("infrared", 960, args.infrared_weights), ("visible", 640, args.visible_weights))
    for _, _, weights in profiles:
        if not weights.is_file():
            raise FileNotFoundError(weights)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    report = {"dataset_dir": str(source), "aliases": CLASS_ALIASES, "modalities": {}}
    review_candidates = []
    rng = Random(42)
    with TemporaryDirectory(prefix=".classifier_preparation_", dir=out_dir.parent) as tmp:
        staged = Path(tmp) / "export"
        for modality, imgsz, weights in profiles:
            args.imgsz = imgsz
            records, groups = load_sources(source / modality)
            gt_counts = {split: Counter(o["label"] for r in records if r["split"] == split for o in r["objects"])
                         for split in SPLITS}
            written = export_gt_crops(records, staged / "gt_positives" / modality)
            if written != sum(len(r["objects"]) for r in records):
                raise ValueError(f"部分真实框未能裁剪：{modality}")
            selected = {}
            for group in sorted(groups):
                items = sorted(groups[group], key=lambda record: record["id"])
                selected.update((r["id"], r) for r in rng.sample(items, min(len(items), args.images_per_source)))
            print(f"{modality}：真值裁剪 {written}，候选抽样 {len(selected)} 张原图", flush=True)
            names = {k: Path(r["image_path"]).relative_to(source).as_posix() for k, r in selected.items()}
            reviews, positives = export_candidates(YOLO(str(weights)), selected, names, modality, args, staged)
            review_candidates.extend(reviews)
            report["modalities"][modality] = {
                "weights": str(weights.resolve()), "imgsz": args.imgsz, "gt_positives": gt_counts,
                "sampled_images": {split: sum(r["split"] == split for r in selected.values()) for split in SPLITS},
                "candidate_positives": {split: {label: n for (part, label), n in positives.items() if part == split}
                                        for split in SPLITS},
                "pending_review": Counter(r["split"] for r in reviews),
            }
        for name, contents in (("summary.json", report), ("review_candidates.json", review_candidates)):
            (staged / name).write_text(json.dumps(contents, ensure_ascii=False, indent=2), encoding="utf-8")
        if out_dir.exists():
            raise FileExistsError(out_dir)
        staged.rename(out_dir)
    print(f"导出到 {out_dir}，共有 {len(review_candidates)} 个候选待审核")


if __name__ == "__main__":
    main()
