"""准备 Labelme 分割试验包；审核轮廓后导出配套分割/检测数据。"""
import argparse
import json
import math
import random
import shutil
import hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(manifest, destination, train_count):
    records = read_json(manifest)
    eligible = [r for r in records if any(o["label"].lower() == "zhedang" for o in r["objects"])]
    train = sorted((r for r in eligible if r["split"] == "train"), key=lambda r: r["id"])
    if not 0 < train_count <= len(train):
        raise ValueError(f"train-count 必须在 1 到 {len(train)} 之间")
    train = random.Random(42).sample(train, train_count)
    selected = train + sorted((r for r in eligible if r["split"] == "val"), key=lambda r: r["id"])
    ids = [r["id"] for r in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest 存在重复 ID")
    # 先确认所有原图可用，避免生成不完整的标注包。
    for r in selected:
        image = Path(r["image_path"])
        if not (image if image.is_absolute() else ROOT / image).is_file():
            raise FileNotFoundError(image)
    destination.mkdir(parents=True, exist_ok=False)
    for r in selected:
        folder = destination / "annotations" / r["split"]
        folder.mkdir(parents=True, exist_ok=True)
        image = Path(r["image_path"])
        image = image if image.is_absolute() else ROOT / image
        filename = r["id"] + image.suffix
        shutil.copy2(image, folder / filename)
        shapes = []
        for i, obj in enumerate(r["objects"]):
            x1, y1, x2, y2 = obj["box"]
            shapes.append({"label": f"reference_{i}_{obj['label']}", "points": [[x1, y1], [x2, y2]],
                           "group_id": None, "shape_type": "rectangle", "flags": {}})
        write_json(folder / (r["id"] + ".json"), {
            "version": "5.0.0", "flags": {"reviewed": False}, "shapes": shapes,
            "imagePath": filename, "imageData": None, "imageHeight": r["height"], "imageWidth": r["width"],
        })
    write_json(destination / "manifest.json", selected)
    print(f"已准备 {len(train)} 张 train、{len(selected)-len(train)} 张 val；未使用 test。")


def validate_polygons(annotation, record, allow_drafts=False):
    """拒绝未审核、漏标、重复实例、越界与退化轮廓。参考矩形永不转为掩膜。"""
    if annotation.get("flags", {}).get("reviewed") is not True and not allow_drafts:
        raise ValueError(f"{record['id']}: 尚未审核；仅实验伪标签可显式使用 --allow-drafts")
    if (annotation.get("imageWidth"), annotation.get("imageHeight")) != (record["width"], record["height"]):
        raise ValueError(f"{record['id']}: 图片尺寸不一致")
    polygons = {}
    for shape in annotation["shapes"]:
        if shape.get("shape_type") == "rectangle" and shape["label"].startswith("reference_"):
            continue
        if shape.get("shape_type") != "polygon" or not shape["label"].startswith("obj_"):
            raise ValueError(f"{record['id']}: 只接受 obj_编号 的 polygon，参考矩形除外")
        try:
            index = int(shape["label"][4:])
        except ValueError as exc:
            raise ValueError("轮廓标签必须为 obj_0、obj_1 等") from exc
        points = shape["points"]
        if index in polygons or index not in range(len(record["objects"])):
            raise ValueError(f"{record['id']}: 重复或未知实例 {index}")
        if len(points) < 3 or any(len(p) != 2 for p in points):
            raise ValueError("多边形至少需要三个二维顶点")
        if any(not math.isfinite(v) for p in points for v in p):
            raise ValueError("轮廓含非有限坐标")
        if any(not (0 <= x <= record["width"] and 0 <= y <= record["height"]) for x, y in points):
            raise ValueError("轮廓坐标越界")
        area = abs(sum(points[i][0] * points[(i+1) % len(points)][1] -
                       points[(i+1) % len(points)][0] * points[i][1] for i in range(len(points)))) / 2
        if area <= 0:
            raise ValueError("多边形面积必须为正")
        polygons[index] = points
    if set(polygons) != set(range(len(record["objects"]))):
        raise ValueError(f"{record['id']}: 必须为图中全部已标注缺陷补充轮廓，不能只标 zhedang")
    return polygons


def export(package, destination, allow_drafts=False):
    records = read_json(package / "manifest.json")
    validated = []
    provenance = []
    seen_ids = set()
    for r in records:
        if r["split"] not in {"train", "val"} or r["id"] in seen_ids:
            raise ValueError("试验包只允许 train/val，且 ID 必须唯一")
        seen_ids.add(r["id"])
        annotation_path = package / "annotations" / r["split"] / (r["id"] + ".json")
        annotation = read_json(annotation_path)
        if Path(annotation["imagePath"]).name != annotation["imagePath"]:
            raise ValueError("imagePath 必须是同目录下的文件名")
        source = package / "annotations" / r["split"] / annotation["imagePath"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stem != r["id"]:
            raise ValueError("图片文件名必须与实例记录 ID 一致")
        validated.append((r, source, validate_polygons(annotation, r, allow_drafts=allow_drafts)))
        provenance.append({
            "id": r["id"], "split": r["split"],
            "annotation": str(annotation_path.relative_to(package)),
            "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
            "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "reviewed": annotation.get("flags", {}).get("reviewed") is True,
            "annotation_flags": annotation.get("flags", {}),
            "polygons": [s for s in annotation["shapes"] if s.get("shape_type") == "polygon"],
        })
    if not records:
        raise ValueError("试验包没有图片")
    # 即使允许草稿，几何及完整性检查仍须全部通过后才建立输出。
    destination.mkdir(parents=True, exist_ok=False)
    for task in ("segment", "detect"):
        for r, source, polygons in validated:
            folder = destination / task
            images = folder / "images" / r["split"]
            labels = folder / "labels" / r["split"]
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, images / source.name)
            lines = []
            for i in range(len(r["objects"])):
                points = polygons[i]
                if task == "segment":
                    values = [v for x, y in points for v in (x / r["width"], y / r["height"])]
                else:
                    # 使用相同审核轮廓的外接框，避免比较两套不同标注。
                    xs, ys = zip(*points)
                    values = [(min(xs)+max(xs))/2/r["width"], (min(ys)+max(ys))/2/r["height"],
                              (max(xs)-min(xs))/r["width"], (max(ys)-min(ys))/r["height"]]
                lines.append("0 " + " ".join(f"{v:.8f}" for v in values))
            (labels / (r["id"] + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        # 不包含机器绝对路径；训练时传入此 YAML 的绝对路径。
        (destination / task / "dataset.yaml").write_text(
            "train: images/train\nval: images/val\nnames:\n  0: defect\n", encoding="utf-8")
    write_json(destination / "experiment.json", {
        "schema_version": 1, "allow_drafts": allow_drafts,
        "label_status": "experimental_pseudo_labels" if any(not p["reviewed"] for p in provenance) else "reviewed",
        "images_by_split": dict(Counter(r["split"] for r in records)),
        "instances_by_split": dict(Counter(r["split"] for r in records for _ in r["objects"])),
        "original_class_counts": dict(Counter(o["label"] for r in records for o in r["objects"])),
        "detection_labels": "Bounding boxes of the same polygons used by segment; all instances use class 0 (defect)",
        "evaluation_reference": "reference_manifest.json retains original GT boxes/classes; do not treat draft mask AP as authoritative",
        "annotations": provenance,
    })
    write_json(destination / "reference_manifest.json", records)
    if (package / "review_index.json").is_file():
        shutil.copy2(package / "review_index.json", destination / "source_review_index.json")
    print(f"已导出相同图片和实例的分割/检测数据：{destination}")
    if any(not p["reviewed"] for p in provenance):
        print("标签状态：实验伪标签；原标注的 reviewed 状态保持不变。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, default=ROOT / "output/dataset/manifest.json")
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--train-count", type=int, default=20)
    exp = sub.add_parser("export")
    exp.add_argument("--package", type=Path, required=True)
    exp.add_argument("--out-dir", type=Path, required=True)
    exp.add_argument("--allow-drafts", action="store_true", help="允许未审核轮廓作为实验伪标签；仍检查几何和完整性")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.out_dir, args.train_count)
    else:
        export(args.package, args.out_dir, allow_drafts=args.allow_drafts)


if __name__ == "__main__":
    main()
