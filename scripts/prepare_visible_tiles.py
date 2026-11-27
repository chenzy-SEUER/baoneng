"""沿用可见光原图的 train/val 划分，导出重叠切片及 YOLO 标注。"""

import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from PIL import Image


DATASET_DIR = Path(__file__).resolve().parents[1] / "output/restart_80_20/by_modality"
TILE_SIZE = 1024
STRIDE = 768


def offsets(length):
    if length < TILE_SIZE:
        raise ValueError(f"原图边长 {length} 小于切片尺寸 {TILE_SIZE}")
    last = length - TILE_SIZE
    return sorted({*range(0, last + 1, STRIDE), last})


def plan_tiles(records):
    tiles = []
    fully_seen = set()
    for record in records:
        image_id, split = record["id"], record["split"]
        for y in offsets(record["height"]):
            for x in offsets(record["width"]):
                objects = []
                for index, obj in enumerate(record["objects"]):
                    x1, y1, x2, y2 = obj["box"]
                    box = [max(0, x1 - x), max(0, y1 - y), min(TILE_SIZE, x2 - x), min(TILE_SIZE, y2 - y)]
                    if box[0] >= box[2] or box[1] >= box[3]:
                        continue
                    objects.append({"label": obj["label"], "box": box, "source_object_index": index})
                    if x <= x1 and y <= y1 and x2 <= x + TILE_SIZE and y2 <= y + TILE_SIZE:
                        fully_seen.add((image_id, index))
                tile_id = f"{image_id}_x{x}_y{y}"
                tiles.append({
                    "id": tile_id, "dataset": record["dataset"], "split": split,
                    "image_path": f"images/{split}/{tile_id}.jpg", "width": TILE_SIZE, "height": TILE_SIZE,
                    "source_image_id": image_id, "offset": [x, y], "objects": objects,
                })
    all_objects = {(r["id"], i) for r in records for i in range(len(r["objects"]))}
    without_full_tile = sorted(all_objects - fully_seen)
    if without_full_tile:
        print(f"有 {len(without_full_tile)} 个框只出现在跨边界的切片中", flush=True)

    # 训练集抽取背景块，验证集保留完整网格，以便回到整图评估。
    background = [tile["id"] for tile in tiles if tile["split"] == "train" and not tile["objects"]]
    train_positive = sum(tile["split"] == "train" and bool(tile["objects"]) for tile in tiles)
    keep_background = set(Random(42).sample(background, min(train_positive, len(background))))
    return [tile for tile in tiles if tile["split"] == "val" or tile["objects"] or tile["id"] in keep_background]


def export_tiles(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    records = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    splits = ("train", "val")
    records = [r for r in records if r["split"] in splits]
    if {r["split"] for r in records} != set(splits) or len({r["id"] for r in records}) != len(records):
        raise ValueError("原图 manifest 必须包含 train 和 val，且图片 ID 唯一")
    images = {}
    for split in splits:
        for path in (source / "images" / split).iterdir():
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                key = (split, path.stem)
                if key in images:
                    raise ValueError(f"原图 ID 重复：{path}")
                images[key] = path
    if set(images) != {(r["split"], r["id"]) for r in records}:
        raise ValueError("原图图片与 manifest 不一致")
    tiles = plan_tiles(records)
    counts = Counter((tile["split"], bool(tile["objects"])) for tile in tiles)
    if not all(counts[split, True] for split in splits):
        raise ValueError("train 和 val 都需要包含有标注的切片")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".visible_tiles_", dir=destination.parent) as temporary_dir:
        staged = Path(temporary_dir) / "visible_tiles"
        for split in splits:
            (staged / "images" / split).mkdir(parents=True)
            (staged / "labels" / split).mkdir(parents=True)
        current_image = None
        rgb = None
        for tile in tiles:
            split, source_id = tile["split"], tile["source_image_id"]
            path = images[split, source_id]
            if path != current_image:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                current_image = path
            x, y = tile["offset"]
            if x + TILE_SIZE > rgb.width or y + TILE_SIZE > rgb.height:
                raise ValueError(f"切片超出图像范围：{path}")
            rgb.crop((x, y, x + TILE_SIZE, y + TILE_SIZE)).save(staged / tile["image_path"], quality=95)
            lines = []
            for obj in tile["objects"]:
                x1, y1, x2, y2 = obj["box"]
                lines.append(f"0 {(x1+x2)/(2*TILE_SIZE)} {(y1+y2)/(2*TILE_SIZE)} "
                             f"{(x2-x1)/TILE_SIZE} {(y2-y1)/TILE_SIZE}")
            (staged / "labels" / split / f"{tile['id']}.txt").write_text("\n".join(lines), encoding="utf-8")
        (staged / "dataset.yaml").write_text(
            "train: images/train\nval: images/val\nnames:\n  0: defect\n", encoding="utf-8",
        )
        (staged / "manifest.json").write_text(json.dumps(tiles, ensure_ascii=False, indent=2), encoding="utf-8")
        if destination.exists():
            raise FileExistsError(destination)
        staged.rename(destination)
    print(f"切片数据：{destination}")
    for split in splits:
        print(f"{split} 正样本 {counts[split, True]}，背景 {counts[split, False]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATASET_DIR / "visible")
    parser.add_argument("--out-dir", type=Path, default=DATASET_DIR / "visible_tiles")
    args = parser.parse_args()
    export_tiles(args.source.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
