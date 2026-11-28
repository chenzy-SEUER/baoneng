"""从现有 YOLO 数据导出独立副本，排除 Data08、Data09 的红外图。"""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "output/restart_80_20"
EXCLUDED_INFRARED = {"Data08", "Data09"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")


def export_modality(source, destination):
    records = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source.name == "infrared":
        records = [record for record in records if record["dataset"] not in EXCLUDED_INFRARED]
    if len({record["id"] for record in records}) != len(records):
        raise ValueError(f"图片 ID 重复：{source}")
    if {record["split"] for record in records} != set(SPLITS):
        raise ValueError(f"数据必须包含 train 和 val：{source}")

    split_counts = {}
    for split in SPLITS:
        expected_ids = {record["id"] for record in records if record["split"] == split}
        image_dir = destination / "images" / split
        label_dir = destination / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        copied_ids = set()
        for image in (source / "images" / split).iterdir():
            image_id = image.stem
            if image.suffix.lower() not in IMAGE_SUFFIXES or image_id not in expected_ids:
                continue
            if image_id in copied_ids:
                raise ValueError(f"多个图片文件对应同一 ID：{image}")
            label = source / "labels" / split / f"{image_id}.txt"
            shutil.copyfile(image, image_dir / image.name)
            shutil.copyfile(label, label_dir / label.name)
            copied_ids.add(image_id)
        if copied_ids != expected_ids:
            raise ValueError(f"找不到清单中的图片：{source} {split} {sorted(expected_ids - copied_ids)}")
        split_counts[split] = len(copied_ids)

    config = yaml.safe_load((source / "dataset.yaml").read_text(encoding="utf-8"))
    config.pop("path", None)
    (destination / "dataset.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8",
    )
    (destination / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return split_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=EXPERIMENT_DIR / "by_modality")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR / "ir_no_data08_data09")
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.out_dir.parent.resolve() / args.out_dir.name
    if destination.exists() or destination.is_symlink():
        parser.error(f"输出已存在：{destination}，请用 --out-dir 指定新目录")
    modalities = ("infrared", "visible", "visible_tiles")
    for modality in modalities:
        if not (source / modality / "manifest.json").is_file():
            parser.error(f"缺少原有的 YOLO 数据：{source / modality}，请检查 --source")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ir_no_data08_data09_", dir=destination.parent) as tmp:
        staged = Path(tmp) / destination.name
        counts = {modality: export_modality(source / modality, staged / modality) for modality in modalities}
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        staged.rename(destination)

    print(f"数据集：{destination}")
    for modality, split_counts in counts.items():
        print(f"{modality}：训练 {split_counts['train']}，验证 {split_counts['val']}")


if __name__ == "__main__":
    main()
