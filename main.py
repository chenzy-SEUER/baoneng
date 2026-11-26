import argparse
import logging
import sys
from pathlib import Path

import pipline.data as data
import pipline.export_yolo as export_yolo
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="五个 VOC 数据集按来源做训练／验证 8:2 划分")
    parser.add_argument("datasets", nargs=3, type=Path, metavar="DATASET", help="三个新增 VOC 数据集目录")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/five_datasets_80_20")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", help="Ultralytics 设备，例如 0、mps 或 cpu")
    parser.add_argument("--prepare-only", action="store_true", help="仅导出 YOLO 数据和 manifest")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必须大于 0")

    dataset_dirs = [ROOT / "data/Data10", ROOT / "data/Data11", *args.datasets]
    dataset_dirs = [path.resolve() for path in dataset_dirs]
    if len({path.name for path in dataset_dirs}) != len(dataset_dirs):
        parser.error("数据集目录名必须唯一，目录名用于生成图片 ID")
    for path in dataset_dirs:
        for subdir in ("images", "annotations"):
            if not (path / subdir).is_dir():
                parser.error(f"数据集缺少目录：{path / subdir}")

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        parser.error(f"实验输出目录必须为空：{out_dir}")
    model_path = ROOT / "yolo11n.pt"
    if not args.prepare_only and not model_path.is_file():
        parser.error(f"缺少模型权重：{model_path}")

    all_records = []
    for path in dataset_dirs:
        records, summary = data.load_dataset(path)
        logging.info("%s 已读取 %d/%d 张图片", path, len(records), summary["image_count"])
        for key in ("missing_annotations", "orphan_annotations", "errors"):
            if summary[key]:
                logging.warning("%s %s = %s", path, key, summary[key])
        if len(records) < 2:
            parser.error(f"至少需要两条有效记录才能划分训练和验证：{path}")
        records.sort(key=lambda record: record["id"])
        train_records, val_records, _ = data.split_records(records, train_ratio=0.8, val_ratio=0.2)
        all_records.extend(train_records)
        all_records.extend(val_records)
    if len({record["id"] for record in all_records}) != len(all_records):
        parser.error("合并后的数据集存在重复图片 ID")

    dataset_dir = out_dir / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir()
    export_yolo.export_detector_dataset(all_records, dataset_dir, include_test=False)
    data.save_manifest(all_records, dataset_dir)
    logging.info("已导出数据集：%s", dataset_dir)
    if args.prepare_only:
        return

    import torch
    from pipline import detector

    best_weights_path = detector.train_detector(
        yaml_path=str(dataset_dir / "dataset.yaml"),
        model_path=str(model_path),
        out_dir=str(out_dir / "yolo_detect"),
        epochs=args.epochs,
        device=args.device or ("0" if torch.cuda.is_available() else "cpu"),
    )
    logging.info("最佳权重：%s", best_weights_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    main()
