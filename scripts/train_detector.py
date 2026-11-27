import argparse
from pathlib import Path

import yaml

from pipline import detector

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "output/restart_80_20"
MODALITIES = ("infrared", "visible")
DEFAULT_IMAGE_SIZES = {"infrared": 960, "visible": 640}
DATASET_NAMES = {"infrared": "infrared", "visible": "visible_tiles"}

#new
def main():
    parser = argparse.ArgumentParser(description="训练红外整图和可见光切片 YOLO 检测器")
    parser.add_argument("--modality", choices=("both", *MODALITIES), default="both")
    parser.add_argument("--dataset-dir", type=Path, default=EXPERIMENT_DIR / "by_modality")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR / "detector_runs_highres_tiles")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, help="覆盖输入尺寸，默认红外 960、可见光 640")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.batch) < 1 or (args.imgsz is not None and args.imgsz < 1):
        parser.error("--epochs、--imgsz、--batch 必须大于 0")

    model_path = ROOT / "yolo11n.pt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    modalities = MODALITIES if args.modality == "both" else (args.modality,)
    configs = {}
    for modality in modalities:
        dataset_dir = (args.dataset_dir / DATASET_NAMES[modality]).resolve()
        yaml_path = dataset_dir / "dataset.yaml"
        if modality == "visible" and not yaml_path.is_file():
            parser.error(f"缺少切片数据：{yaml_path}。请先执行 python -m scripts.prepare_visible_tiles")
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        # 数据根目录以当前机器上的 YAML 位置为准。
        config["path"] = str(dataset_dir)
        for split in ("train", "val"):
            split_path = dataset_dir / config[split]
            if not split_path.is_dir():
                raise FileNotFoundError(split_path)
        configs[modality] = config

    for modality, config in configs.items():
        run_dir = args.out_dir.resolve() / modality
        run_dir.mkdir(parents=True, exist_ok=True)
        local_yaml = run_dir / "dataset.yaml"
        local_yaml.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"开始训练 {modality}：{local_yaml}", flush=True)
        best_weights_path = detector.train_detector(
            yaml_path=str(local_yaml),
            model_path=str(model_path),
            out_dir=str(run_dir),
            epochs=args.epochs,
            imgsz=args.imgsz or DEFAULT_IMAGE_SIZES[modality],
            batch=args.batch,
            device=args.device,
        )
        print(f"{modality} 最佳权重：{best_weights_path}", flush=True)


if __name__ == "__main__":
    main()
