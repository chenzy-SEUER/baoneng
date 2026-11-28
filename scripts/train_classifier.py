import argparse
from pathlib import Path

import torch

from pipline import classifier

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "output/restart_80_20"
MODALITIES = ("infrared", "visible")


def main():
    parser = argparse.ArgumentParser(description="分别训练红外和可见光二阶段分类器")
    parser.add_argument("--modality", choices=("both", *MODALITIES), default="both")
    parser.add_argument("--dataset-dir", type=Path, default=EXPERIMENT_DIR / "classifier_dataset")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR / "classifier_runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1:
        parser.error("--epochs 和 --batch-size 必须大于 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("当前 PyTorch 无法使用 CUDA，请检查 GPU 环境或指定 --device cpu")

    modalities = MODALITIES if args.modality == "both" else (args.modality,)
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    for modality in modalities:
        for split in ("train", "val"):
            path = dataset_dir / modality / split
            if not path.is_dir():
                raise FileNotFoundError(path)
        run_dir = out_dir / modality
        if run_dir.exists() or run_dir.is_symlink():
            raise FileExistsError(f"训练输出已存在：{run_dir}，请使用 --out-dir 指定新目录")

    for modality in modalities:
        run_dir = out_dir / modality
        print(f"开始训练 {modality}", flush=True)
        classifier.train_classifier(
            dataset_dir / modality, run_dir, epochs=args.epochs, batch_size=args.batch_size,
            imgsz=224, learning_rate=1e-4, device=device,
        )
        print(f"{modality} 分类权重保存到 {run_dir / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
