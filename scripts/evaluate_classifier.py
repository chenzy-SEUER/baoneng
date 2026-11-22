from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from pipline import classifier

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "output/classifier_runs/run02_merged/last.pt"
CROPS_DIR = ROOT / "output/classifier_crops_merged"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, transform, checkpoint = classifier.build_classifier_with_weights(WEIGHTS_PATH, device=device)

    val_ds = datasets.ImageFolder(CROPS_DIR / "val", transform=transform)
    class_to_idx = checkpoint["class_to_idx"]
    if val_ds.class_to_idx != class_to_idx:
        raise ValueError(
            f"验证集和权重类别映射不一致："
            f"val={val_ds.class_to_idx}, checkpoint={class_to_idx}"
        )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    confusion = classifier.evaluate_classifier_confusion(
        model, val_loader, device, checkpoint["num_classes"]
    )
    assert confusion.sum().item() == len(val_ds), "混淆矩阵总数与验证集数量不一致"

    class_names = sorted(class_to_idx, key=class_to_idx.get)
    print("评估设备：", device)
    print("类别映射：", class_to_idx)
    print("行列类别顺序：", class_names)
    print("混淆矩阵（行=真实类别，列=预测类别）：")
    print(confusion)
    print("验证集样本总数：", confusion.sum().item())
    return confusion


if __name__ == "__main__":
    main()
