from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50

from pipline import classifier

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "output/classifier_runs/run01/last.pt"
CROPS_DIR = ROOT / "output/classifier_crops"


def main():
    # 这里读取的是保存的字典，还不是可直接推理的模型。
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]
    if checkpoint["architecture"] != "resnet50":
        raise ValueError("权重的模型结构不是 resnet50")
    if num_classes != len(class_to_idx) or sorted(class_to_idx.values()) != list(range(num_classes)):
        raise ValueError("权重中的类别数量或编号不一致")

    # 沿用训练保存的预处理参数，仅加载验证集。
    transform = transforms.Compose([
        transforms.Resize((checkpoint["imgsz"], checkpoint["imgsz"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
    ])
    val_ds = datasets.ImageFolder(CROPS_DIR / "val", transform=transform)
    if val_ds.class_to_idx != class_to_idx:
        raise ValueError(
            f"验证集和权重类别映射不一致："
            f"val={val_ds.class_to_idx}, checkpoint={class_to_idx}"
        )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    # 构建相同结构并载入训练参数，不再下载预训练权重。
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    confusion = classifier.evaluate_classifier_confusion(
        model, val_loader, device, num_classes
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
