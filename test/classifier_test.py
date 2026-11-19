from pathlib import Path
import torch
from torch import nn, optim

from pipline import classifier

ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = ROOT / "output/classifier_crops"


if __name__ == "__main__":
    train_loader, val_loader, class_to_idx = classifier.build_classifier_loaders(
        CROPS_DIR, 32, 224
    )
    images, labels = next(iter(train_loader))

    print("图片形状：", images.shape)
    print("标签：", labels)
    print("标签形状：", labels.shape)
    print("类别映射：", class_to_idx)

    s = classifier.count_classifier_samples(train_loader, class_to_idx)
    print("train 数据集",s)
    print("train 数据集总数：", sum(s.values()))
    s = classifier.count_classifier_samples(val_loader, class_to_idx)
    print("val 数据集", s)
    print("val 数据集总数：", sum(s.values()))

    device = torch.device("cuda:0")
    model = classifier.build_classifier(len(class_to_idx))
    model.to(device)
    # model.eval()
    # with torch.no_grad():
    #     outputs = model(images)
    #     assert outputs.shape == (images.shape[0], len(class_to_idx))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics = classifier.train_one_epoch(model, [(images, labels)], criterion, optimizer, device)

    print("单批次训练结果：", metrics)

    val_images, val_labels = next(iter(val_loader))
    val_metrics = classifier.evaluate_classifier(
        model, [(val_images, val_labels)], criterion, device
    )
    print("单批次验证结果：", val_metrics)

