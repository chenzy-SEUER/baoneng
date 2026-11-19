import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import ResNet50_Weights, resnet50


def build_classifier_loaders(crops_dir, batch_size=32, imgsz=224):
    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.ImageFolder(root=os.path.join(crops_dir,'train'), transform=transform)
    val_ds = datasets.ImageFolder(root=os.path.join(crops_dir, 'val'), transform=transform)

    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError(
            f"训练集和验证集类别映射不一致："
            f"train={train_ds.class_to_idx}, val={val_ds.class_to_idx}"
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_ds.class_to_idx


def count_classifier_samples(loader, class_to_idx):
    counts = Counter(loader.dataset.targets)
    result = dict()
    for cls, idx in class_to_idx.items():
        result[cls] = counts[idx] if counts[idx] else 0
    return result


def export_negative_classifier_crops(review_dir, crops_dir):
    manifest_path = Path(review_dir)/"review_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for split in ('train', 'val'):
        target_dir = Path(crops_dir) / split / 'non_defect'
        target_dir.mkdir(parents=True, exist_ok=True)
        has_file = any(item.is_file() for item in target_dir.iterdir())
        if has_file:
            raise FileExistsError("文件夹已存在")

    train_copy_cnt = 0
    val_copy_cnt = 0
    for review in manifest:
        split = review["split"]
        if split not in ("train", "val"):
            continue
        target_dir = Path(crops_dir) / split / "non_defect"
        objects = review["objects"]
        for obj in objects:
            crop_path = obj['crop']
            shutil.copy2(Path(review_dir)/crop_path, target_dir/f"{Path(crop_path).name}")
            if split == "train":
                train_copy_cnt += 1
            if split == "val":
                val_copy_cnt += 1

    return { "train": train_copy_cnt, "val": val_copy_cnt }


def build_classifier(num_classes):
    """构建预训练 ResNet-50，输出 num_classes 个未归一化的分类分数。"""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # 保留预训练的特征提取部分，重新创建适合当前类别数量的分类层。
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model



'''
要求是：
1. 创建带预训练权重的 ResNet-50。
2. 读取原有 model.fc.in_features。
3. 将 model.fc 替换为输出 num_classes 个分数的线性层。
4. 返回模型，末尾不加 Softmax，后面训练直接使用交叉熵损失。
验证时，用类别映射的长度作为 num_classes，取一个 batch，设置 model.eval()，在 torch.no_grad() 中前向计算。预期：
输入：[32, 3, 224, 224]
输出：[32, 8]
使用预训练权重时，还需要给现有图片变换补上对应的归一化。这一步先完成模型构建和前向验证，不开始训练。
'''
















