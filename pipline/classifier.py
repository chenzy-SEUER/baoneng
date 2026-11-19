import json
import os
import shutil
from collections import Counter, defaultdict
from math import floor, ceil
from pathlib import Path

import torch
from PIL import Image
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

def build_classifier_with_weights(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    class_to_idx = checkpoint['class_to_idx']
    num_classes = checkpoint['num_classes']
    if checkpoint['architecture'] != 'resnet50':
        raise ValueError("权重的模型结构不是 resnet50")
    if num_classes != len(class_to_idx) or sorted(class_to_idx.values()) != list(range(num_classes)):
        raise ValueError("权重中的类别数量或编号不一致")

    transform = transforms.Compose([
        transforms.Resize((checkpoint['imgsz'], checkpoint['imgsz'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=checkpoint['mean'], std=checkpoint['std']),
    ])

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    idx_to_class = {
        idx: name for name, idx in checkpoint["class_to_idx"].items()
    }
    return model, transform, idx_to_class


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss,correct_count, total_samples= 0, 0, 0
    for batch in loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*images.size(0)
        pred = outputs.argmax(dim=1)
        correct_count += (pred==labels).sum().item()
        total_samples += images.size(0)

    return {"loss": total_loss/total_samples,"accuracy":correct_count/total_samples}




def evaluate_classifier(model, loader, criterion, device):
    model.eval()
    total_loss, correct_count, total_samples= 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            total_loss += loss.item()*images.size(0)
            predicted_labels = preds.argmax(dim=1)
            correct_count += (predicted_labels == labels).sum().item()
            total_samples += images.size(0)
    return {
        "loss": total_loss / total_samples,
        "accuracy": correct_count / total_samples,
    }


'''混淆矩阵统计函数'''
def evaluate_classifier_confusion(model, loader, device, num_classes):
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for images, true_labels in loader:
            images = images.to(device)
            preds = model(images)
            predicted_labels = preds.argmax(dim=1).cpu()
            for true_label, predicted_label in zip(true_labels, predicted_labels):
                confusion[true_label.item(), predicted_label.item()] += 1
    return confusion



def classify_candidates(
    model, records, predictions, transform, idx_to_class, device
):
    """分类原图上的候选区域，返回保留的缺陷和剔除的非缺陷。

    model 应已移动到 device；idx_to_class 使用整数类别编号作为键。
    返回的 box 是实际裁剪区域的原图整数 xyxy 坐标。
    """
    kept, rejected = {}, {}
    model.eval()
    with torch.no_grad():
        for record in records:
            image_path = record["image_path"]
            image_id = record['id']
            kept[image_id], rejected[image_id] = [], []
            with Image.open(image_path) as source:
                image = source.convert('RGB')
                width, height = image.size
                for pred in predictions[image_id]:
                    xmin, ymin, xmax, ymax = pred['box']
                    # 先截断至原图边界并排除无效框，再向外取整。
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(width, xmax), min(height, ymax)
                    if xmin >= xmax or ymin >= ymax:
                        continue
                    box = [floor(xmin), floor(ymin), ceil(xmax), ceil(ymax)]
                    crop = image.crop(tuple(box))
                    inputs = transform(crop).unsqueeze(0).to(device)#unsqueeze 增加 batch 维度

                    logits = model(inputs)
                    probabilities = torch.softmax(logits, dim=1)
                    cls_score, cls_idx = probabilities[0].max(dim=0)
                    label = idx_to_class[cls_idx.item()]
                    result = {
                        "box": box,
                        "det_score": float(pred['score']),
                        "label": label,
                        "cls_score": cls_score.item(),
                    }
                    if label == "non_defect":
                        rejected[image_id].append(result)
                    else:
                        kept[image_id].append(result)
    return kept, rejected




'''
要求：
1. model.eval()，在 torch.no_grad() 中执行。
2. 每张原图只打开一次，转换为 RGB。
3. 遍历该图片的候选框，限制坐标范围，左上取 floor、右下取 ceil，跳过无效框。
4. 裁剪后应用权重保存的预处理参数，加上 batch 维度并移动到设备。
5. 模型输出经过 softmax(dim=1)，取最高分的类别及其分数。
6. 返回两个按图片 ID 组织的字典：`kept` 和 rejected。
   - non_defect 放入 rejected。
   - 其他类别放入 kept。
   - 没有结果的图片也保留空列表。
每个结果保存：
{
    "box": [...],       # 原图坐标
    "det_score": ...,   # YOLO 分数
    "label": ...,       # 分类名称
    "cls_score": ...,   # 分类分数
}
先按最高分是否为 non_defect 决定保留，不额外加分类阈值，也不把两个分数相乘。

这个函数完成后，我们再接独立测试脚本和可视化：先用少量测试图验证接线，再跑全部 597 张，保存分类前后结果和带类别的框图

'''








