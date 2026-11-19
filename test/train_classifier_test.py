from pathlib import Path

import torch

from pipline import classifier

ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = ROOT / "output/classifier_crops"

if __name__ == '__main__':
    train_loader, val_loader, class_to_idx = classifier.build_classifier_loaders(CROPS_DIR, 16, 224)
    model = classifier.build_classifier(len(class_to_idx))
    device = torch.device("cpu")
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 1

    for epoch in range(epochs):
        train_metrics = classifier.train_one_epoch(model, train_loader,loss, optimizer, device)
        val_metrics = classifier.evaluate_classifier(model, val_loader, loss, device)
        print(f"{epoch=}")
        print(train_metrics)
        print(val_metrics)










'''
1. 加载 train、val 数据。
2. 设置设备，创建模型并移动到设备。
3. 创建交叉熵损失和优化器。
4. 设置 epochs = 1，先做一轮完整流程验证。
5. 在 epoch 循环中依次调用：train_metrics = classifier.train_one_epoch(
       model, train_loader, criterion, optimizer, device
   )
   val_metrics = classifier.evaluate_classifier(
       model, val_loader, criterion, device
   )
6. 每轮打印 epoch、训练 loss/accuracy、验证 loss/accuracy。
模型和优化器必须在循环外创建，否则每轮都会重新初始化。脚本仍使用 if __name__ == "__main__": 保护。

你先写好入口，我检查后再运行。下一步再补权重保存和类别映射保存，确保训练结果能够用于后续推理。

'''