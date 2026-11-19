from pathlib import Path

import torch

from pipline import classifier

ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = ROOT / "output/classifier_crops"

run_dir = ROOT / "output/classifier_runs/run01"

if __name__ == '__main__':
    train_loader, val_loader, class_to_idx = classifier.build_classifier_loaders(CROPS_DIR, 16, 224)
    device = torch.device("cuda:0")
    run_dir.mkdir(parents=True, exist_ok=False)
    model = classifier.build_classifier(len(class_to_idx))
    model.to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 1

    for epoch in range(epochs):

        train_metrics = classifier.train_one_epoch(model, train_loader,loss, optimizer, device)
        val_metrics = classifier.evaluate_classifier(model, val_loader, loss, device)
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": class_to_idx,
                "num_classes": len(class_to_idx),
                "architecture": "resnet50",
                "imgsz": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            run_dir / "last.pt",
        )











'''

'''