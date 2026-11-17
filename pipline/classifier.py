import os

from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def build_classifier_loaders(crops_dir, batch_size=32, imgsz=224):
    transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFoler(root=os.path.join(crops_dir,'train'), transform=transform)
    val_ds = datasets.ImageFoler(root=os.path.join(crops_dir, 'val'), transform=transform)

    train_ds.class_to_idx

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_ds.class_to_idx

'''要求：
- 使用 torchvision.datasets.ImageFolder 分别读取 train、val。
- 图片统一缩放到 (imgsz, imgsz)，再转成 Tensor。
- train 设置 shuffle=True，val 设置 shuffle=False。
- 检查两者的 class_to_idx 完全一致。
- 返回训练 loader、验证 loader 和类别映射。
写完后，取一个 batch，打印图片张量形状、标签和类别映射；这一步先不训练。
'''


















