import os.path
import shutil
import yaml
from pathlib import Path


def export_detector_dataset(records, out_dir, *, include_test=True):
    yaml_path = Path(out_dir) / "dataset.yaml"
    yaml_data = {
        'path': str(Path(out_dir).resolve()),
        'train': 'images/train',
        'val': 'images/val',
    }
    if include_test:
        yaml_data['test'] = 'images/test'
    yaml_data['names'] = {0: 'defect'}
    yaml_path.parent.mkdir(parents=True, exist_ok=True)#父目录
    #不存在时创建，存在时覆盖
    with yaml_path.open(mode='w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_data, f,allow_unicode=True, sort_keys=False, default_flow_style=False)


    images_dir = Path(out_dir)/'images'
    labels_dir = Path(out_dir)/'labels'
    for r in records:
        img_dataset_dir = images_dir / r['split']
        img_name = r['id']+Path(r['image_path']).suffix
        img_dataset_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r['image_path'], img_dataset_dir/img_name)

        label_text_path = labels_dir / r['split']/(r['id']+'.txt')
        label_text_path.parent.mkdir(parents=True, exist_ok=True)
        width = r['width']
        height = r['height']
        text_content = []
        for obj in r['objects']:
            xmin, ymin, xmax, ymax = obj['box']
            xc = (xmin + xmax) / 2 / width
            yc = (ymin + ymax) / 2 / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            text_content.append(f'0 {xc} {yc} {bw} {bh}')
        label_text_path.write_text('\n'.join(text_content), encoding='utf-8')







'''out_dir/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml


path: /导出目录的绝对路径
train: images/train
val: images/val
test: images/test
names:
  0: defect。

'''
