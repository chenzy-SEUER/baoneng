"""在 Ubuntu GPU 机训练配对检测/分割实验，不覆盖既有输出。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--task', choices=['both', 'detect', 'segment'], default='both')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    if args.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('未发现CUDA；请在Ubuntu GPU机器运行全量训练')
    dataset = args.dataset_dir.resolve()
    project = args.project.resolve()
    metadata = json.loads((dataset/'experiment.json').read_text(encoding='utf-8'))
    tasks = ['detect', 'segment'] if args.task == 'both' else [args.task]
    for task in tasks:
        if not (dataset/task/'dataset.yaml').is_file():
            raise FileNotFoundError(dataset/task/'dataset.yaml')
        if (project/task).exists():
            raise FileExistsError(f'输出目录已存在，请更换project：{project/task}')
    print('标签状态：', metadata['label_status'], '数据量：', metadata['images_by_split'], flush=True)
    for task in tasks:
        model_name = 'yolo11n.pt' if task == 'detect' else 'yolo11n-seg.pt'
        local = ROOT/model_name
        model = YOLO(str(local) if local.is_file() else model_name)
        data = dataset/task/'dataset.yaml'
        model.train(data=str(data), project=str(project), name=task, exist_ok=False,
                    epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, workers=args.workers,
                    device=args.device, seed=42, deterministic=True)
        folder = Path(model.trainer.save_dir)
        (folder/'comparison_source.json').write_text(json.dumps({
            'dataset_metadata_sha256': hashlib.sha256((dataset/'experiment.json').read_bytes()).hexdigest(),
            'label_status': metadata['label_status'], 'dataset_dir': str(dataset),
            'weights': model_name, 'task': task, 'epochs': args.epochs, 'batch': args.batch,
            'imgsz': args.imgsz, 'seed': 42, 'deterministic': True,
        }, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print('完成：', folder, flush=True)


if __name__ == '__main__':
    main()
