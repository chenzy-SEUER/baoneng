import logging
import sys
from pathlib import Path

import pipline.data as data
import pipline.export_yolo as export_yolo
import pipline.detector as detector


def main():
    logging.info("程序启动")
    data10_dir = 'data/Data10'
    data11_dir = 'data/Data11'
    OUT_DIR = 'output'
    IMAGE_DATASET_DIR = f'{OUT_DIR}/dataset'
    YOLO_DETECT_DIR = f'{OUT_DIR}/yolo_detect'
    DATASET_YAML = f'{IMAGE_DATASET_DIR}/dataset.yaml'
    PREDICT_DIT = f'{OUT_DIR}/predictions'
    PREDICT_JSON_FILE = f'{PREDICT_DIT}/sample.json'


    out_dir = Path(IMAGE_DATASET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise ValueError("output dir not empty")
    data10_records, summary = data.load_dataset(data10_dir)
    logging.info(f'loaded {data10_dir} image count:%d, missing_annotations:%s, orphan_annotations:%s, errors:%s',
                 summary['image_count'], summary['missing_annotations'], summary['orphan_annotations'], summary['errors'])
    data11_records, summary = data.load_dataset(data11_dir)
    logging.info(f'loaded {data11_dir} image count:%d, missing_annotations:%s, orphan_annotations:%s, errors:%s',
                 summary['image_count'], summary['missing_annotations'], summary['orphan_annotations'],
                 summary['errors'])

    all_records = []
    for ds1, ds2 in zip(data.split_records(data10_records),data.split_records(data11_records)):
        ds = ds1 + ds2
        all_records.extend(ds)
        export_yolo.export_detector_dataset(ds, str(IMAGE_DATASET_DIR))
    data.save_manifest(all_records, str(IMAGE_DATASET_DIR))

    best_weights_path = detector.train_detector(yaml_path=DATASET_YAML,model_path='yolo11n.pt',out_dir=YOLO_DETECT_DIR,epochs=1)




    '''
    下一步可以用已有的 best.pt 做真实推理验证：
1. 从 manifest.json 选取少量训练集、验证集图片，例如各 5 张。
2. 调用 predict_records()，将结果保存到 output/predictions/sample.json。
3. 检查每张图片都有对应 ID，框坐标和置信度正确保存。
直接加载已有权重即可，无需重新训练。本次尚未运行真实模型推理。
    '''


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    main()
