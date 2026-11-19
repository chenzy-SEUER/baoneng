import json
from pathlib import Path

import torch
from pipline import data,classifier,detector

ROOT = Path(__file__).resolve().parents[1]

manifest=ROOT / "output/dataset/manifest.json"
detector_weights = ROOT/"output/yolo_detect_50ep/train2/weights/best.pt"
classifier_weight = ROOT/"output/classifier_runs/run02_merged/last.pt"
pred_out_dir = ROOT/"output/two_stage_predictions/pred/sample05"
classifier_out_dir = ROOT/"output/two_stage_predictions/classifier/sample05"


if __name__ == '__main__':

    for path in (manifest, detector_weights, classifier_weight):
        if not path.is_file():
            raise FileNotFoundError(path)
    pred_out_dir.mkdir(parents=True, exist_ok=False)
    out_dir = classifier_out_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #2
    records = data.load_manifest(manifest)
    test_data = [record for record in records if record['split'] == "test"]
    test_data10 = [record for record in test_data if record['dataset'] == "Data10"][:5]
    test_data11 = [record for record in test_data if record['dataset'] == "Data11"][:5]
    test_data = []
    test_data.extend(test_data10)
    test_data.extend(test_data11)

    #3
    model, transform, idx_to_class = classifier.build_classifier_with_weights(classifier_weight)

    #4
    predictions = detector.predict_records(
        detector_weights, test_data,
        pred_out_dir / "candidates.json",
        conf=0.03,
        device=str(device),
    )

    #5
    kept, rejected = classifier.classify_candidates(
        model, test_data, predictions, transform, idx_to_class, device
    )

    #6

    kept_file = out_dir / "kept.json"
    with open(kept_file, "w", encoding='utf-8') as f:
        json.dump(kept, f)
    rejected_file = out_dir / "rejected.json"
    with open(rejected_file, "w", encoding='utf-8') as f:
        json.dump(rejected, f)
    for record in test_data:
        detector.draw_classified_prediction(record, kept, out_dir/"vis" / f"{record['id']}.jpg")

    candidate_count = sum(len(v) for v in predictions.values())
    kept_count = sum(len(v) for v in kept.values())
    rejected_count = sum(len(v) for v in rejected.values())
    skipped_count = candidate_count - kept_count - rejected_count

    print(
        f"图片数={len(test_data)}, 候选数={candidate_count}, "
        f"保留数={kept_count}, 剔除数={rejected_count}, "
        f"无效框跳过数={skipped_count}"
    )


'''
1. 配置路径
   - manifest：`output/dataset/manifest.json`
   - YOLO 权重：此前使用的 output/yolo_detect_50ep/train2/weights/best.pt
   - 分类权重：`output/classifier_runs/run02_merged/last.pt`
   - 新输出目录：`output/two_stage_predictions/sample01`
   检查输入文件存在，输出目录使用 exist_ok=False，避免覆盖。
2. 读取测试记录
   用 data.load_manifest()，筛选 split == "test"，再按 dataset 各取 5 张。将相对 image_path 按 ROOT 转为绝对路径。
3. 加载分类模型
   参考现有评估脚本：读取 checkpoint，创建 resnet50(weights=None)，替换 fc、加载参数并移动到设备。根据 checkpoint 创建预处理，并反转类别映射：
   idx_to_class = {
       idx: name for name, idx in checkpoint["class_to_idx"].items()
   }
4. 第一阶段检测
   predictions = detector.predict_records(
       detector_weights, records,
       out_dir / "candidates.json",
       conf=0.03,
       device=str(device),
   )
5. 第二阶段分类
   kept, rejected = classifier.classify_candidates(
       model, records, predictions, transform, idx_to_class, device
   )
6. 保存结果和图片
   - 将 kept、`rejected` 分别写入 UTF-8 JSON。
   - 对每张图片调用：detector.draw_classified_prediction(
         record, kept, out_dir / "vis" / f"{record['id']}.jpg"
     )
   - 打印图片数、候选框总数、保留数和剔除数。
当前分类函数会跳过无效框，因此可能出现“保留数＋剔除数小于候选数”；将差值打印为无效框跳过数即可。
'''