from pathlib import Path
from pipline import data, detector
from pipline import evaluate

ROOT = Path(__file__).resolve().parents[1]
best_weights_path = ROOT / "output/yolo_detect_50ep/train2/weights/best.pt"
manifest_path = ROOT / "output/dataset/manifest.json"
sample_json_prefix = ROOT / "output/predictions/"
prediction_draw = ROOT / "output/predictions/vis_crop_sample"

if __name__ == "__main__":

    for split in ('train', 'val'):
        records = data.load_dataset_from_manifest(manifest_path, split, count=5)
        predictions = detector.predict_records(best_weights_path, records,
                                         f'{sample_json_prefix}/{split}/crop.json', device="cpu")
        image_count = 0
        pred_box_cnt = 0
        no_lap_cnt = 0
        lap_cnt = 0
        for record in records:
            image_count += 1
            image_id = record['id']
            pred_box_cnt += len(predictions[image_id])
            detector.draw_prediction(record, predictions, Path(prediction_draw) / split / f'{image_id}.jpg')

            objects = record.get('objects', [])
            for prediction in predictions[image_id]:
                iou, gt_coverage = evaluate.measure_candidate_overlap(prediction["box"], objects)
                if iou==gt_coverage==0.0:
                    no_lap_cnt += 1
                else:
                    lap_cnt += 1
        assert no_lap_cnt + lap_cnt == pred_box_cnt
        print(f'{split} dataset: {no_lap_cnt=}, {lap_cnt=}')
        print(f"{split} dataset: {image_count=}, {pred_box_cnt=}")

        saved_count = data.export_candidate_crops(
            records, predictions, ROOT / "output/candidate_review"
        )
        print(f"{split} 实际保存裁剪数：{saved_count}")



'''
- 对每个候选框调用该函数。
- 两项都为 0，计入“与标注无重叠候选数”。
- 其余计入“与标注有重叠候选数”。
- 分别打印 train、val 的结果，检查两组数量之和等于候选框总数。
暂时称它们为“无重叠候选”，还不能直接认定为非缺陷，后面需要通过裁剪图检查是否存在漏标
'''
