from collections import defaultdict
from pathlib import Path
import json

from PIL import Image, ImageDraw
from ultralytics import YOLO


def train_detector(
        yaml_path, model_path, out_dir, epochs=1,
        imgsz=640, batch=4, device="0",
):
    model = YOLO(model_path)
    model.train(data=yaml_path, project=out_dir, epochs=epochs, imgsz=imgsz, batch=batch, device=device, workers=4)

    best_weights_path = Path(model.trainer.best)
    if not best_weights_path.exists():
        raise FileNotFoundError(best_weights_path)
    return str(best_weights_path)


def predict_records(
    weights_path,
    records,
    save_path,
    conf=0.03,
    imgsz=640,
    device="0",
):
    model = YOLO(weights_path)
    predictions = {}
    for record in records:
        predictions[record["id"]]=[]
        results = model.predict(source=record['image_path'], conf=conf, imgsz=imgsz, device=device)
        boxes = results[0].boxes
        xyxys = boxes.xyxy.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        for xyxy, score in zip(xyxys, scores):
            predictions[record["id"]].append({"box":xyxy,"score":score})
    json_file_path = Path(save_path)
    json_file_path.parent.mkdir(parents=True, exist_ok=True)
    with json_file_path.open('w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    return predictions

#save_path 是完整路径xx/xx.jpg
def draw_prediction(record, predictions, save_path):
    image_path = record['image_path']
    image_id = record['id']
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for obj in record['objects']:
        draw.rectangle(obj['box'],outline='green', width=2)
        xmin, ymin, xmax, ymax = obj['box']
        draw.text((xmin, ymin - 5), text=obj['label'])
    for pred in predictions[image_id]:
        draw.rectangle(pred['box'], outline='white', width=2)
        xmin, ymin, xmax, ymax = pred['box']
        draw.text((xmax, ymax + 5), text=f"{pred['score']:.2f}")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.resolve() == Path(image_path).resolve():
        raise ValueError("输出路径不能与原图相同")
    image.save(save_path)


#save_path 是完整路径xx/xx.jpg
def draw_classified_prediction(record, predictions, save_path):
    image_path = record['image_path']
    image_id = record['id']
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    objects = record['objects']
    for obj in objects:
        displayname = obj['label'] if obj['label']!='GB' else 'crack'
        box = obj['box']
        xmin, ymin, xmax, ymax = box
        draw.rectangle(box,outline='green', width=2)
        draw.text(
            (xmin, max(0, ymin - 12)),
            text=displayname,
            fill="green",
        )
    for pred in predictions[image_id]:
        box = pred['box']
        xmin, ymin, xmax, ymax = box
        draw.rectangle(box, outline='white', width=2)
        text = (
            f"{pred['label']} "
            f"D:{pred['det_score']:.2f} "
            f"C:{pred['cls_score']:.2f}"
        )
        draw.text((xmax, ymax + 5), text=text, fill="red")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.resolve() == Path(image_path).resolve():
        raise ValueError("输出路径不能与原图相同")
    image.save(save_path)


'''
可以参考现有的 draw_prediction，实现要求：
1. 打开原图，转换为 RGB。
2. 绘制真实标注：
   - 绿色框。
   - 显示真实类别；`GB` 显示为 crack，与当前分类口径一致，但不要修改原 record。
3. 遍历 predictions[record["id"]]，绘制分类后保留的候选：
   - 红色框，坐标取 pred["box"]。
   - 显示类别、检测分数和分类分数，例如：crack D:0.83 C:0.96
   - 对应字段是 label、`det_score`、`cls_score`。
4. 创建输出父目录，保存到完整图片路径。
5. 输出路径与原图相同时报错，避免覆盖。
6. 没有保留框时，也正常保存图片。
这里传入的是 classify_candidates 返回的 kept 字典；`rejected` 先只存 JSON，不画在最终结果图上。
'''



























