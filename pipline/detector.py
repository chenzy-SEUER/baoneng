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
        draw.rectangle(pred['box'], outline='red', width=2)
        xmin, ymin, xmax, ymax = pred['box']
        draw.text((xmax, ymax + 5), text=f"{pred['score']:.2f}")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.resolve() == Path(image_path).resolve():
        raise ValueError("输出路径不能与原图相同")
    image.save(save_path)



'''
- 读取 record["image_path"]。
- 真实框用绿色，显示原始类别。
- 预测框用红色，显示置信度；第一阶段只有统一的 defect 类。
- 保存到新文件，避免覆盖原图。
- 没有预测框时也能正常保存。


'''



























