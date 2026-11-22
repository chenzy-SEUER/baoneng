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



#预测一张图片
def predict_image(model, image_path, conf=0.03, imgsz=640, device="0", nms_iou=None, verbose=None):
    options = {
        "conf": conf,
        "imgsz": imgsz,
        "device": device,
    }
    if nms_iou is not None:
        options["iou"] = nms_iou
    if verbose is not None:
        options["verbose"] = verbose

    results = model.predict(source=str(image_path), **options)
    boxes = results[0].boxes
    xyxys = boxes.xyxy.cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    results = []
    for xyxy, score in zip(xyxys, scores):
        results.append({"box":xyxy,"score":score})
    return results


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
        candidates = predict_image(model, record["image_path"], conf=conf, imgsz=imgsz, device=device)
        predictions[record["id"]] = candidates
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
        draw.text((xmax, ymax + 5), text=text, fill="white")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.resolve() == Path(image_path).resolve():
        raise ValueError("输出路径不能与原图相同")
    image.save(save_path)


def keep_top_half_if_over_ten(candidates):
    if len(candidates) > 10:
        candidates.sort(key=lambda prediction: prediction["score"], reverse=True)
        return candidates[:len(candidates) // 2]
    return candidates
