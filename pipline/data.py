import shutil
from collections import defaultdict
from math import floor, ceil
from pathlib import Path
import random
from PIL import Image, ImageDraw
import os
import xml.etree.ElementTree as ET
import logging
import json
from pipline import evaluate

LABELS = {"PID": "pid", "crack": "crack", "diode": "diode",
          "hotspot": "hotspot", "shelter": "shelter",
          "GB": "gb", "zhedang": "zhedang"}

IMG_SUFFIX = ".jpg"
XML_SUFFIX = ".xml"
RANDOM_SEED = 42


# TODO 留着后面完善
def audit_annotations(dataset_dir):
    issues = []
    records, _ = load_dataset(dataset_dir)
    for record in records:
        issue = {}
        for idx, obj in enumerate(record["objects"]):
            xmin, ymin, xmax, ymax = obj["box"]
            if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                issue['reason'] = 'negative coordinates'
            if ymax > record['height'] or xmax > record['width']:
                issue['reason'] = 'ymax or xmax out of bounds'
            if xmin >= xmax or ymin >= ymax:
                issue['reason'] = 'xmin or ymin greater equal than xmax or ymax'

            if issue.get('reason'):
                issue['xml_path'] = os.path.splitext(Path(record["image_path"]))[0] + XML_SUFFIX
                issue['label'] = obj["label"]

                issues.append(issue)

    return issues


'''
{
    "xml_path": "...",
    "object_index": 0,
    "label": "hotspot",
    "box": [78.0, 622.0, 101.0, 641.0],
    "image_size": [640, 640],
    "reason": "ymax_out_of_bounds",
    "overflow": {"right": 0.0, "bottom": 1.0}
}

重点要求：
1. 一个文件里所有框都检查完，不要遇到第一个异常就退出。
2. 区分负坐标、右／下越界、零面积或反向框、尺寸不一致。
3. 统计异常图片数、异常框数，以及越界幅度分布。
我们先判断这些问题是否集中在 超出1～2像素，再决定如何处理。390是异常图片记录数，不一定等于异常框数。'''


def draw_record(record, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    img_path = record['image_path']
    image = Image.open(img_path).convert('RGB')  # 统一输入通道数

    draw = ImageDraw.Draw(image)
    for obj in record['objects']:
        xmin, ymin, xmax, ymax = obj['box']
        draw.rectangle((xmin, ymin, xmax, ymax), outline='red', width=3)
        draw.text((xmin, ymin - 3), text=obj['label'])
    target_dir = save_dir / Path(img_path).name
    if target_dir.resolve() == Path(img_path).resolve():
        raise ValueError('输出路径不能与原图相同')
    image.save(Path(save_dir) / f"{Path(img_path).name}")


def draw_box(img_path, box, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(img_path).convert('RGB')  # 统一输入通道数

    draw = ImageDraw.Draw(image)
    xmin, ymin, xmax, ymax = box
    draw.rectangle((xmin, ymin, xmax, ymax), outline='red', width=4)
    image.save(Path(save_path))



def split_records(records, train_ratio=0.7, val_ratio=0.15, seed=42):
    copy_records = [dict(r) for r in records]
    rng = random.Random(seed)
    rng.shuffle(copy_records)
    split1, split2 = int(len(copy_records) * train_ratio), int(len(copy_records) * val_ratio)
    train_records = copy_records[:split1]
    val_records = copy_records[split1:split1 + split2]
    test_records = copy_records[split1 + split2:]
    ids = set()
    for r in train_records: ids.add(r['id'])
    for r in val_records: ids.add(r['id'])
    for r in test_records: ids.add(r['id'])
    if len(ids) != len(records):
        raise ValueError("数据集之间存在重复id")

    split_set = [(train_records, "train"), (val_records, "val"), (test_records, "test")]

    for splitted_records, split_name in split_set:
        label_dict = defaultdict(int)
        for r in splitted_records:
            r['split'] = split_name
            for obj in r["objects"]:
                label_dict[obj["label"]] += 1
        logging.info("数据集合：%s, 图片总数：%d, 各类别框数为：%s", split_name, len(splitted_records), label_dict)
    return train_records, val_records, test_records


def load_dataset(dataset_dir):
    dataset_name = Path(dataset_dir).name
    img_dir = os.path.join(dataset_dir, "images")
    xml_dir = os.path.join(dataset_dir, "annotations")
    records = []
    summary = {
        "image_count": 0,
        "loaded_count": 0,
        "object_count": 0,
        "class_counts": {},
        "missing_annotations": [],
        "orphan_annotations": [],
        "errors": []
    }

    class_counts = defaultdict(int)
    missing_annotations, orphan_annotations, errors = [], [], []
    img_names = set()
    for f in Path(img_dir).glob(f"*{IMG_SUFFIX}"):
        img_stem_name = f.stem
        img_name = f.name
        img_names.add(img_stem_name)
        img_path = os.path.join(img_dir, img_name)
        xml_path = os.path.join(xml_dir, img_stem_name + f'{XML_SUFFIX}')
        summary['image_count'] += 1
        if not Path(xml_path).exists():
            missing_annotations.append(xml_path)
            continue
        try:
            record = load_record(img_path, xml_path, dataset_name)
            records.append(record)
            summary['object_count'] += len(record["objects"])
            summary['loaded_count'] += 1
            for obj in record["objects"]:
                class_counts[obj["label"]] += 1
        except (ValueError, ET.ParseError, OSError) as e:
            errors.append(f"{xml_path}: {e}")

    for f in Path(xml_dir).glob("*.xml"):
        xml_name = f.stem
        if xml_name not in img_names:
            orphan_annotations.append(f.name)
    summary["class_counts"] = class_counts
    summary["missing_annotations"] = missing_annotations
    summary["orphan_annotations"] = orphan_annotations
    summary["errors"] = errors

    return records, summary


def load_record(image_path: str, xml_path, dataset_name):
    """读取一张图片及对应的VOC XML标注。"""
    record = {"image_path": image_path, "dataset": dataset_name, "objects": []}
    record["id"] = f'{dataset_name}_{Path(image_path).stem}'

    tree = ET.parse(xml_path)
    root = tree.getroot()

    with Image.open(image_path) as img:
        record["width"], record["height"] = img.size

    if record["width"] != int(root.findtext('size/width')) or record["height"] != int(root.findtext('size/height')):
        raise ValueError(f'{xml_path} size error')

    for obj in root.findall('object'):
        xmin, ymin = float(obj.findtext('bndbox/xmin')), float(obj.findtext('bndbox/ymin'))
        xmax, ymax = float(obj.findtext('bndbox/xmax')), float(obj.findtext('bndbox/ymax'))
        if not (0 <= xmin < xmax <= record["width"]
                and 0 <= ymin < ymax <= record["height"]):
            raise ValueError(
                f"{xml_path}: invalid bbox {[xmin, ymin, xmax, ymax]}"
            )
        label = obj.findtext('name')
        record["objects"].append({"label": label, "box": [xmin, ymin, xmax, ymax]})

    return record

    '''
    {
        "id": "Data10_000001",
        "split": "train",后加
        "dataset": "Data10",
        "image_path": "/完整路径/Data10/images/000001.jpg",
        "width": 640,
        "height": 640,
        "objects": [
            {
                "label": "GB",
                "box": [545.0, 147.0, 579.0, 167.0]
            }
        ]
    }
    '''


def save_manifest(records, save_path):
    file_path = Path(save_path) / "manifest.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_manifest(file):
    file_path = Path(file)
    if not file_path.exists():
        return []
    with file_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_dataset_from_manifest(file,split, count=5):
    json = load_manifest(file)
    cnt = 0
    results = []
    for record in json:
        if record['split'] == split:
            results.append(record)
            cnt+=1
        if cnt == count:
            return results
    return results


def export_gt_crops(records, out_dir):
    crop_cnt = 0
    for record in records:
        split = record['split']
        image_path = record['image_path']
        image_id = record['id']
        if split not in ('train', 'val'):
            continue
        image = Image.open(image_path).convert('RGB')
        for idx, obj in enumerate(record['objects']):
            label = obj['label']
            xmin, ymin, xmax, ymax = obj['box']
            width, height = image.size
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width, xmax)
            ymax = min(height, ymax)
            if xmin >= xmax or ymin >= ymax:
                continue
            xmin, ymin, xmax, ymax = floor(xmin), floor(ymin), ceil(xmax), ceil(ymax)
            im_crop = image.crop((xmin, ymin, xmax, ymax))
            crop_path = Path(out_dir) /split/label/f'{image_id}_{idx}.jpg'
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            im_crop.save(crop_path)
            crop_cnt += 1
    return crop_cnt

def export_non_defect_crop(image_path, box, out_path):
    image = Image.open(image_path).convert('RGB')
    xmin, ymin, xmax, ymax = box
    width, height = image.size
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)
    if xmin >= xmax or ymin >= ymax:
        return False
    xmin, ymin, xmax, ymax = floor(xmin), floor(ymin), ceil(xmax), ceil(ymax)
    im_crop = image.crop((xmin, ymin, xmax, ymax))
    crop_path = Path(out_path)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    im_crop.save(crop_path)

    return True

def export_candidate_crops(records, predictions, out_dir):
    train_saved_cnt, val_saved_cnt = 0, 0
    for record in records:
        image_id = record['id']
        image_path = record['image_path']
        split = record['split']
        objects = record.get('objects', [])
        for idx, prediction in enumerate(predictions[image_id]):
            iou, gt_coverage = evaluate.measure_candidate_overlap(prediction["box"], objects)
            if iou == gt_coverage == 0.0:
                exported = export_non_defect_crop(image_path, prediction["box"], f'{out_dir}/{split}/{image_id}_pred_{idx}.jpg')
                if exported:
                    draw_box(image_path, prediction["box"], f'{out_dir}/{split}/raw/{image_id}_pred_{idx}.jpg')
                if exported and split == 'train':
                    train_saved_cnt += 1
                elif exported and split == 'val':
                    val_saved_cnt += 1
    print(f'{train_saved_cnt=}, {val_saved_cnt=}')
    return train_saved_cnt + val_saved_cnt



def merge_classifier_classes(src_dir, dst_dir):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if any(dst_dir.iterdir()):
        raise FileExistsError('目标文件夹存在')
    result = {'train':defaultdict(int),'val':defaultdict(int)}
    for split in ('train', 'val'):
        if not (src_dir / split).is_dir():
            raise FileNotFoundError(f"源目录不存在：{src_dir / split}")
        split_src_dir = src_dir / split
        split_dst_dir = dst_dir / split
        split_dst_dir.mkdir(parents=True, exist_ok=True)
        for p in split_src_dir.iterdir():
            if not p.is_dir():
                continue
            current_dir = Path(p)
            count = sum(1 for path in current_dir.iterdir() if path.is_file())

            cls_name = p.name if p.name !='GB' else 'crack'
            shutil.copytree(split_src_dir / p.name, split_dst_dir / cls_name, dirs_exist_ok=True)
            result[split][cls_name] += count

    return result

'''
要求：
1. 从现有 output/classifier_crops 读取，导出到新目录 output/classifier_crops_merged，保留旧数据和原始 XML。
2. 保持 train、val 划分不变。
3. GB 和 crack 的图片都复制到新目录的 crack/。
4. 其他类别按原名复制。
5. 文件名前加原类别，例如 GB_原文件名.jpg，避免合并后同名覆盖。
6. 目标目录已存在且非空时拒绝执行。
7. 返回各 split、各类别的实际复制数量。
'''

























