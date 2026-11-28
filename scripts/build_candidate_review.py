"""生成可离线审核检测候选的缩略图页面。"""

import argparse
from base64 import b64encode
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "output/restart_80_20"


@lru_cache(maxsize=4)
def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def context_image(path, box):
    image = load_image(path)
    x1, y1, x2, y2 = box
    padding = max(72, round(max(x2 - x1, y2 - y1) * 1.25))
    left, top = max(0, x1 - padding), max(0, y1 - padding)
    right, bottom = min(image.width, x2 + padding), min(image.height, y2 + padding)
    preview = ImageOps.contain(image.crop((left, top, right, bottom)), (600, 500))
    sx, sy = preview.width / (right - left), preview.height / (bottom - top)
    outline = ((x1 - left) * sx, (y1 - top) * sy, (x2 - left) * sx, (y2 - top) * sy)
    draw = ImageDraw.Draw(preview)
    draw.rectangle(outline, outline="#131510", width=4)
    draw.rectangle(outline, outline="#ffe060", width=2)
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=85)
    return b64encode(buffer.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=EXPERIMENT_DIR / "classifier_preparation")
    parser.add_argument("--dataset-dir", type=Path, default=EXPERIMENT_DIR / "ir_no_data08_data09")
    parser.add_argument("--out", type=Path, help="默认在审核目录生成 review_board.html")
    args = parser.parse_args()

    raw = (args.review_dir / "review_candidates.json").read_bytes()
    records = json.loads(raw)
    if not records or len({record["crop"] for record in records}) != len(records):
        raise ValueError("审核清单为空或含重复裁剪")
    suggestions = {}
    for modality in sorted({record["modality"] for record in records}):
        manifest = args.dataset_dir / modality / "manifest.json"
        if manifest.is_file():
            originals = json.loads(manifest.read_text(encoding="utf-8"))
            suggestions[modality] = sorted({obj["label"] for image in originals for obj in image["objects"]})
    payload = {"id": sha256(raw).hexdigest(), "items": [], "suggestions": suggestions}
    for record in records:
        crop = args.review_dir / record["crop"]
        payload["items"].append({
            "record": record,
            "name": crop.name,
            "thumbnail": b64encode(crop.read_bytes()).decode("ascii"),
            "context": context_image(args.dataset_dir / record["source_image"], record["xyxy"]),
        })

    template = Path(__file__).with_name("candidate_review.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    out = args.out or args.review_dir / "review_board.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.replace("{{REVIEW_DATA}}", data), encoding="utf-8")
    print(f"已生成 {len(records)} 张候选的审核页：{out.resolve()}")


if __name__ == "__main__":
    main()
