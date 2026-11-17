from pathlib import Path
from pipline import data

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT / "output/dataset/manifest.json"

# records = []
# for split in ("train", "val"):
#     records.extend(
#         data.load_dataset_from_manifest(manifest_path, split, count=5)
#     )
#
# count = data.export_gt_crops(
#     records, ROOT / "output/classifier_crops"
# )
# print(f"保存了 {count} 张裁剪图片")

records = data.load_manifest(manifest_path)
count = data.export_gt_crops(
    records, ROOT / "output/classifier_crops"
)
print(f"保存了 {count} 张裁剪图片")