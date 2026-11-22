from pathlib import Path

from pipline import data

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "output/dataset/manifest.json"
CROPS_DIR = ROOT / "output/classifier_crops"


def main():
    records = data.load_manifest(MANIFEST_PATH)
    count = data.export_gt_crops(records, CROPS_DIR)
    print(f"保存了 {count} 张裁剪图片")


if __name__ == "__main__":
    main()
