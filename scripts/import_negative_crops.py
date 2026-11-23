from pathlib import Path

from pipline.classifier import export_negative_classifier_crops

ROOT = Path(__file__).resolve().parents[1]


def main():
    counts = export_negative_classifier_crops(
        review_dir=ROOT / "output/data11_negative_voc_all_review",
        crops_dir=ROOT / "output/classifier_crops",
    )
    print("负样本导入数量：", counts)


if __name__ == "__main__":
    main()
