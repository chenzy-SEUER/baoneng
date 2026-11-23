"""将分类裁剪数据集中的 GB 合并到 crack。"""

import argparse
from pathlib import Path

from pipline import data

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "output/classifier_crops"
DST_DIR = ROOT / "output/classifier_crops_merged"


def main(*, check_expected_counts=False):
    counts = data.merge_classifier_classes(SRC_DIR, DST_DIR)
    if check_expected_counts:
        check_counts(counts)
    print(f"类别合并完成，输出目录：{DST_DIR}")
    return counts


def check_counts(counts):
    expected_totals = {"train": 13376, "val": 2793}
    expected_crack = {"train": 572, "val": 107}

    for split in ("train", "val"):
        # 按实际生成的文件统计，确认没有因同名覆盖等原因丢失图片。
        actual_counts = {
            folder.name: sum(p.is_file() for p in folder.iterdir())
            for folder in (DST_DIR / split).iterdir()
            if folder.is_dir()
        }
        total = sum(actual_counts.values())
        print(f"{split} 各类别数量：{actual_counts}")
        print(f"{split} 总数：{total}")

        assert actual_counts == dict(counts[split]), f"{split} 实际文件数与复制统计不一致"
        assert "GB" not in actual_counts, f"{split} 仍存在 GB 类别"
        assert len(actual_counts) == 7, f"{split} 应包含 7 个类别"
        assert actual_counts["crack"] == expected_crack[split], f"{split} crack 数量不符合预期"
        assert total == expected_totals[split], f"{split} 总数不符合预期"

    print("类别合并及数量检查通过，输出目录：", DST_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-expected-counts", action="store_true", help="核对原实验数据集的样本数量")
    args = parser.parse_args()
    main(check_expected_counts=args.check_expected_counts)
