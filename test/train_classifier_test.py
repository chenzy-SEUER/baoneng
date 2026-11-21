from pathlib import Path

from pipline import classifier

ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = ROOT / "output/classifier_crops_merged"
RUN_DIR = ROOT / "output/classifier_runs/run02_merged"


def main():
    return classifier.train_classifier(CROPS_DIR, RUN_DIR)


if __name__ == "__main__":
    main()
