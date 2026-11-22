from pathlib import Path

from pipline import classifier

ROOT = Path(__file__).resolve().parents[1]
CROPS_DIR = ROOT / "output/classifier_crops_merged"
RUN_DIR = ROOT / "output/classifier_runs/run02_merged"


def main():
    config = {
        "epochs": 1,
        "batch_size": 16,
        "imgsz" : 224,
        "learning_rate": 1e-4,
        "device": "cuda:0"
    }

    return classifier.train_classifier(CROPS_DIR, RUN_DIR, **config)


if __name__ == "__main__":
    main()
