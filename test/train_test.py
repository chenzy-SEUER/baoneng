from pathlib import Path
from pipline import detector

ROOT = Path(__file__).resolve().parents[1]


def main():
    yaml_path = ROOT / "output/dataset/dataset.yaml"
    model_path = ROOT / "yolo11n.pt"

    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    best_weights_path = detector.train_detector(
        yaml_path=str(yaml_path),
        model_path=str(model_path),
        out_dir=str(ROOT / "output/yolo_detect_50ep"),
        epochs=50,
        imgsz=640,
        batch=4,
        device="0",
    )
    print(f"最佳权重：{best_weights_path}")


if __name__ == "__main__":
    main()