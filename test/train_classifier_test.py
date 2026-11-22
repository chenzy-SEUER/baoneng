"""兼容旧入口，正式命令为 python -m scripts.train_classifier。"""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_classifier import main


if __name__ == "__main__":
    main()
