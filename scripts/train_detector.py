"""Train the plant detector of the two-stage pipeline: scripts/train_maskrcnn.py with
configs/detector.yaml (configs/maskrcnn.yaml on the plant-only data). See docs/twostage.md.
"""

import train_maskrcnn

CONFIG = "configs/detector.yaml"

if __name__ == "__main__":
    train_maskrcnn.main(CONFIG)
