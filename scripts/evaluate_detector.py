"""Evaluate the plant detector of the two-stage pipeline on its own (does it find more plants
than the species model?): scripts/evaluate_maskrcnn.py with configs/detector.yaml. Taxonomic
levels and decisions are skipped, since `plant` is not a species. See docs/twostage.md.
"""

import evaluate_maskrcnn

CONFIG = "configs/detector.yaml"

if __name__ == "__main__":
    evaluate_maskrcnn.main(CONFIG)
