"""Shared paths and constants for the Person/Head detection dataset (all 3 tracks)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "Person Detection with head.v2i.yolov8"
DATA_YAML = DATASET_ROOT / "data.yaml"

TRAIN_IMAGES = DATASET_ROOT / "train" / "images"
TRAIN_LABELS = DATASET_ROOT / "train" / "labels"
VALID_IMAGES = DATASET_ROOT / "valid" / "images"
VALID_LABELS = DATASET_ROOT / "valid" / "labels"
TEST_IMAGES = DATASET_ROOT / "test" / "images"
TEST_LABELS = DATASET_ROOT / "test" / "labels"

CLASS_NAMES = ["Head", "Person"]
NUM_CLASSES = len(CLASS_NAMES)
