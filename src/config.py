"""Paths and constants. Override paths with environment variables if needed."""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE_DIR = Path(os.environ.get("MARITIME_SCENE_DIR", ROOT / "data" / "processed_encounters_v5"))
MANIFEST = SCENE_DIR / "day_to_scenes.json"
CHECKPOINT = Path(os.environ.get("MARITIME_CHECKPOINT", ROOT / "checkpoints" / "checkpoint_model_b_v3.pt"))

# Must match the preprocessing pipeline
ORIGIN_LON, ORIGIN_LAT = 11.0, 55.5
METERS_PER_DEG_LAT = 111132.954
METERS_PER_DEG_LON = 111132.954 * np.cos(np.radians(ORIGIN_LAT))

OBS_LEN, PRED_LEN = 8, 12
DT_SECONDS = 30
DIST_THRESH = 3500.0
SPLIT_SEED = 42            # same seed/ratio as training split
VAL_FRACTION = 0.2

# Reference numbers from the training run (edit if you retrain)
REF_VAL_ADE = 87.3
REF_VAL_FDE = 189.8
