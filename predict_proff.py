from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "crack_yolo11s_seg_20261031_111244" / "weights" / "best.pt"
IMAGE_FOLDER = ROOT / "proff_images"
OUTPUT_FOLDER = ROOT / "runs"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

if not IMAGE_FOLDER.is_dir():
    raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")

model = YOLO(str(MODEL_PATH))

# Stream predictions one image at a time to limit GPU memory use.
for _ in model.predict(
    source=str(IMAGE_FOLDER),
    imgsz=640,
    conf=0.10,
    iou=0.50,
    retina_masks=True,
    device=0,
    save=True,
    stream=True,
    project=str(OUTPUT_FOLDER),
    name="proff_predictions",
    exist_ok=True,
    verbose=False,
):
    pass

print(f"Predictions saved in: {OUTPUT_FOLDER / 'proff_predictions'}")