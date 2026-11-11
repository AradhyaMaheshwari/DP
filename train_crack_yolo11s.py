#!/usr/bin/env python3
"""
Crack segmentation - production training pipeline (YOLO11s-seg)
================================================================
What this script does, end to end:

  1. Converts your mask images into YOLO polygon labels (robust to 0/1 masks,
     0/255 masks, inverted masks, JPEG noise, hair-thin cracks).
  2. Builds a stratified train/val split (crack / no-crack images are balanced
     across both) and keeps archive/test untouched for the final exam.
  3. Trains yolo11s-seg with settings tuned for THIN structures
     (higher resolution, full-resolution mask loss, orientation-safe augmentation).
  4. Automatically retries with a smaller batch if the GPU runs out of memory.
  5. Evaluates with standard mAP *and* pixel-level crack metrics, then picks the
     best confidence threshold on the VALIDATION set (no test-set leakage).
  6. Writes deploy/<run_name>/ (weights, deployment_config.json, ONNX). Nothing is overwritten;
     add --promote to copy a run to deploy/, which predict_cracks.py reads by default.

Usage (run from inside your DP folder, next to 'archive'):
    pip install -U ultralytics opencv-python
    python train_crack_yolo11s.py

Useful options:
    python train_crack_yolo11s.py --epochs 150 --imgsz 800 --batch 4
    python train_crack_yolo11s.py --resume          # continue an interrupted run
    python train_crack_yolo11s.py --skip-prepare    # reuse the converted dataset
    python train_crack_yolo11s.py --train-on-test   # FINAL model only; needs LABELLED archive/test (images + masks)
    python train_crack_yolo11s.py --promote         # make this run the one predict_cracks.py uses
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = ROOT / "archive"
DATASET = ROOT / "crack_yolo_dataset"
RUNS_DIR = ROOT / "runs"
DEPLOY_DIR = ROOT / "deploy"
LOG_DIR = ROOT / "logs"

MODEL_WEIGHTS = "yolo11s-seg.pt"
RUN_PREFIX = "crack_yolo11s_seg"
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIN_CONTOUR_LEN = 8          # ignore specks shorter than this many pixels (perimeter)
TOL_PX = 2                   # pixel tolerance for the "tolerant" crack metrics
PREDICT_IOU = 0.5            # NMS IoU used at inference (also used for threshold search)
CONF_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

log = logging.getLogger("crack")


# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
def setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, encoding="utf-8")],
    )
    return log_file


# ----------------------------------------------------------------------------
# 1. MASKS -> YOLO POLYGON LABELS
# ----------------------------------------------------------------------------
def load_binary_mask(path: Path):
    """Read a mask of any common flavour and return a clean uint8 {0,255} array."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., :3].max(axis=2)
    mx = int(m.max())
    if mx == 0:
        return np.zeros(m.shape[:2], np.uint8)
    # 0/1 masks, 0/255 masks, 16-bit; the absolute floor stops noise-only masks (tiny max value) becoming 'cracks'
    binary = (m > 0) if mx <= 1 else (m > max(mx * 0.5, 16))
    return binary.astype(np.uint8) * 255


def load_mask_oriented(path: Path):
    """Binary mask with cracks = 255. Polarity is decided PER MASK: a crack never covers most of an
    image, so a mask that is >50% white is dark-cracks-on-white and gets inverted.
    Returns (mask or None, was_inverted)."""
    b = load_binary_mask(path)
    if b is None:
        return None, False
    if float((b > 0).mean()) > 0.5:
        return cv2.bitwise_not(b), True
    return b, False


def count_enclosed_holes(binary, min_area=30) -> int:
    """Loop/mesh cracks enclose background. YOLO polygons cannot represent holes, so the enclosed
    region becomes 'crack' in the training label. This only counts them so the log can warn you."""
    contours, hier = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    return sum(1 for i, c in enumerate(contours) if hier[0][i][3] != -1 and cv2.contourArea(c) >= min_area)


def mask_to_polygon_lines(binary):
    """Binary mask -> list of YOLO-seg label lines (class 0 = crack)."""
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.arcLength(cnt, True) < MIN_CONTOUR_LEN:
            continue
        pts = cv2.approxPolyDP(cnt, 0.5, True)
        if len(pts) < 3:                                  # hair-thin straight crack: give it 1px width
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(cnt)
            pts = cv2.boxPoints(((cx, cy), (rw + 1.0, rh + 1.0), ang)).reshape(-1, 1, 2)
        pts = pts.reshape(-1, 2).astype(np.float64)
        pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
        lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))
    return lines


# ----------------------------------------------------------------------------
# 2. DATASET DISCOVERY, SPLIT, CONVERSION
# ----------------------------------------------------------------------------
def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def find_image_and_mask_dirs(split_dir: Path):
    subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
    mask_dir = next((d for d in subdirs if "mask" in d.name.lower()), None)
    img_dir = next((d for d in subdirs if d != mask_dir), None)
    return img_dir, mask_dir


def match_mask(img_path: Path, lookup: dict):
    s = img_path.stem.lower()
    for key in (s, f"{s}_mask", f"{s}_label", f"{s}_gt", f"mask_{s}", f"mask{s}", f"{s}_seg"):
        if key in lookup:
            return lookup[key]
    return None


def collect_pairs(split_dir: Path):
    if not split_dir.exists():
        return []
    img_dir, mask_dir = find_image_and_mask_dirs(split_dir)
    if img_dir is None or mask_dir is None:
        log.warning("Could not find an images/masks folder pair in %s", split_dir)
        return []
    lookup = {p.stem.lower(): p for p in list_images(mask_dir)}
    pairs, missing = [], 0
    for img in list_images(img_dir):
        m = match_mask(img, lookup)
        if m is None:
            missing += 1
        else:
            pairs.append((img, m))
    log.info("%-5s images='%s' masks='%s' -> %d pairs, %d images without a mask",
             split_dir.name, img_dir.name, mask_dir.name, len(pairs), missing)
    return pairs


def stratified_split(pairs, flags, val_split):
    """Keep the crack / no-crack ratio identical in train and val."""
    pos = [p for p, f in zip(pairs, flags) if f]
    neg = [p for p, f in zip(pairs, flags) if not f]
    rng = random.Random(SEED)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def cut(group):
        if len(group) < 2:
            return 0
        return max(1, int(round(len(group) * val_split)))

    n_pos, n_neg = cut(pos), cut(neg)
    return pos[n_pos:] + neg[n_neg:], pos[:n_pos] + neg[:n_neg]


def convert_split(pairs, split):
    img_out = DATASET / "images" / split
    lbl_out = DATASET / "labels" / split
    msk_out = DATASET / "masks" / split
    for d in (img_out, lbl_out, msk_out):
        d.mkdir(parents=True, exist_ok=True)

    stats = dict(images=0, with_cracks=0, instances=0, skipped_unreadable=0, skipped_aspect=0,
                 skipped_duplicate=0, inverted_masks=0, suspicious_large_masks=0, masks_with_holes=0)
    seen = set()
    for img_path, mask_path in pairs:
        # Output name carries the origin folder (train__x / test__x) so merged sets can never overwrite
        # each other; Windows paths are case-insensitive, hence .lower().
        stem = f"{img_path.parent.parent.name}__{img_path.stem}"
        if stem.lower() in seen:
            log.warning("Duplicate image stem, skipped: %s", img_path)
            stats["skipped_duplicate"] += 1
            continue
        img = cv2.imread(str(img_path))
        binary, was_inverted = load_mask_oriented(mask_path)
        if img is None or binary is None:
            log.warning("Unreadable image or mask, skipped: %s", img_path.name)
            stats["skipped_unreadable"] += 1
            continue
        ih, iw = img.shape[:2]
        mh, mw = binary.shape
        if abs(iw / ih - mw / mh) > 0.02:                 # labels are normalised: aspect must agree
            log.warning("Image/mask aspect mismatch (%dx%d vs %dx%d), skipped: %s", iw, ih, mw, mh, img_path.name)
            stats["skipped_aspect"] += 1
            continue
        seen.add(stem.lower())

        lines = mask_to_polygon_lines(binary)
        shutil.copy2(img_path, img_out / f"{stem}{img_path.suffix.lower()}")
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
        cv2.imwrite(str(msk_out / f"{stem}.png"), binary)

        stats["images"] += 1
        stats["with_cracks"] += int(bool(lines))
        stats["instances"] += len(lines)
        stats["inverted_masks"] += int(was_inverted)
        stats["suspicious_large_masks"] += int((binary > 0).mean() > 0.25)
        stats["masks_with_holes"] += int(count_enclosed_holes(binary) > 0)

    log.info("  [%-5s] %d images | %d with cracks | %d regions | skipped: %d unreadable, %d aspect, %d duplicate",
             split, stats["images"], stats["with_cracks"], stats["instances"],
             stats["skipped_unreadable"], stats["skipped_aspect"], stats["skipped_duplicate"])
    if stats["inverted_masks"]:
        log.warning("  [%s] %d/%d masks were dark-on-white and were inverted - check your sources are consistent",
                    split, stats["inverted_masks"], stats["images"])
    if stats["suspicious_large_masks"]:
        log.warning("  [%s] %d masks have >25%% foreground - open a few, labels may be wrong",
                    split, stats["suspicious_large_masks"])
    if stats["masks_with_holes"]:
        log.warning("  [%s] %d masks contain closed crack loops; the enclosed area is labelled as crack "
                    "(YOLO polygons cannot have holes)", split, stats["masks_with_holes"])
    return stats


def build_dataset(archive: Path, val_split: float, train_on_test: bool = False):
    log.info("Preparing dataset from %s", archive)

    train_pairs = collect_pairs(archive / "train")
    if not train_pairs:
        raise SystemExit(f"No image/mask pairs found in {archive / 'train'}. "
                         "Check the folder names (the masks folder must contain 'mask').")
    test_pairs = collect_pairs(archive / "test")
    if train_on_test and not test_pairs:
        raise SystemExit("--train-on-test needs LABELLED test data (archive/test/images + archive/test/masks). "
                         "None found, so there is nothing to add to training. Run without the flag.")

    if DATASET.exists():                                  # only now, after all inputs were validated
        log.warning("Deleting previously converted dataset %s and rebuilding it", DATASET)
        shutil.rmtree(DATASET)

    flags = []
    for _, m in train_pairs:
        b, _ = load_mask_oriented(m)
        flags.append(bool(b is not None and b.any()))
    train_set, val_set = stratified_split(train_pairs, flags, val_split)

    has_test = False
    if train_on_test:
        log.warning("--train-on-test: archive/test is added to TRAINING. There will be no untouched "
                    "test set, so judge the model by the metrics of your earlier run.")
        train_set = train_set + test_pairs
    elif test_pairs:
        has_test = True

    stats = {"train": convert_split(train_set, "train"), "val": convert_split(val_set, "val")}
    if has_test:
        stats["test"] = convert_split(test_pairs, "test")
    stats["_meta"] = dict(train_on_test_applied=bool(train_on_test), has_test_split=has_test,
                          val_split=val_split, archive=str(archive), seed=SEED,
                          built_at=datetime.now().isoformat(timespec="seconds"))

    yaml_path = DATASET / "data.yaml"
    yaml_path.write_text(
        f"path: {DATASET.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        + ("test: images/test\n" if has_test else "")
        + "names:\n  0: crack\n"
    )
    (DATASET / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return yaml_path, stats


# ----------------------------------------------------------------------------
# 3. PIXEL-LEVEL CRACK EVALUATION  (what actually matters for thin cracks)
# ----------------------------------------------------------------------------
def evaluate_pixelwise(model, split, thresholds, imgsz, device):
    """Run the model on every image of a split and score it against the true masks.

    Reports, per confidence threshold:
      strict IoU / Dice          - exact pixel overlap (harsh for 1-2px wide cracks)
      tolerant P / R / F1        - a predicted pixel counts if it is within TOL_PX of a real crack
      image-level P / R          - "does this image contain a crack?"
    """
    img_dir, mask_dir = DATASET / "images" / split, DATASET / "masks" / split
    files = list_images(img_dir)
    thresholds = sorted(thresholds)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL_PX + 1, 2 * TOL_PX + 1))
    keys = ("inter", "pred", "gt", "pred_ok", "gt_ok", "itp", "ifp", "ifn", "itn")
    acc = {t: dict.fromkeys(keys, 0) for t in thresholds}

    for img_path in files:
        gt = cv2.imread(str(mask_dir / f"{img_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt = gt > 0
        res = model.predict(source=str(img_path), imgsz=imgsz, conf=thresholds[0], iou=PREDICT_IOU,
                            retina_masks=True, device=device, verbose=False)[0]

        pm, confs = None, np.zeros(0)
        if res.masks is not None and len(res.masks):
            pm = res.masks.data.cpu().numpy() > 0.5
            confs = res.boxes.conf.cpu().numpy()
            if pm.shape[1:] != gt.shape:
                pm = np.stack([cv2.resize(m.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                          interpolation=cv2.INTER_NEAREST) for m in pm]).astype(bool)

        gt_dil = cv2.dilate(gt.astype(np.uint8), kernel).astype(bool)
        gt_has = bool(gt.any())

        for t in thresholds:
            sel = confs >= t
            pred = pm[sel].any(axis=0) if (pm is not None and sel.any()) else np.zeros_like(gt)
            a = acc[t]
            a["inter"] += int((pred & gt).sum())
            a["pred"] += int(pred.sum())
            a["gt"] += int(gt.sum())
            a["pred_ok"] += int((pred & gt_dil).sum())
            if pred.any():
                pred_dil = cv2.dilate(pred.astype(np.uint8), kernel).astype(bool)
                a["gt_ok"] += int((gt & pred_dil).sum())
            pred_has = bool(sel.any())
            a["itp"] += int(pred_has and gt_has)
            a["ifp"] += int(pred_has and not gt_has)
            a["ifn"] += int(gt_has and not pred_has)
            a["itn"] += int(not pred_has and not gt_has)

    eps = 1e-9
    rows = []
    for t in thresholds:
        a = acc[t]
        p_tol = a["pred_ok"] / (a["pred"] + eps)
        r_tol = a["gt_ok"] / (a["gt"] + eps)
        rows.append(dict(
            conf=t,
            iou=a["inter"] / (a["pred"] + a["gt"] - a["inter"] + eps),
            dice=2 * a["inter"] / (a["pred"] + a["gt"] + eps),
            precision_tol=p_tol, recall_tol=r_tol,
            f1_tol=2 * p_tol * r_tol / (p_tol + r_tol + eps),
            img_precision=a["itp"] / (a["itp"] + a["ifp"] + eps),
            img_recall=a["itp"] / (a["itp"] + a["ifn"] + eps),
            img_fp=a["ifp"], img_fn=a["ifn"],
        ))
    return rows


def log_metric_table(title, rows):
    log.info("%s", title)
    log.info("  conf |  IoU  | Dice  | P(tol) | R(tol) | F1(tol) | img-P | img-R | img FP/FN")
    for r in rows:
        log.info("  %.2f | %.3f | %.3f | %.3f  | %.3f  | %.3f   | %.3f | %.3f | %d/%d",
                 r["conf"], r["iou"], r["dice"], r["precision_tol"], r["recall_tol"], r["f1_tol"],
                 r["img_precision"], r["img_recall"], r["img_fp"], r["img_fn"])


# ----------------------------------------------------------------------------
# 4. TRAINING (with OOM protection)
# ----------------------------------------------------------------------------
def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError"


def build_train_args(args, yaml_path, run_name, device, batch):
    return dict(
        data=str(yaml_path), epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
        batch=batch, device=device, workers=args.workers, cache=args.cache, seed=SEED,
        pretrained=True, amp=True,
        # optimiser: explicit AdamW + cosine decay is stable for fine-tuning on small/medium datasets
        optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005, warmup_epochs=3, cos_lr=True,
        # NOTE: leave mask_ratio at its default (4). The seg loss runs at prototype resolution (imgsz/4);
        # mask_ratio=1 only makes the dataloader rasterise at full size and then nearest-downsample it.
        # augmentation: cracks are orientation-free, lighting varies a lot in the field
        fliplr=0.5, flipud=0.5, degrees=15, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, mosaic=1.0, close_mosaic=15, mixup=0.0, copy_paste=0.0,
        # outputs
        project=str(RUNS_DIR), name=run_name, exist_ok=True, plots=True, save_period=10, verbose=True,
    )


def train_with_fallback(YOLO, torch, args, yaml_path, run_name, device):
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    batches, b = [], args.batch
    while b >= 1:                                         # e.g. 8 -> 4 -> 2 -> 1 ; 7 -> 3 -> 1
        batches.append(b)
        b //= 2
    for bs in batches:
        shutil.rmtree(RUNS_DIR / run_name, ignore_errors=True)   # discard partial output of a failed attempt
        model, oom = None, False
        try:
            log.info("Starting training: model=%s imgsz=%d batch=%d epochs=%d",
                     args.model, args.imgsz, bs, args.epochs)
            model = YOLO(args.model)
            model.train(**build_train_args(args, yaml_path, run_name, device, bs))
            return model, bs
        except (RuntimeError, MemoryError) as e:
            if not is_oom(e):
                raise
            oom = True
        # cleanup happens here, after the except block: inside it the traceback still pins the GPU tensors
        if oom:
            log.warning("GPU out of memory at batch=%d -> retrying with a smaller batch", bs)
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise SystemExit("Out of GPU memory even at batch=1. Lower --imgsz (e.g. 512) and retry.")


def find_latest_last_pt():
    candidates = sorted(RUNS_DIR.glob(f"{RUN_PREFIX}_*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train YOLO11s-seg for crack segmentation")
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--model", default=MODEL_WEIGHTS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640, help="multiple of 32; 640 (default, fits 6 GB) / 800 / 1024")
    ap.add_argument("--batch", type=int, default=8, help="starting batch; halves automatically on OOM")
    ap.add_argument("--workers", type=int, default=2, help="2 is safe on Windows laptops; use 0 if multiprocessing errors appear")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--cache", default="False", choices=["False", "ram", "disk"])
    ap.add_argument("--resume", action="store_true", help="resume the most recent interrupted run")
    ap.add_argument("--skip-prepare", action="store_true", help="reuse the already converted dataset")
    ap.add_argument("--train-on-test", action="store_true",
                    help="final model only: add archive/test to the training data (no clean test set left)")
    ap.add_argument("--promote", action="store_true",
                    help="also copy this run's bundle to deploy/ (what predict_cracks.py reads by default)")
    ap.add_argument("--no-export", action="store_true", help="skip ONNX export")
    ap.add_argument("--allow-cpu", action="store_true", help="allow training without a GPU (very slow)")
    args = ap.parse_args()
    args.cache = False if args.cache == "False" else args.cache
    return args


def main():
    args = parse_args()
    log_file = setup_logging()

    import torch
    import ultralytics
    from ultralytics import YOLO

    log.info("Python %s | torch %s | ultralytics %s | opencv %s | %s",
             platform.python_version(), torch.__version__, ultralytics.__version__,
             cv2.__version__, platform.platform())

    if torch.cuda.is_available():
        device = 0
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s (%.1f GB VRAM)", props.name, props.total_memory / 1024 ** 3)
    elif args.allow_cpu:
        device = "cpu"
        log.warning("No GPU found - training on CPU will be very slow")
    else:
        raise SystemExit("No CUDA GPU detected. Install the CUDA build of PyTorch "
                         "(https://pytorch.org/get-started/locally/) or pass --allow-cpu.")

    # ---- dataset ----
    yaml_path = DATASET / "data.yaml"
    if args.skip_prepare or args.resume:
        if not yaml_path.exists():
            raise SystemExit("Converted dataset not found - run once without --skip-prepare/--resume.")
        log.info("Reusing converted dataset at %s", DATASET)
    else:
        yaml_path, _ = build_dataset(args.archive, args.val_split, args.train_on_test)
    ds_stats = json.loads((DATASET / "dataset_stats.json").read_text())
    ds_meta = ds_stats.get("_meta", {})
    test_in_training = bool(ds_meta.get("train_on_test_applied", False))     # what the DATASET contains, not the CLI
    has_test = (DATASET / "images" / "test").exists()
    if args.train_on_test and not test_in_training:
        raise SystemExit("--train-on-test was passed but the reused dataset was built WITHOUT test data. "
                         "Re-run without --skip-prepare/--resume so it is rebuilt.")
    if test_in_training and not args.train_on_test:
        log.warning("The reused dataset already contains test data in TRAINING (built with --train-on-test).")

    # ---- train ----
    run_name = f"{RUN_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}"
    if args.resume:
        last = find_latest_last_pt()
        if last is None:
            raise SystemExit("No previous run found to resume.")
        log.info("Resuming from %s", last)
        model = YOLO(str(last))
        model.train(resume=True)
        batch_used = model.trainer.args.batch
    else:
        model, batch_used = train_with_fallback(YOLO, torch, args, yaml_path, run_name, device)

    save_dir = Path(model.trainer.save_dir)
    args.imgsz = int(model.trainer.args.imgsz)            # a resumed run keeps ITS settings, not today's CLI defaults
    args.epochs = int(model.trainer.args.epochs)
    best_pt = save_dir / "weights" / "best.pt"
    weights_source = "best.pt"
    if not best_pt.exists():
        best_pt, weights_source = save_dir / "weights" / "last.pt", "last.pt"
        log.warning("best.pt not found - using last.pt")
    log.info("Training finished. Best weights: %s", best_pt)
    best = YOLO(str(best_pt))

    # ---- standard (COCO-style) metrics ----
    report = {}
    for split in ("val", "test") if has_test else ("val",):
        try:
            m = best.val(data=str(yaml_path), split=split, imgsz=args.imgsz, batch=max(1, batch_used),
                         conf=0.001, iou=0.6, device=device, plots=True, verbose=False,
                         project=str(save_dir), name=f"eval_{split}", exist_ok=True)
            report[f"map_{split}"] = dict(mask_map50=float(m.seg.map50), mask_map50_95=float(m.seg.map),
                                          mask_precision=float(m.seg.mp), mask_recall=float(m.seg.mr),
                                          box_map50=float(m.box.map50), box_map50_95=float(m.box.map))
            log.info("[%s] mask mAP50=%.3f  mAP50-95=%.3f  P=%.3f  R=%.3f", split,
                     m.seg.map50, m.seg.map, m.seg.mp, m.seg.mr)
        except Exception as e:                                       # never lose the model over a metric
            log.warning("Ultralytics validation on '%s' failed: %s", split, e)

    # ---- pixel-level metrics + threshold selection (on VAL only) ----
    val_rows = evaluate_pixelwise(best, "val", CONF_GRID, args.imgsz, device)
    log_metric_table("Pixel-level crack metrics on VALIDATION set", val_rows)
    best_row = max(val_rows, key=lambda r: r["f1_tol"])
    best_conf = best_row["conf"]
    log.info("Selected confidence threshold (max tolerant F1 on val): %.2f", best_conf)
    if best_conf in (min(CONF_GRID), max(CONF_GRID)):
        log.warning("Best threshold sits at the edge of the search grid - the true optimum may be outside it")
    log.warning("val was also used to pick best.pt and this threshold, so val numbers are optimistic. "
                "Only a labelled, untouched test set gives an honest estimate.")
    report["val_pixel"] = best_row

    if has_test:
        test_rows = evaluate_pixelwise(best, "test", [best_conf], args.imgsz, device)
        log_metric_table("Pixel-level crack metrics on held-out TEST set (threshold fixed from val)", test_rows)
        report["test_pixel"] = test_rows[0]
    else:
        log.warning("No labelled test split -> no test metrics. (An images-only archive/test folder cannot be scored.)")

    # ---- sample predictions for visual inspection ----
    samples = list_images(DATASET / "images" / ("test" if has_test else "val"))
    random.Random(SEED).shuffle(samples)
    if samples:
        best.predict(source=[str(x) for x in samples[:12]], imgsz=args.imgsz, conf=best_conf,
                     iou=PREDICT_IOU, retina_masks=True, boxes=False, device=device, save=True,
                     project=str(save_dir), name="sample_predictions", exist_ok=True, verbose=False)

    # ---- qualitative predictions on an UNLABELLED test folder (no metrics possible) ----
    raw_test = args.archive / "test"
    raw_imgs = (list_images(raw_test / "images") if (raw_test / "images").is_dir()
                else list_images(raw_test) if raw_test.is_dir() else [])
    if raw_imgs and not has_test and not test_in_training:
        log.info("Predicting on %d unlabelled test images (visual check only)", len(raw_imgs))
        for _ in best.predict(source=[str(x) for x in raw_imgs], imgsz=args.imgsz, conf=best_conf,
                              iou=PREDICT_IOU, retina_masks=True, boxes=False, device=device, save=True,
                              stream=True, project=str(save_dir), name="unlabelled_test_predictions",
                              exist_ok=True, verbose=False):
            pass

    # ---- deployment bundle: one folder per run, nothing is overwritten ----
    bundle = DEPLOY_DIR / save_dir.name
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, bundle / "crack_yolo11s_seg.pt")

    onnx_name = None
    if not args.no_export:
        try:
            onnx_path = best.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False)
            shutil.copy2(onnx_path, bundle / "crack_yolo11s_seg.onnx")
            onnx_name = "crack_yolo11s_seg.onnx"
            log.info("ONNX model exported to %s", bundle / onnx_name)
        except Exception as e:
            log.warning("ONNX export FAILED (bundle has no ONNX file): %s", e)

    config = dict(
        model="yolo11s-seg", weights="crack_yolo11s_seg.pt", weights_source=weights_source, onnx=onnx_name,
        classes=["crack"], imgsz=args.imgsz, conf=best_conf, iou=PREDICT_IOU, retina_masks=True,
        trained_on_test_set=test_in_training,            # taken from the dataset that was actually used
        dataset=ds_stats, run=save_dir.name,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        epochs_requested=args.epochs, batch_used=int(batch_used),
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      ultralytics=ultralytics.__version__, opencv=cv2.__version__),
        metrics=report,
        notes="val metrics are optimistic (val chose best.pt and conf); test metrics exist only if a labelled test split was used",
    )
    (bundle / "deployment_config.json").write_text(json.dumps(config, indent=2))
    (save_dir / "run_config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    if args.promote:
        DEPLOY_DIR.mkdir(exist_ok=True)
        for f in bundle.iterdir():
            shutil.copy2(f, DEPLOY_DIR / f.name)
        log.info("Promoted %s to %s (predict_cracks.py default)", save_dir.name, DEPLOY_DIR)

    log.info("DONE. Bundle: %s  (use --promote to make it the default for predict_cracks.py)", bundle)
    log.info("Look at %s to visually check the predictions, and the log at %s",
             save_dir / "sample_predictions", log_file)


if __name__ == "__main__":          # required on Windows because the dataloader uses worker processes
    main()
