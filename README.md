# Kazim — Person + Head Detection

My contribution to the group object-detection project. Everything here is additive: no existing
file in the repo was modified.

```
Kazim/
├── README.md
├── notebooks/
│   └── Person_Head_Detection_Kazim.ipynb   # main deliverable — trains AND tests all 3 tracks
├── src/
│   └── metrics.py                          # shared mAP implementation (has its own self-tests)
└── outputs/                                # created at runtime (weights, plots, results.json)
```

A copy of the notebook is also placed at `notebooks/Person_Head_Detection_Kazim.ipynb` in the repo
root, as requested.

---

## The problem I set out to fix

The repo already trains three detectors, in `notebooks/Object_Detection_3_Tracks.ipynb`:

| Track | Model | How it was scored |
|---|---|---|
| A | YOLO26 / Ultralytics, fine-tuned | Ultralytics' internal validator → mAP50, mAP50-95 |
| B | RF-DETR Nano, fine-tuned | `supervision.metrics.MeanAveragePrecision`, capped at 200 images |
| C | `ScratchDetector` CNN, from scratch | precision / recall / F1 at **one fixed** confidence threshold |

Three tracks, three different scoring tools, and one of them isn't even computing mAP.
`src/track_c_scratch/evaluate.py` is candid about this in its own docstring — it calls its metric
"simplified" and notes that Tracks A/B use different evaluation.

**That means the cross-track comparison in the report was not measuring the same thing.** A
precision/recall pair at conf=0.3 and a COCO mAP50-95 are not comparable numbers; you cannot put
them in one table and rank them. Track B's 200-image cap also meant it was scored on a different
subset than Track A.

## What I did about it

**1. One shared metric — `Kazim/src/metrics.py`.**
A from-scratch VOC/COCO-style mAP (all-point interpolation, IoU sweep 0.50:0.05:0.95) that takes
plain numpy boxes, so any model can be scored by it. It handles the cases that quietly corrupt
naive implementations:

- a ground-truth box can be matched **once**; further detections on it become false positives, so
  duplicate-box spam can't inflate recall
- a class absent from the split returns `NaN` and is **excluded** from the mean rather than scored
  as 0, which would otherwise silently halve the mAP
- detections are ranked globally across all images by confidence, not per-image

It ships with self-checks — run `python Kazim/src/metrics.py`:

```
perfect      -> mAP50=1.0000  mAP50-95=1.0000
duplicate box -> Head AP50=0.8333 (penalised, correct)
wrong class  -> mAP50=0.0000
absent class -> mAP50=1.0000 (Person excluded, not zeroed)
All metric self-checks passed.
```

The 0.8333 is hand-checkable: with 2 GT boxes and a duplicate ranked between them, AP =
0.5×1.0 + 0.5×(2/3).

**2. Every track scored through it, on the same images.** Track A's Ultralytics result is also
printed as a cross-check — if the two diverge badly, boxes are being fed in the wrong coordinate
space, and the notebook surfaces that instead of hiding it.

**3. Dataset EDA first.** Class balance, objects-per-image, and box-area distributions — which turn
"Track C is worse" into a measured explanation (see below).

**4. Qualitative comparison.** Same test images, ground truth and every track side by side.

**5. Results saved** to `outputs/results.json` + `results.csv` so the report can cite them directly.

---

## Running it

### Dataset

The [v2 YOLOv8 export](https://universe.roboflow.com/person-xf2dz/person-detection-with-head-8aw41/dataset/2)
from Roboflow Universe. Classes are `["Head", "Person"]` (Head is class 0).

Roboflow requires an account to export, so the download is manual: on the dataset page choose
**Download Dataset → YOLOv8 → "download zip to computer"**. Unzip it in the repo root as
`Person Detection with head.v2i.yolov8/` — that's the path `src/common/dataset_paths.py` already
expects, so both my notebook and the existing scripts find it.

This is already done in this working copy (27,964 files, 1.34 GB, `data.yaml` reporting
`nc: 2`, `names: ['Head', 'Person']`, version 2). The folder is covered by `.gitignore`, so it stays
out of version control — anyone cloning fresh has to repeat the download above.

The notebook auto-locates the dataset on Colab, Kaggle, and locally; if it can't, it fails with a
message telling you exactly what to set.

### Compute

Designed for a free Colab/Kaggle T4:

- **Colab:** `Runtime > Change runtime type > T4 GPU`, upload the zip to Drive, edit `DATASET_ZIP`.
- **Kaggle:** `Settings > Accelerator > GPU T4 x2` **and `Internet: On`** (needed for pip + pretrained checkpoints).

Set `QUICK_MODE = True` (the default) for a few-minute end-to-end smoke test. Set it to `False` for
the real numbers — that's ~50 epochs of YOLO on 12k images, so budget a few GPU-hours.

The notebook also selects CUDA → MPS → CPU automatically, so it will open and run locally on an
Apple Silicon Mac, but Track B (RF-DETR) needs CUDA and is wrapped in a `try/except` that skips it
with a clear message rather than killing the run.

---

## What the data actually looks like

Measured on the real v2 export (the notebook's EDA cells reproduce all of this):

| split | images | objects | Head | Person | objects/image |
|---|---|---|---|---|---|
| train | 12,231 | 115,279 | 54,310 | 60,969 | 9.43 |
| valid | 899 | 8,936 | 4,145 | 4,791 | 9.94 |
| test | 846 | 7,953 | 3,812 | 4,141 | 9.40 |

Every image has a matching label file, and the two classes are close to balanced — so any large
Head-vs-Person AP gap is a **difficulty** effect, not a class-imbalance artifact. The difficulty is
size:

| class | median box area | boxes under 1% of image area |
|---|---|---|
| Head | 0.264% | **92.6%** |
| Person | 2.635% | 25.1% |

Heads are roughly 10× smaller than persons, and nine in ten are tiny. Expect `Head` AP to trail
`Person` AP for every track.

## Why Track C loses — measured, not assumed

**1. One box per grid cell — a hard recall ceiling.** This is usually asserted; the notebook measures
it. At ~9.4 objects per image, boxes collide in cells constantly, and a collided box is dropped from
the training target entirely — it never reaches the loss, so the model cannot learn it:

| grid | boxes kept | boxes dropped | share lost | recall ceiling |
|---|---|---|---|---|
| **8×8** (what Track C uses) | 81,405 | 33,874 | **29.4%** | **70.6%** |
| 16×16 | 103,062 | 12,217 | 10.6% | 89.4% |
| 32×32 | 111,817 | 3,462 | 3.0% | 97.0% |

**Nearly a third of all training boxes are invisible to Track C's loss function.** Its recall cannot
exceed ~70.6% regardless of epochs, architecture width, or learning rate. This is the single most
useful sentence for the report: it explains Track C's result as a design consequence rather than a
training failure, and it shows the fix is a denser grid, not more compute.

**2. 128×128 input.** With 92.6% of Head boxes under 1% of image area, a head occupies only a few
pixels at 128px. Tracks A and B see 640px — 25× the pixel area.

**3. No pretraining.** A and B start from COCO checkpoints that already encode general object
features; C starts from random weights.

Ranked by expected payoff: **denser grid / multiple anchors per cell** (recovers ~19 points of recall
ceiling by itself) → raise input resolution to 416px → multi-scale FPN-style heads → augmentation.

---

## Notes / caveats

- **The `demo/app.py` Streamlit app reads different paths than this notebook writes.** It expects
  `src/track_a_yolo26/runs/yolo26_finetune/weights/best.pt` etc., while the notebook writes to
  `Kazim/outputs/track_*/`. Copy the weights across, or edit the three `TRACK_*_WEIGHTS` constants
  at the top of `demo/app.py`. The app degrades gracefully (per-track "weights not found" warning)
  rather than crashing, so you can run it before every track is trained:
  ```
  pip install -r requirements.txt
  streamlit run demo/app.py
  ```
  It needs torch + ultralytics + streamlit installed locally, which is a separate ~2.5 GB install
  from the Colab/Kaggle training path above.
- **`yolo26n.pt` requires a recent Ultralytics.** The notebook tries `yolo26n → yolo11n → yolov8n`
  and reports which one it got, so it runs regardless of the installed version. The dataset is a
  YOLOv8 export, so `yolov8n` is a perfectly valid fallback.
- **`QUICK_MODE` numbers are not reportable.** They exist to prove the pipeline runs. Anything
  quoted in the report should come from a `QUICK_MODE = False` run, and the flag is recorded in
  `results.json` so you can tell them apart later.
