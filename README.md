# Pneumonia Detection from Chest X-Rays

Binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA** using PyTorch CNNs and transfer learning.

## Dataset

Uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset by Kermany, Zhang & Goldbaum (2018), published on [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj) and licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The **test split** (624 images) is bundled in this repo under `data/test/` so the demo app runs without external downloads.

For full training you also need the `train/` and `val/` splits. Place the extracted data so the directory looks like:

```
Pneumonia_project/
├── chest_xray/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
└── pneumonia_detection/
    └── (this project)
```

## Setup

```bash
cd pneumonia_detection
pip install -r requirements.txt
```

## Experiments

There are 6 experiments in total:

| # | Config | Architecture | Weighted Sampling | Augmentation | Epochs | LR |
|---|--------|-------------|-------------------|--------------|--------|----|
| 1 | `configs/baseline.yaml` | 3-layer CNN | No | No | 8 | 0.001 |
| 2 | `configs/weighted_sampling.yaml` | 3-layer CNN | Yes | No | 8 | 0.001 |
| 3 | `configs/augmentation.yaml` | 3-layer CNN | Yes | Yes | 15 | 0.001 |
| 4 | `configs/deeper_cnn.yaml` | 5-layer CNN + Dropout | Yes | Yes | 8 | 0.001 |
| 5 | `configs/resnet18.yaml` | Pretrained ResNet-18 | Yes | Yes | 5 | 0.0001 |
| 6 | `configs/distillation_best.yaml` | DeeperCNN distilled from ResNet-18 | Yes | Yes | 5 | 0.001 |

Experiments 1–5 use `train.py`. Experiment 6 (knowledge distillation) uses `distill_best.py` — see the [Knowledge Distillation](#knowledge-distillation) section below for details.

## Training

Train experiments 1–5 (standard training):

```bash
python train.py --config configs/baseline.yaml
python train.py --config configs/weighted_sampling.yaml
python train.py --config configs/augmentation.yaml
python train.py --config configs/deeper_cnn.yaml
python train.py --config configs/resnet18.yaml
```

Train experiment 6 (knowledge distillation):

```bash
python distill_best.py --config configs/distillation_best.yaml
```

Trained models are saved to the `checkpoints/` directory.

## Evaluation

Evaluate a trained model on the test set:

```bash
python evaluate.py --config configs/baseline.yaml
```

Evaluate all experiments:

```bash
python evaluate.py --config configs/baseline.yaml
python evaluate.py --config configs/weighted_sampling.yaml
python evaluate.py --config configs/augmentation.yaml
python evaluate.py --config configs/deeper_cnn.yaml
python evaluate.py --config configs/resnet18.yaml
```

To evaluate with a specific checkpoint file:

```bash
python evaluate.py --config configs/resnet18.yaml --checkpoint path/to/model.pth
```

Evaluation prints confusion matrix, classification report, and threshold analysis for each config's defined thresholds.

## Project Structure

```
pneumonia_detection/
├── configs/                       # Experiment configs
│   ├── baseline.yaml
│   ├── weighted_sampling.yaml
│   ├── augmentation.yaml
│   ├── deeper_cnn.yaml
│   ├── resnet18.yaml
│   ├── distillation_best.yaml     # Best distillation config (alpha=0.5)
│   └── distillation_sweep.yaml    # Alpha sweep config (0.7, 0.5, 0.3)
├── src/                           # Source modules
│   ├── dataset.py                 # Data loading, transforms, weighted sampler, dual-transform dataset
│   ├── models.py                  # CNN, DeeperCNN, ResNet-18 architectures
│   ├── train.py                   # Standard training and validation loops
│   ├── distill.py                 # Distillation training logic (shared by distill_best and distill_sweep)
│   ├── evaluate.py                # Inference, metrics, threshold analysis
│   └── utils.py                   # Config loading, device selection, seeding
├── checkpoints/                   # Saved model weights (gitignored)
├── train.py                       # Entry point: standard training
├── evaluate.py                    # Entry point: evaluate standard models
├── distill_best.py                # Entry point: train best distilled model (alpha=0.5)
├── distill_sweep.py               # Entry point: alpha comparison sweep
├── evaluate_distilled.py          # Entry point: evaluate distilled model without retraining
├── plot_predictions.py            # Entry point: visualise X-ray predictions as a grid
├── plot_metrics.py                # Entry point: generate comparison plots (ROC, confusion matrix, bar chart)
├── compare_models.py              # Entry point: side-by-side comparison of all 3 key models (predictions + metrics)
├── save_with_metadata.py          # One-time: embed threshold metadata into existing checkpoints
├── plots/                         # Saved plots (gitignored)
├── requirements.txt
└── .gitignore
```

## Key Models

This project centres on three models that form the distillation pipeline. Everything else (baseline, weighted sampling, augmentation experiments) exists to build intuition before arriving at these three.

### DeeperCNN — Standalone Student

A 5-layer custom CNN trained entirely from scratch on grayscale chest X-rays.

| Property | Value |
|---|---|
| Architecture | 5 Conv layers + Dropout (0.3, 0.4) + GAP |
| Input | 1-channel grayscale, 224×224 |
| Parameters | ~1M |
| Training | 8 epochs, lr=0.001, weighted sampling, augmentation |
| Best threshold | **0.9** |
| Test accuracy | **87%** |
| Macro F1 | **0.86** |

```bash
python train.py --config configs/deeper_cnn.yaml
python evaluate.py --config configs/deeper_cnn.yaml
```

---

### ResNet-18 — Teacher

A pretrained ResNet-18 fine-tuned on 3-channel chest X-rays with ImageNet normalisation. This is the teacher whose soft predictions guide the student during distillation.

| Property | Value |
|---|---|
| Architecture | Pretrained ResNet-18, custom FC head |
| Input | 3-channel, 224×224, ImageNet normalised |
| Parameters | ~11M |
| Training | 5 epochs, lr=0.0001, weighted sampling, augmentation |
| Best threshold | **0.95** |
| Test accuracy | **92%** |
| Macro F1 | **0.91** |

```bash
python train.py --config configs/resnet18.yaml
python evaluate.py --config configs/resnet18.yaml
```

---

### Distilled DeeperCNN — Student

The same DeeperCNN architecture as above, but trained using knowledge distillation from the frozen ResNet-18 teacher instead of hard labels alone. It learns to mimic the teacher's soft probability outputs, which carries more signal than binary 0/1 labels.

| Property | Value |
|---|---|
| Architecture | Same DeeperCNN as standalone (~1M params) |
| Input | 1-channel grayscale, 224×224 |
| Parameters | ~1M |
| Training | 5 epochs, lr=0.001, alpha=0.5, weighted sampling, augmentation |
| Best threshold | **0.85** |
| Test accuracy | **88%** |
| Macro F1 | **0.87** |

```bash
python distill_best.py --config configs/distillation_best.yaml
python evaluate_distilled.py --config configs/distillation_best.yaml
```

---

### Three-Model Comparison

| Model | Params | Best Threshold | Accuracy | Macro F1 |
|---|---|---|---|---|
| DeeperCNN (standalone) | ~1M | 0.9 | 87% | 0.86 |
| **Distilled DeeperCNN** | **~1M** | **0.85** | **88%** | **0.87** |
| ResNet-18 (teacher) | ~11M | 0.95 | 92% | 0.91 |

The distilled student has the same parameter count as the standalone DeeperCNN but closes the accuracy gap from 87% → 88% by learning from the teacher's soft predictions. The teacher still leads at 92%, but at 11× the parameter cost.

#### Metrics

![Model Metrics Comparison](assets/comparison_metrics.png)

#### Sample Predictions

Each row is one model evaluated at its best threshold. Green border = correct, red = incorrect.

![Chest X-Ray Predictions](assets/comparison_predictions.png)

To embed the best threshold into each checkpoint file (run once after training):
```bash
python save_with_metadata.py
```

To regenerate the comparison plots:
```bash
python compare_models.py
```

---

## Visualisation

`plot_predictions.py` samples chest X-ray images from the test set, runs inference, and saves a grid showing the true label vs. the model's predicted label. Correct predictions get a green border; incorrect ones get red.

```bash
# visualise predictions for any config (saves to plots/predictions_<config>.png)
python plot_predictions.py --config configs/resnet18.yaml
python plot_predictions.py --config configs/deeper_cnn.yaml
python plot_predictions.py --config configs/distillation_best.yaml

# override threshold or checkpoint
python plot_predictions.py --config configs/resnet18.yaml --threshold 0.7
python plot_predictions.py --config configs/resnet18.yaml --checkpoint path/to/model.pth

# control number of images shown (default 16, must be a multiple of 4)
python plot_predictions.py --config configs/resnet18.yaml --num_images 20
```

The script always shows a balanced mix — half correct predictions and half incorrect — so both success and failure cases are visible at a glance.

## Knowledge Distillation

The distilled student (DeeperCNN, ~1M params) learns from the frozen ResNet-18 teacher (~11M params) using a combined loss:

```
Loss = alpha * BCE(student_logit, sigmoid(teacher_logit))   ← soft label loss
     + (1 - alpha) * BCE(student_logit, hard_label)         ← ground truth loss
```

The teacher is frozen throughout — only the student's weights are updated. Both teacher and student receive the same augmented image but through different transforms (3-channel ImageNet-normalized for the teacher, 1-channel grayscale for the student). A weighted sampler handles class imbalance during training. Best student checkpoint is saved based on validation accuracy.

### Scripts

| Script | Config | Purpose |
|---|---|---|
| `distill_best.py` | `configs/distillation_best.yaml` | Train final best model (alpha=0.5) from scratch |
| `distill_sweep.py` | `configs/distillation_sweep.yaml` | Compare alpha=0.7, 0.5, 0.3 — useful for showing experimentation |
| `evaluate_distilled.py` | `configs/distillation_best.yaml` | Evaluate already-trained distilled model, no retraining needed |

### Training

**Train the best known configuration (alpha=0.5, threshold=0.85):**

```bash
python distill_best.py --config configs/distillation_best.yaml
```

Trains from scratch, saves best model by validation accuracy, evaluates across all thresholds, and saves a final checkpoint with embedded metadata (alpha, threshold, class names).

**Run the alpha comparison sweep (0.7, 0.5, 0.3):**

```bash
python distill_sweep.py --config configs/distillation_sweep.yaml
```

Trains three separate student models from scratch (one per alpha), evaluates each across all thresholds, and prints a summary table at the end. Use this to compare alphas and justify the choice of alpha=0.5.

### Evaluation (no retraining)

If the distilled model is already trained (checkpoint exists in `checkpoints/`), evaluate it directly without retraining:

```bash
python evaluate_distilled.py --config configs/distillation_best.yaml
```

This script handles both plain state dict checkpoints and metadata-rich checkpoints (saved by `distill_best.py` or from notebook experiments). It will print the alpha, threshold, and class info embedded in the checkpoint before running evaluation.

To evaluate with a specific checkpoint:

```bash
python evaluate_distilled.py --config configs/distillation_best.yaml --checkpoint path/to/model.pth
```

### Results (best config — alpha=0.5, threshold=0.85)

| Model | Params | Best Threshold | Accuracy | Macro F1 |
|---|---|---|---|---|
| Standalone DeeperCNN | ~1M | 0.9 | 87% | 0.86 |
| **Distilled DeeperCNN** | **~1M** | **0.85** | **88%** | **0.87** |
| Teacher ResNet-18 | ~11M | 0.95 | 92% | 0.91 |

The distilled student matches the standalone DeeperCNN's parameter footprint (~1M) while improving accuracy from 87% → 88% by learning from the teacher's soft predictions.

## Adding a New Experiment

1. Create a new YAML file in `configs/` (copy an existing one as a starting point).
2. Adjust the parameters (model, lr, epochs, augmentation, etc.).
3. Run `python train.py --config configs/your_new_config.yaml`.

No code changes needed unless you're adding a new model architecture (in that case, add it to `src/models.py` and register it in `get_model()`).
