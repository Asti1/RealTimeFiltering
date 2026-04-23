# Neural Style Transfer

**CS 5330 Pattern Recognition and Computer Vision — Final Project**
Astitva Goel | Northeastern University | Spring 2026

---

## Overview

This project implements and compares two foundational approaches to Neural Style Transfer (NST) — the task of rendering a photograph in the visual style of a painting while preserving its semantic content.

**Method 1 — Gatys et al. (2016):** Optimization-based NST. Iteratively updates a generated image by minimizing a combined content and style loss computed from VGG-19 feature activations. Produces high-quality results but requires ~10 minutes per image on CPU.

**Method 2 — Johnson et al. (2016):** Feed-forward NST. Trains a lightweight encoder-decoder network (TransformNet) to stylize images in a single forward pass. After training, inference takes ~4ms — approximately 150,000× faster than Gatys.

---

## Demo Video

[Click here to watch the demo](https://drive.google.com/file/d/1Tz_68DoHRT6ltPSaNcBvLdlVoPh6T4f3/view?usp=sharing)

---

## Results

| Method               | Training data       | Inference time | Flexible?       |
| -------------------- | ------------------- | -------------- | --------------- |
| Gatys et al.         | None (optimization) | ~600s / image  | Yes — any style |
| Johnson (ours)       | 5k COCO, 10 epochs  | 3.9ms / image  | No — one style  |
| Johnson (pretrained) | 118k COCO, 2 epochs | 4.7ms / image  | No — one style  |

---

## Environment

**Operating System:** macOS (Apple M-series)
**IDE:** Visual Studio Code
**Language:** Python 3.12
**Key Libraries:** PyTorch, torchvision, Pillow, matplotlib, numpy

---

## Setup

```bash
# Clone or download the project
cd Project6

# Create and activate virtual environment
python -m venv nst_env
source nst_env/bin/activate        # Mac/Linux
# nst_env\Scripts\activate         # Windows

# Install dependencies
pip install torch torchvision pillow matplotlib numpy
```

---

## Running the Code

### Method 1 — Gatys et al. (single run)

Place your content and style images in `images/` then:

```bash
python neural_style_transfer.py
```

Outputs saved to `outputs/`. Edit these values at the top of the file to configure:

```python
CONTENT_IMG = "images/content.jpg"
STYLE_IMG   = "images/style.jpg"
NUM_STEPS   = 300     # optimization steps
ALPHA       = 1       # content weight
BETA        = 1e6     # style weight (Gatys default)
```

### Method 1 — α/β Experiment (4 runs, comparison grid)

```bash
python experiment_runner.py
```

Runs Gatys with β = 1e4, 1e5, 1e6, 1e7 and saves a 2×2 comparison grid to `outputs/experiments/comparison_grid.png`.

### Method 2 — Johnson et al. (local training)

First, download COCO val2017 training images:

```bash
mkdir -p images/train/coco
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mv val2017/* images/train/coco/
```

Then train and run inference:

```bash
python fast_style_transfer.py
```

Trained model saved to `outputs/transform_net.pth`. Inference output saved to `outputs/fast_output.jpg`.

### Method 2 — Johnson et al. (Google Colab, recommended)

Use `nst.py` on Google Colab with a T4 GPU for faster training. The notebook:

- Downloads COCO val2017 automatically
- Trains TransformNet for 20 epochs
- Loads 4 pretrained Johnson models (candy, mosaic, rain_princess, udnie)
- Generates and downloads all comparison figures

Runtime: ~2 hours on T4 GPU.

### Gatys vs Johnson Comparison Figure

```bash
python compare_methods.py
```

Runs Gatys (300 steps) and loads the saved Johnson model, times both, and saves a side-by-side comparison to `outputs/gatys_vs_johnson.png`.

---

## Implementation Notes

- VGG-19 weights are downloaded automatically from torchvision on first run (~550MB, cached after)
- Gatys optimization uses L-BFGS (not Adam) — standard for NST, converges faster
- Johnson TransformNet uses Instance Normalization (not Batch Norm) — better for per-image style
- All convolutions use reflect-padding to reduce border artifacts
- The `* 0.5` scaling on TransformNet output was removed to allow full stylization range
- Style loss uses batched Gram matrices (`torch.bmm`) during Johnson training

---

## References

1. L. A. Gatys, A. S. Ecker, and M. Bethge, "A Neural Algorithm of Artistic Style," CVPR 2016.
2. J. Johnson, A. Alahi, and L. Fei-Fei, "Perceptual Losses for Real-Time Style Transfer and Super-Resolution," ECCV 2016.
3. X. Huang and S. Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization," ICCV 2017.
4. K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," ICLR 2015.

---

## Time Travel Days

Not used.
