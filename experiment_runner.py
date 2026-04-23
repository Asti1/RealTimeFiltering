# Astitva Goel
"""
Experiment runner for Neural Style Transfer
Loops through different beta values and saves each result + a comparison grid.
"""

# ── Imports (same as main file) ───────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np

# ── Copy over the shared setup from neural_style_transfer.py ─────────────────
# (device, image utils, VGGFeatures, loss functions, run_style_transfer)
# To avoid duplicating everything, we just import from your main file directly
import sys
sys.path.insert(0, os.path.dirname(__file__))
from neural_style_transfer import (
    device, load_image, tensor_to_pil, show_image,
    VGGFeatures, gram_matrix, content_loss, style_loss,
    total_loss, run_style_transfer
)

# ── Config ────────────────────────────────────────────────────────────────────
CONTENT_IMG  = "images/content.jpg"   # same images as your main file
STYLE_IMG    = "images/style.jpg"
OUTPUT_DIR   = "outputs/experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_STEPS    = 300   # same for all runs so results are comparable

# ── Experiment definitions ────────────────────────────────────────────────────
# Each entry: (label for filename, alpha, beta)
EXPERIMENTS = [
    ("beta_1e4",  1, 1e4),    # content dominates — faint style tint
    ("beta_1e5",  1, 1e5),    # balanced — mild style
    ("beta_1e6",  1, 1e6),    # Gatys default — strong style (your current result)
    ("beta_1e7",  1, 1e7),    # style dominates — city barely visible
]

# ── Run all experiments ───────────────────────────────────────────────────────
def run_experiments():
    # Load images once — reused across all runs
    content = load_image(CONTENT_IMG)
    style   = load_image(STYLE_IMG)
    results = []   # store (label, PIL image) for the comparison grid

    for label, alpha, beta in EXPERIMENTS:
        print(f"\n{'='*50}")
        print(f"Running: {label}  (alpha={alpha}, beta={beta:.0e})")
        print(f"{'='*50}")

        output = run_style_transfer(
            content, style,
            num_steps=NUM_STEPS,
            alpha=alpha,
            beta=beta,
            show_every=100
        )

        # Save individual result
        pil = tensor_to_pil(output)
        path = os.path.join(OUTPUT_DIR, f"{label}.jpg")
        pil.save(path)
        print(f"Saved: {path}")

        results.append((label, pil))

    return results

# ── Save a 2x2 comparison grid ────────────────────────────────────────────────
def save_comparison_grid(results):
    """
    Creates a 2x2 figure with all four results side by side.
    This is ready to drop straight into your report.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    titles = {
        "beta_1e4": "α/β = 1/10,000\n(content dominates)",
        "beta_1e5": "α/β = 1/100,000\n(mild style)",
        "beta_1e6": "α/β = 1/1,000,000\n(Gatys default)",
        "beta_1e7": "α/β = 1/10,000,000\n(style dominates)",
    }

    for ax, (label, pil) in zip(axes, results):
        ax.imshow(np.array(pil))
        ax.set_title(titles.get(label, label), fontsize=11)
        ax.axis("off")

    plt.suptitle("Effect of α/β ratio on Neural Style Transfer", fontsize=13, y=1.01)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "comparison_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison grid saved: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Running {len(EXPERIMENTS)} experiments x {NUM_STEPS} steps each")
    print(f"Device: {device}")
    print(f"Results will be saved to: {OUTPUT_DIR}/\n")

    results = run_experiments()
    save_comparison_grid(results)

    print("\nAll experiments done.")
    print(f"Open {OUTPUT_DIR}/ to see individual results.")
    print(f"comparison_grid.png is ready for your report.")