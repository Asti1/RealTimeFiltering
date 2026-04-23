# Astitva Goel
"""
Fast Neural Style Transfer — Johnson et al. (2016)
Trains a lightweight transform network to stylize images in one forward pass.
After training, inference is instant (no optimization loop).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from neural_style_transfer import (
    device, load_image, tensor_to_pil, show_image,
    VGGFeatures, total_loss
)

# ── Config ────────────────────────────────────────────────────────────────────
STYLE_IMG       = "images/style.jpg"       # the one style to train on
CONTENT_IMG     = "images/content.jpg"     # used only for a test at the end
TRAIN_IMG_DIR   = "images/train"           # folder of training images (COCO subset)
MODEL_SAVE_PATH = "outputs/transform_net.pth"
OUTPUT_DIR      = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE    = 256
BATCH_SIZE  = 4
EPOCHS      = 2       # 2 epochs over a small dataset is enough to see results
LR          = 1e-3
ALPHA       = 1       # content weight
BETA        = 1e5     # style weight (lower than Gatys — network learns faster)

# ── Transform Network ─────────────────────────────────────────────────────────
# Encoder → residual blocks → decoder
# Residual blocks let gradients flow cleanly through deep layers
# Instance normalization instead of batch norm — works better per-image

class ResidualBlock(nn.Module):
    """Two conv layers with a skip connection — preserves content structure."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        # Skip connection: add input directly to output
        # This means the block only needs to learn the residual (difference)
        return x + self.block(x)

class TransformNet(nn.Module):
    """
    Encoder: downsample input image into compact feature representation
    Residual blocks: transform style while preserving content structure
    Decoder: upsample back to full image size
    """
    def __init__(self):
        super().__init__()

        # Encoder — 3 conv layers, each doubles channels and halves spatial size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9, stride=1, padding=4, padding_mode="reflect"),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Residual blocks — style transformation happens here
        # 5 blocks is standard in Johnson et al.
        self.residuals = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])

        # Decoder — upsample back to original size using fractional stride conv
        # Upsample + conv avoids checkerboard artifacts from transposed conv
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, 3, stride=1, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            # Final layer: output 3 channels (RGB), tanh squashes to [-1, 1]
            nn.Conv2d(32, 3, 9, stride=1, padding=4, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        x = self.decoder(x)
        # Scale tanh output [-1,1] to roughly match normalized image range
        return x * 0.5

# ── Training dataset ──────────────────────────────────────────────────────────
# We need a folder of diverse content images to train on.
# Recommended: download ~1000 images from COCO val2017 into images/train/any_subfolder/
# COCO download: wget http://images.cocodataset.org/zips/val2017.zip
# Then: mkdir -p images/train/coco && mv val2017/* images/train/coco/

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Pre-compute style Gram matrices ──────────────────────────────────────────
# We only need to do this ONCE before training — style image never changes
def get_style_grams(vgg_model, style_img_path):
    """
    Load style image and compute its Gram matrices at all style layers.
    These are fixed targets the transform network trains toward.
    """
    style_tensor = load_image(style_img_path)
    with torch.no_grad():
        style_feats, _ = vgg_model(style_tensor)
    # Compute and store Gram matrix for each style layer
    style_grams = {}
    from neural_style_transfer import gram_matrix
    for layer_idx, feat in style_feats.items():
        style_grams[layer_idx] = gram_matrix(feat).detach()
    return style_grams

# ── Training loop ─────────────────────────────────────────────────────────────
def train(style_img_path, train_dir, epochs=EPOCHS):
    print("Loading dataset...")
    dataset = ImageFolder(train_dir, transform=train_transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=0)
    print(f"  {len(dataset)} training images found\n")

    # Initialize models
    transform_net = TransformNet().to(device)
    vgg_model     = VGGFeatures()
    optimizer     = optim.Adam(transform_net.parameters(), lr=LR)

    # Pre-compute style targets — fixed throughout training
    print("Pre-computing style Gram matrices...")
    style_grams = get_style_grams(vgg_model, style_img_path)
    print("  Done\n")

    transform_net.train()

    for epoch in range(1, epochs + 1):
        total_loss_sum = 0

        for batch_idx, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)

            # Forward pass through transform network
            stylized = transform_net(imgs)

            # Get VGG features for both stylized and original (content) images
            stylized_style_feats, stylized_content_feat = vgg_model(stylized)
            _,                    content_feat           = vgg_model(imgs)

            # Content loss — MSE between stylized and original activations
            # Both are [B, N, H, W] — torch.mean handles batched tensors fine
            l_content = torch.mean((stylized_content_feat - content_feat) ** 2)

            # Style loss — stylized image Gram matrices should match style targets
            # Must compute Gram matrix per image in the batch, then average
            l_style = 0
            from neural_style_transfer import STYLE_LAYERS
            for layer_idx in STYLE_LAYERS:
                feat = stylized_style_feats[layer_idx]   # [B, N, H, W]
                B, N, H, W = feat.shape
                # Reshape each image in batch separately: [B, N, H*W]
                f = feat.view(B, N, H * W)
                # Batch matrix multiply: [B, N, H*W] x [B, H*W, N] -> [B, N, N]
                G_stylized = torch.bmm(f, f.transpose(1, 2)) / (N * H * W)
                # Average Gram matrix across the batch
                G_stylized_mean = G_stylized.mean(dim=0)   # [N, N]
                G_style = style_grams[layer_idx]           # [N, N]
                l_style += torch.mean((G_stylized_mean - G_style) ** 2)
            l_style /= len(STYLE_LAYERS)

            loss = ALPHA * l_content + BETA * l_style

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(loader)} "
                      f"| Loss: {loss.item():.2f} "
                      f"| Content: {l_content.item():.4f} "
                      f"| Style: {l_style.item():.6f}")

        print(f"Epoch {epoch} complete | Avg loss: {total_loss_sum/len(loader):.2f}\n")

    # Save trained model weights
    torch.save(transform_net.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved: {MODEL_SAVE_PATH}")
    return transform_net

# ── Inference — single forward pass ──────────────────────────────────────────
def stylize(transform_net, content_img_path, output_name="fast_output"):
    """
    Stylize a single image using the trained transform network.
    No optimization loop — just one forward pass.
    """
    transform_net.eval()
    content = load_image(content_img_path)

    with torch.no_grad():
        output = transform_net(content)

    pil = tensor_to_pil(output)
    path = os.path.join(OUTPUT_DIR, f"{output_name}.jpg")
    pil.save(path)
    print(f"Saved: {path}")
    return output

# ── Load a previously trained model ──────────────────────────────────────────
def load_model(path):
    net = TransformNet().to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    print(f"Loaded model from {path}")
    return net

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # STEP 1: Train the transform network
    # Requires a folder of training images at images/train/
    # See comment above about downloading COCO val2017
    transform_net = train(STYLE_IMG, TRAIN_IMG_DIR)

    # STEP 2: Stylize your content image — instant after training
    stylize(transform_net, CONTENT_IMG, output_name="fast_output")

    # To load a saved model later and skip retraining:
    # transform_net = load_model(MODEL_SAVE_PATH)
    # stylize(transform_net, CONTENT_IMG)