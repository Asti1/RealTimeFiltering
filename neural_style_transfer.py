# Astitva Goel

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt
import os
import numpy as np

# ── Device ────────────────────────────────────────────────────────────────────
# Use GPU if available otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 512 if torch.cuda.is_available() else 256
# Smaller size on CPU so optimization does not take forever

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# VGG Layers are named 0,1,2,... in vgg.features
# conv1_2 = 2, conv2_2 = 7, conv3_4 = 16, conv4_2 = 21, conv4_4 = 25
STYLE_LAYERS  = [2, 7, 16, 21, 25]
CONTENT_LAYER = 21

# ── Image utilities ───────────────────────────────────────────────────────────

# PREPROCESSING
# VGG was trained on ImageNet so it expects the exact normalisation as done there
loader = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Reverse transformation to display images: undo normalization and clamp to [0,1]
unloader = transforms.Compose([
    transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    ),
    transforms.Lambda(lambda t: t.clamp(0, 1))
])

def load_image(path):
    img = Image.open(path).convert('RGB')
    # Adding batch dimension [C,H,W] -> [1,C,H,W]
    img = loader(img).unsqueeze(0)
    return img.to(device)

def tensor_to_pil(tensor):
    """Convert a [1,C,H,W] tensor to a PIL image for saving."""
    img = tensor.squeeze(0).cpu()
    img = unloader(img)
    img = img.permute(1, 2, 0).numpy()   # [C,H,W] -> [H,W,C]
    img = (img * 255).astype("uint8")
    return Image.fromarray(img)

def show_image(tensor, title="output"):
    # Saves to outputs/ folder instead of opening a window
    pil = tensor_to_pil(tensor)
    filename = title.replace(" ", "_").lower() + ".jpg"
    path = os.path.join(OUTPUT_DIR, filename)
    pil.save(path)
    print(f"  Saved: {path}")

# ── VGG Feature Extraction ────────────────────────────────────────────────────
class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        # Loading pretrained VGG19 except the classifier
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        vgg = vgg.to(device).eval()  # Set to eval mode and move to device

        # Freeze VGG weights — we NEVER update these
        for param in vgg.parameters():
            param.requires_grad_(False)

        self.layers = vgg

    def forward(self, x):
        # Run forward pass and return activations at all style layers
        # and the content layer
        style_feats  = {}
        content_feat = None
        stop_at = max(STYLE_LAYERS + [CONTENT_LAYER])

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in STYLE_LAYERS:
                style_feats[i] = x
            if i == CONTENT_LAYER:
                content_feat = x
            if i >= stop_at:
                break

        return style_feats, content_feat

# ── Loss Functions ────────────────────────────────────────────────────────────
def gram_matrix(feature):
    # feature: [1, N, H, W]  (batch=1, N filters, H height, W width)
    # returns: [N, N] Gram matrix
    _, N, H, W = feature.shape
    # Reshape to [N, H*W] - each row is one filter's spatial response
    f = feature.view(N, H * W)
    # Matrix multiplication to get Gram matrix
    G = torch.mm(f, f.t())
    # Normalize so loss scale is consistent across layers of different sizes
    return G / (N * H * W)

def content_loss(gen_feat, content_feat):
    # MSE between activation tensors at conv4_2
    return torch.mean((gen_feat - content_feat) ** 2)

def style_loss(gen_style_feat, style_style_feat):
    # Sum of Gram matrix MSE across all 5 style layers
    # Equal weight (1/5) per layer
    loss = 0
    for layer_idx in STYLE_LAYERS:
        G_gen   = gram_matrix(gen_style_feat[layer_idx])
        G_style = gram_matrix(style_style_feat[layer_idx])
        loss += torch.mean((G_gen - G_style) ** 2)
    return loss / len(STYLE_LAYERS)   # outside loop — averages across all layers

def total_loss(vgg_model, gen, content_img, style_img, alpha=1, beta=1e6):
    # alpha: content weight
    # beta:  style weight  (default ratio 1:1,000,000 — Gatys default)
    gen_style_feats,    gen_content_feat    = vgg_model(gen)
    _,                  target_content_feat = vgg_model(content_img)
    target_style_feats, _                  = vgg_model(style_img)

    l_content = content_loss(gen_content_feat, target_content_feat)
    l_style   = style_loss(gen_style_feats, target_style_feats)

    return alpha * l_content + beta * l_style, l_content, l_style

# ── Optimization Loop ─────────────────────────────────────────────────────────
def run_style_transfer(content_img, style_img, num_steps=300, alpha=1, beta=1e6, show_every=100):
    # content image, style image : preprocessed tensors [1,3,H,W]
    # num_steps: number of optimization steps
    # alpha, beta: content and style weights
    # show_every: save the generated image every N steps

    # Initialize VGG once per run — weights stay frozen throughout
    vgg_model = VGGFeatures()

    # Initialize G as a copy of the content image
    G = content_img.clone().requires_grad_(True)

    # L-BFGS optimizer is standard for NST - converges faster than Adam
    optimizer = optim.LBFGS([G])

    # L-BFGS requires a closure, a function it can call multiple times to evaluate the loss
    def closure():
        with torch.no_grad():
            G.clamp_(-2.5, 2.5)
        optimizer.zero_grad()
        loss, l_content, l_style = total_loss(vgg_model, G, content_img, style_img, alpha, beta)
        loss.backward()
        return loss

    # Outer loop counts actual optimizer steps (not closure calls)
    # L-BFGS calls closure multiple times per step internally for line search
    for step in range(1, num_steps + 1):
        optimizer.step(closure)
        # Print every single step so you can see progress
        with torch.no_grad():
            loss, l_content, l_style = total_loss(vgg_model, G, content_img, style_img, alpha, beta)
        print(f"Step {step:4d}/{num_steps} | Total: {loss.item():.2f} "
              f"| Content: {l_content.item():.4f} | Style: {l_style.item():.6f}")
        if step % show_every == 0:
            show_image(G.detach(), title=f"step_{step}")

    return G.detach()

# ── Main ──────────────────────────────────────────────────────────────────────
# This block only runs when you execute this file directly,
# not when it's imported by experiment_runner.py
if __name__ == "__main__":
    # Load your images (update paths to wherever your files are)
    content = load_image("images/content.jpg")
    style   = load_image("images/style.jpg")

    # Quick sanity check — make sure they loaded correctly
    show_image(content, "content_image")
    show_image(style,   "style_image")

    # Run the transfer
    output = run_style_transfer(
        content, style,
        num_steps=20,
        alpha=1,
        beta=1e6,
        show_every=10
    )

    # Final result
    show_image(output, "final_result")