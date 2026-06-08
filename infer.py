"""
Extract embeddings from a folder and save to embeddings.json
Output: { "A1A": [0.023, -0.184, ...], "A1B": [...], "B1A": [...], ... }

Usage:
    python infer.py                                      # uses default paths
    python infer.py --folder processed_data/train        # custom folder
    python infer.py --folder processed_data/train --save_json my_embeddings.json
"""

import os
import json
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 112
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class FaceEmbedder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        bb = models.resnet18(weights=None)  # ResNet-18 backbone
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.bn     = nn.BatchNorm1d(512)   # ResNet-18 outputs 512 features
        self.drop   = nn.Dropout(p=0.0)
        self.fc     = nn.Linear(512, embed_dim, bias=False)

    def forward(self, x):
        x = self.stem(x);   x = self.layer1(x)
        x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x);     x = self.drop(x)
        return self.fc(x)

class SoftmaxHead(nn.Module):
    def __init__(self, embed_dim, num_classes, s=30.0):
        super().__init__()
        self.s  = s
        self.fc = nn.Linear(embed_dim, num_classes, bias=False)
    def forward(self, x):
        return self.s * nn.functional.linear(
            nn.functional.normalize(x, dim=1),
            nn.functional.normalize(self.fc.weight, dim=1))

class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes, s=32.0):
        super().__init__()
        self.s      = s
        self.cos_m  = math.cos(0.0)
        self.sin_m  = math.sin(0.0)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
    def forward(self, x, labels):
        return nn.functional.linear(
            nn.functional.normalize(x, dim=1),
            nn.functional.normalize(self.weight, dim=1)).clamp(-1+1e-7, 1-1e-7) * self.s

class FaceModel(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.embedder     = FaceEmbedder(embed_dim)
        self.softmax_head = SoftmaxHead(embed_dim, num_classes, s=30.0)
        self.arcface_head = ArcFaceHead(embed_dim, num_classes, s=32.0)
        self.phase        = "softmax"
    def forward(self, x, labels=None):
        emb = self.embedder(x)
        if self.phase == "arcface" and labels is not None:
            return self.arcface_head(emb, labels), emb
        return self.softmax_head(emb), emb
    def get_embeddings(self, x):
        return nn.functional.normalize(self.embedder(x), dim=1)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model(checkpoint_path):
    ckpt      = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    classes   = ckpt["classes"]
    embed_dim = ckpt.get("embed_dim", 512)
    model     = FaceModel(embed_dim, len(classes))
    model.load_state_dict(ckpt["model"])
    model.phase = ckpt.get("phase", "softmax")
    model.to(DEVICE).eval()
    print(f"[INFO] Loaded  val_acc={ckpt.get('val_acc',0):.2f}%  phase={model.phase}")
    return model


# ─────────────────────────────────────────────
# EXTRACT
# ─────────────────────────────────────────────
infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

@torch.no_grad()
def extract(model, img_path):
    t = infer_tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    return model.get_embeddings(t).squeeze().cpu().numpy().tolist()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints/best_model.pth")
    parser.add_argument("--folder",     default="../train_me/train")
    parser.add_argument("--save_json",  default="embeddings.json")
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    # Walk through identity subfolders
    embeddings = {}
    identities = sorted([
        d for d in os.listdir(args.folder)
        if os.path.isdir(os.path.join(args.folder, d))
    ])

    total = sum(
        len([f for f in os.listdir(os.path.join(args.folder, d))
             if f.lower().endswith(IMG_EXTS)])
        for d in identities
    )

    done = 0
    print(f"[INFO] {len(identities)} identities, {total} images total\n")

    for identity in identities:
        id_dir = os.path.join(args.folder, identity)
        files  = sorted([f for f in os.listdir(id_dir) if f.lower().endswith(IMG_EXTS)])
        for fname in files:
            img_name              = os.path.splitext(fname)[0]   # e.g. "A1A"
            embeddings[img_name]  = extract(model, os.path.join(id_dir, fname))
            done += 1
            print(f"  [{done}/{total}] {img_name}")

    # Save
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)

    print(f"\n[SAVED] {args.save_json}  ({len(embeddings)} embeddings)")
    print(f"[INFO]  Format: {{ \"A1A\": [0.023, -0.184, ...], \"A1B\": [...], ... }}")


if __name__ == "__main__":
    main()