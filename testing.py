"""
Testing — Compare every test image against every train image
Shows:
    A1A vs A1B    score: 0.80   GENUINE  ✓
    A1A vs B1A    score: 0.20   IMPOSTOR ✓
Summary:
    Genuine accuracy, Impostor accuracy, Total accuracy

Expects embeddings.json format:
    { "A1A": [0.023, -0.184, ...], "A1B": [...], "B1A": [...], ... }
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
from datetime import datetime

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 112
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class FaceEmbedder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        bb = models.resnet18(weights=None)
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.bn     = nn.BatchNorm1d(512)
        self.drop   = nn.Dropout(p=0.0)
        self.fc     = nn.Linear(512, embed_dim, bias=False)

    def forward(self, x):
        x = self.stem(x);   x = self.layer1(x)
        x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x);     x = self.drop(x)
        return self.fc(x)

class SoftmaxHead(nn.Module):
    def __init__(self, embed_dim, num_classes, s=24.0):
        super().__init__()
        self.s  = s
        self.fc = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, x):
        return self.s * nn.functional.linear(
            nn.functional.normalize(x, dim=1),
            nn.functional.normalize(self.fc.weight, dim=1))

class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes, s=26.0):
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
        self.softmax_head = SoftmaxHead(embed_dim, num_classes, s=24.0)
        self.arcface_head = ArcFaceHead(embed_dim, num_classes, s=26.0)
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
    embed_dim = ckpt.get("embed_dim", 256)
    model     = FaceModel(embed_dim, len(classes))
    model.load_state_dict(ckpt["model"])
    model.phase = ckpt.get("phase", "softmax")
    model.to(DEVICE).eval()
    print(f"[INFO] Model loaded  |  val_acc={ckpt.get('val_acc',0):.2f}%  |  phase={model.phase}")
    return model, ckpt.get("val_acc", 0)


# ─────────────────────────────────────────────
# LOAD EMBEDDINGS JSON
# Builds img_name -> identity map from the actual train folder
# so any naming convention (A1A, A1AA, A1AAA...) works correctly
# ─────────────────────────────────────────────
def load_train_embeddings(json_path, train_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build reverse map: img_name (no ext) -> identity folder name
    img_to_identity = {}
    if train_dir and os.path.isdir(train_dir):
        for identity in os.listdir(train_dir):
            id_dir = os.path.join(train_dir, identity)
            if not os.path.isdir(id_dir):
                continue
            for fname in os.listdir(id_dir):
                img_name = os.path.splitext(fname)[0]
                img_to_identity[img_name] = identity
    else:
        print(f"[WARN] train_dir not found: {train_dir}  —  falling back to key=identity")

    flat       = []
    unresolved = []
    for img_name, vec in data.items():
        if img_name in img_to_identity:
            identity = img_to_identity[img_name]
        else:
            # Averaged embeddings — key is already the identity (e.g. "A1")
            identity = img_name
            unresolved.append(img_name)
        flat.append((identity, img_name, np.array(vec, dtype=np.float32)))

    identities = set(item[0] for item in flat)
    if unresolved:
        print(f"[INFO] {len(unresolved)} keys not found in train_dir "
              f"— treated as identity directly (averaged mode)")
    print(f"[INFO] Train embeddings: {len(flat)} entries across {len(identities)} identities")

    # Debug: show sample mappings
    print(f"[DEBUG] Sample gallery  — img: identity")
    for identity, img_name, _ in flat[:5]:
        print(f"         {img_name} -> {identity}")

    return flat


# ─────────────────────────────────────────────
# EXTRACT EMBEDDING FROM IMAGE
# ─────────────────────────────────────────────
infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

@torch.no_grad()
def extract(model, img_path):
    t = infer_tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    return model.get_embeddings(t).squeeze().cpu().numpy()


# ─────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ─────────────────────────────────────────────
# WRITE RESULTS TO TXT
# ─────────────────────────────────────────────
def save_txt(path, args, val_acc, results_by_test, summary):
    W = 80

    with open(path, "w", encoding="utf-8") as f:

        f.write("=" * W + "\n")
        f.write("  FACE RECOGNITION TEST RESULTS\n")
        f.write("=" * W + "\n")
        f.write(f"  Date       : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n")
        f.write(f"  Checkpoint : {args.checkpoint}\n")
        f.write(f"  Embeddings : {args.embeddings}\n")
        f.write(f"  Train dir  : {args.train_dir}\n")
        f.write(f"  Test dir   : {args.test_dir}\n")
        f.write(f"  Threshold  : {args.threshold}  (>= genuine, < impostor)\n")
        f.write(f"  Model val  : {val_acc:.2f}%\n")
        f.write("=" * W + "\n\n")

        f.write("DETAILED COMPARISONS\n")
        f.write("-" * W + "\n")
        f.write(f"  {'TEST IMAGE':<20}  {'TRAIN IMAGE':<20}  {'SCORE':>7}  {'TYPE':<9}  RESULT\n")
        f.write("-" * W + "\n")

        for test_name, pairs in results_by_test.items():
            f.write(f"\n  >> {test_name}\n")

            genuine_pairs  = [p for p in pairs if p["genuine"]]
            impostor_pairs = [p for p in pairs if not p["genuine"]]

            if genuine_pairs:
                f.write(f"  {'':4}{'--- Genuine ---'}\n")
                for p in genuine_pairs:
                    mark = "✓" if p["correct"] else "✗"
                    f.write(f"  {'':4}{p['test_image']:<20}  {p['train_image']:<20}"
                            f"  {p['score']:>7.4f}  {'GENUINE':<9}  {mark}\n")

            if impostor_pairs:
                f.write(f"  {'':4}{'--- Impostor ---'}\n")
                for p in impostor_pairs:
                    mark = "✓" if p["correct"] else "✗"
                    f.write(f"  {'':4}{p['test_image']:<20}  {p['train_image']:<20}"
                            f"  {p['score']:>7.4f}  {'IMPOSTOR':<9}  {mark}\n")

            t_gen  = len(genuine_pairs);  c_gen = sum(p["correct"] for p in genuine_pairs)
            t_imp  = len(impostor_pairs); c_imp = sum(p["correct"] for p in impostor_pairs)
            f.write(f"\n  {'':4}Genuine: {c_gen}/{t_gen}   "
                    f"Impostor: {c_imp}/{t_imp}   "
                    f"Total: {c_gen+c_imp}/{t_gen+t_imp}\n")
            f.write("  " + "·" * (W - 2) + "\n")

        f.write("\n" + "=" * W + "\n")
        f.write("  OVERALL SUMMARY\n")
        f.write("=" * W + "\n")
        f.write(f"  {'Metric':<30}  {'Correct':>8}  {'Total':>8}  {'Accuracy':>10}\n")
        f.write("  " + "-" * (W - 2) + "\n")
        f.write(f"  {'Genuine  pairs':<30}  "
                f"{summary['genuine_correct']:>8}  "
                f"{summary['genuine_total']:>8}  "
                f"{summary['genuine_acc']*100:>9.2f}%\n")
        f.write(f"  {'Impostor pairs':<30}  "
                f"{summary['impostor_correct']:>8}  "
                f"{summary['impostor_total']:>8}  "
                f"{summary['impostor_acc']*100:>9.2f}%\n")
        f.write("  " + "-" * (W - 2) + "\n")
        f.write(f"  {'TOTAL':<30}  "
                f"{summary['total_correct']:>8}  "
                f"{summary['total_pairs']:>8}  "
                f"{summary['total_acc']*100:>9.2f}%\n")
        f.write("=" * W + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints/best_model.pth")
    parser.add_argument("--embeddings", default="../infer/embeddings.json")
    parser.add_argument("--train_dir",  default="../train_me/train",
                        help="Train folder used during infer.py — maps image names to identities")
    parser.add_argument("--test_dir",   default="../train_me/test")
    parser.add_argument("--threshold",  default=0.35, type=float,
                        help="Score >= threshold -> GENUINE, else IMPOSTOR")
    parser.add_argument("--save_txt",   default="test_results.txt")
    args = parser.parse_args()

    model, val_acc = load_model(args.checkpoint)
    train_db       = load_train_embeddings(args.embeddings, args.train_dir)

    # Debug: show sample test folders to verify they match gallery identities
    test_folders = sorted(os.listdir(args.test_dir))
    print(f"[DEBUG] Sample test folders: {test_folders[:5]}")

    # Collect test images
    test_images = []
    for identity in sorted(os.listdir(args.test_dir)):
        id_dir = os.path.join(args.test_dir, identity)
        if not os.path.isdir(id_dir):
            continue
        for fname in sorted(os.listdir(id_dir)):
            if fname.lower().endswith(IMG_EXTS):
                test_images.append((identity, fname, os.path.join(id_dir, fname)))

    if not test_images:
        print(f"[ERROR] No images found in {args.test_dir}")
        return

    print(f"[INFO] Test images : {len(test_images)}")
    print(f"[INFO] Threshold   : {args.threshold}")
    print(f"[INFO] Total pairs : {len(test_images) * len(train_db)}\n")

    genuine_correct  = 0
    genuine_total    = 0
    impostor_correct = 0
    impostor_total   = 0
    results_by_test  = {}

    for test_id, test_fname, test_path in test_images:
        test_emb  = extract(model, test_path)
        test_name = os.path.splitext(test_fname)[0]
        results_by_test[test_name] = []

        for train_id, train_name, train_emb in train_db:
            score             = cosine(test_emb, train_emb)
            is_genuine        = (test_id == train_id)
            predicted_genuine = score >= args.threshold
            correct           = (is_genuine == predicted_genuine)
            mark              = "✓" if correct else "✗"
            pair_type         = "GENUINE " if is_genuine else "IMPOSTOR"

            print(f"  {test_name:<20} vs {train_name:<20}  "
                  f"score: {score:.4f}   {pair_type}  {mark}")

            if is_genuine:
                genuine_total   += 1
                genuine_correct += correct
            else:
                impostor_total   += 1
                impostor_correct += correct

            results_by_test[test_name].append({
                "test_image":  test_name,
                "test_id":     test_id,
                "train_image": train_name,
                "train_id":    train_id,
                "score":       round(score, 6),
                "genuine":     is_genuine,
                "correct":     bool(correct),
            })

    total_correct = genuine_correct + impostor_correct
    total_pairs   = genuine_total   + impostor_total
    genuine_acc   = genuine_correct  / genuine_total  if genuine_total  else 0
    impostor_acc  = impostor_correct / impostor_total if impostor_total else 0
    total_acc     = total_correct    / total_pairs    if total_pairs    else 0

    summary = {
        "genuine_correct":  genuine_correct,
        "genuine_total":    genuine_total,
        "genuine_acc":      round(genuine_acc,  4),
        "impostor_correct": impostor_correct,
        "impostor_total":   impostor_total,
        "impostor_acc":     round(impostor_acc, 4),
        "total_correct":    total_correct,
        "total_pairs":      total_pairs,
        "total_acc":        round(total_acc,    4),
    }

    print(f"\n{'='*60}")
    print(f"  SUMMARY  (threshold = {args.threshold})")
    print(f"{'='*60}")
    print(f"  Genuine  pairs : {genuine_correct:>5} / {genuine_total:<6}  acc = {genuine_acc*100:.2f}%")
    print(f"  Impostor pairs : {impostor_correct:>5} / {impostor_total:<6}  acc = {impostor_acc*100:.2f}%")
    print(f"  {'─'*50}")
    print(f"  Total          : {total_correct:>5} / {total_pairs:<6}  acc = {total_acc*100:.2f}%")
    print(f"{'='*60}")

    save_txt(args.save_txt, args, val_acc, results_by_test, summary)
    print(f"\n[SAVED] {args.save_txt}")


if __name__ == "__main__":
    main()