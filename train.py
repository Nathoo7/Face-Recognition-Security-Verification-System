"""
ArcFace Training (ResNet18 backbone) — tuned for ~5 000-image datasets.
Saves checkpoint history to checkpoints/training_log.txt after every epoch.
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT     = "./train_me"
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
WARMUP_EPOCHS = 12
EMBED_DIM     = 256
LR_MAX        = 1e-3
WEIGHT_DECAY  = 5e-4
IMG_SIZE      = 112
NUM_WORKERS   = 4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR      = "./checkpoints"
LOG_PATH      = os.path.join(SAVE_DIR, "training_log.txt")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"[INFO] Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ─────────────────────────────────────────────
# COLLECT IMAGE FILENAMES PER IDENTITY
# ─────────────────────────────────────────────
def collect_image_index(split_dir):
    index    = {}
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    for identity in sorted(os.listdir(split_dir)):
        id_path = os.path.join(split_dir, identity)
        if not os.path.isdir(id_path):
            continue
        files = sorted([f for f in os.listdir(id_path) if f.lower().endswith(img_exts)])
        if files:
            index[identity] = files
    return index


# ─────────────────────────────────────────────
# LIGHT DATA AUGMENTATION AND PREPROCESSING FOR TRAINING SET(for casia-webface subset)
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.08),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
])

# ─────────────────────────────────────────────
#PREPROCESSING FOR VALIDATION (no augmentation, just resizing and normalization)
# ─────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),   transform=val_transform)

NUM_CLASSES = len(train_dataset.classes)
print(f"[INFO] Identities : {NUM_CLASSES}")
print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)}")
print(f"[INFO] Steps/epoch: {len(train_dataset) // BATCH_SIZE}")

train_image_index    = collect_image_index(os.path.join(DATA_ROOT, "train"))
validate_image_index = collect_image_index(os.path.join(DATA_ROOT, "val"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


# ─────────────────────────────────────────────
# MODEL  — ResNet18 backbone
# ─────────────────────────────────────────────
class FaceEmbedder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.bn   = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(p=0.3)
        self.fc   = nn.Linear(512, embed_dim, bias=False)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')
        for p in self.stem.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x);     x = self.drop(x)
        return self.fc(x)


class SoftmaxHead(nn.Module):
    def __init__(self, embed_dim, num_classes, s=24.0):
        super().__init__()
        self.s  = s
        self.fc = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        return self.s * nn.functional.linear(
            nn.functional.normalize(x, dim=1),
            nn.functional.normalize(self.fc.weight, dim=1))


class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes, s=26.0, m_target=0.25):
        super().__init__()
        self.s        = s
        self.m_target = m_target
        self.m        = 0.0
        self.weight   = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self._upd()

    def _upd(self):
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

    def set_margin(self, epoch, start, total):
        t      = (epoch - start) / max(total - start, 1)
        self.m = self.m_target * min(t, 1.0)
        self._upd()

    def forward(self, x, labels):
        x_n = nn.functional.normalize(x, dim=1)
        w_n = nn.functional.normalize(self.weight, dim=1)
        cos = nn.functional.linear(x_n, w_n).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1.0 - cos ** 2)
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > 0, phi, cos)
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        return (one_hot * phi + (1.0 - one_hot) * cos) * self.s


class FaceModel(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.embedder     = FaceEmbedder(embed_dim)
        self.softmax_head = SoftmaxHead(embed_dim, num_classes, s=24.0)
        self.arcface_head = ArcFaceHead(embed_dim, num_classes, s=26.0, m_target=0.25)
        self.phase        = "softmax"

    def forward(self, x, labels=None):
        emb = self.embedder(x)
        if self.phase == "arcface" and labels is not None:
            return self.arcface_head(emb, labels), emb
        return self.softmax_head(emb), emb

    def to_arcface(self):
        self.phase = "arcface"
        print("[INFO] Phase -> ArcFace  (margin 0.0 -> 0.25)")

    def get_embeddings(self, x):
        return nn.functional.normalize(self.embedder(x), dim=1)


# ─────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits, _ = model(imgs, labels)
            loss      = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total * 100


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with autocast():
            logits, _ = model(imgs, labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100


# ─────────────────────────────────────────────
# TXT LOG HELPERS
# ─────────────────────────────────────────────
def init_log(path, num_classes, classes):
    """Write the header section of the log file."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("         ARCFACE TRAINING LOG\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Started       : {now}\n")
        f.write(f"  Backbone      : ResNet18\n")
        f.write(f"  Data Root     : {DATA_ROOT}\n")
        f.write(f"  Batch Size    : {BATCH_SIZE}\n")
        f.write(f"  Epochs        : {NUM_EPOCHS}  (warmup: {WARMUP_EPOCHS})\n")
        f.write(f"  Embed Dim     : {EMBED_DIM}\n")
        f.write(f"  Learning Rate : {LR_MAX}  (weight decay: {WEIGHT_DECAY})\n")
        f.write(f"  Image Size    : {IMG_SIZE} x {IMG_SIZE}\n")
        f.write(f"  Identities    : {num_classes}\n")
        f.write(f"  Classes       : {', '.join(classes)}\n")
        f.write("=" * 70 + "\n\n")
        f.write("EPOCH RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Epoch':>6}  {'Phase':<10}  {'Margin':>6}  {'LR':>9}  "
                f"{'Train Loss':>10}  {'Train Acc':>9}  {'Val Acc':>8}  "
                f"{'Gap':>7}  {'Time(s)':>7}  {'Best?':>5}\n")
        f.write("-" * 70 + "\n")


def append_epoch_log(path, epoch_data):
    """Append one epoch result row to the log file."""
    e = epoch_data
    best_marker = "  <--" if e["is_best"] else ""
    warn_marker = "  [OVERFIT]" if e["gap"] > 25 else ""

    with open(path, "a") as f:
        f.write(
            f"{e['epoch']:>6}  "
            f"{e['phase']:<10}  "
            f"{e['margin']:>6.3f}  "
            f"{e['lr']:>9.2e}  "
            f"{e['train_loss']:>10.4f}  "
            f"{e['train_acc']:>8.2f}%  "
            f"{e['val_acc']:>7.2f}%  "
            f"{e['gap']:>+7.2f}  "
            f"{e['elapsed_s']:>7.1f}"
            f"{best_marker}{warn_marker}\n"
        )


def write_summary(path, best_info):
    """Append the final summary section to the log file."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a") as f:
        f.write("-" * 70 + "\n\n")
        f.write("=" * 70 + "\n")
        f.write("         FINAL SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Finished        : {now}\n")
        f.write(f"  Best Val Epoch  : {best_info['epoch']}\n")
        f.write(f"  Best Val Acc    : {best_info['val_acc']:.2f}%\n")
        f.write(f"  Best Train Acc  : {best_info['train_acc']:.2f}%\n")
        f.write(f"  Best Train Loss : {best_info['train_loss']:.4f}\n")
        f.write(f"  Best Phase      : {best_info['phase']}\n")
        f.write(f"  Checkpoint      : {best_info['checkpoint']}\n")
        f.write("=" * 70 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    model     = FaceModel(EMBED_DIM, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = GradScaler()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] Trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")

    optimizer = optim.AdamW(trainable, lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR_MAX,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0,
    )

    best_val_acc = 0.0
    best_info    = {}

    init_log(LOG_PATH, NUM_CLASSES, list(train_dataset.classes))
    print(f"[INFO] Training log → {LOG_PATH}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        if epoch == WARMUP_EPOCHS + 1 and model.phase == "softmax":
            model.to_arcface()

        if model.phase == "arcface":
            model.arcface_head.set_margin(epoch, WARMUP_EPOCHS + 1, NUM_EPOCHS)
            phase_str = f"ArcFace  m={model.arcface_head.m:.3f}"
            margin    = round(model.arcface_head.m, 4)
        else:
            phase_str = "Softmax"
            margin    = 0.0

        lr_now = optimizer.param_groups[0]['lr']
        print(f"\n{'─'*60}")
        print(f" Epoch {epoch:03d}/{NUM_EPOCHS}  |  {phase_str}  |  LR: {lr_now:.2e}")
        print(f"{'─'*60}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler)
        val_acc = evaluate(model, val_loader)
        elapsed = round(time.time() - t0, 1)
        gap     = round(train_acc - val_acc, 2)

        print(f"  Train Loss : {train_loss:.4f}  |  Train Acc : {train_acc:.2f}%")
        print(f"  Val   Acc  : {val_acc:.2f}%  |  Gap: {gap:+.1f}pp  ({elapsed}s)")
        if gap > 25:
            print(f"  [WARN] Overfitting gap={gap:.1f}pp")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            ckpt_path    = os.path.join(SAVE_DIR, "best_model_1.pth")
            torch.save({
                "epoch":                epoch,
                "model":                model.state_dict(),
                "val_acc":              val_acc,
                "phase":                model.phase,
                "embed_dim":            EMBED_DIM,
                "classes":              train_dataset.classes,
                "train_image_index":    train_image_index,
                "validate_image_index": validate_image_index,
                "data_root":            os.path.abspath(DATA_ROOT),
            }, ckpt_path)
            print(f"  [BEST] Saved  val={val_acc:.2f}%")

            best_info = {
                "epoch":      epoch,
                "val_acc":    round(val_acc,    2),
                "train_acc":  round(train_acc,  2),
                "train_loss": round(train_loss, 4),
                "phase":      model.phase,
                "checkpoint": ckpt_path,
            }

        append_epoch_log(LOG_PATH, {
            "epoch":      epoch,
            "phase":      model.phase,
            "margin":     margin,
            "lr":         lr_now,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  2),
            "val_acc":    round(val_acc,    2),
            "gap":        gap,
            "is_best":    is_best,
            "elapsed_s":  elapsed,
        })

    print(f"\n{'='*60}")
    print(f" Best Val Acc  : {best_val_acc:.2f}%")

    write_summary(LOG_PATH, best_info)
    print(f"\n[INFO] Full training log saved → {LOG_PATH}")


if __name__ == "__main__":
    main()