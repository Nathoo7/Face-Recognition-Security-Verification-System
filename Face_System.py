"""
face_system.py
==============
Face verification system:
  - Detection & alignment : InsightFace RetinaFace (from buffalo_l)
  - Embedding             : YOUR trained arcface_backbone_best.pth
  - Storage               : per-model face_descriptors_<key>.json

Install dependencies:
  pip install insightface onnxruntime-gpu opencv-python torch torchvision

Usage:
  python face_system.py
"""

import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from backbone import FaceBackbone
import torchvision.models as models

# ── Matches the exact architecture used during training ───────────────────────
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
        self.drop   = nn.Dropout(p=0.3)
        self.fc     = nn.Linear(512, embed_dim, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x);     x = self.drop(x)
        return self.fc(x)

# InsightFace — only used for RetinaFace detection + alignment
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# =========================
# CONFIGURATION
# =========================
IMG_SIZE  = 112
THRESHOLD = 0.45
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model registry — each key maps to its weights file AND descriptor store ───
AVAILABLE_CONFIGS = {
    "imperial": {
        "weights":     "Imperial_model.pth",
        "descriptors": "face_descriptors_imperial.json",
    },
    "casia": {
        "weights":     "casia_model.pth",
        "descriptors": "face_descriptors_casia.json",
    },
}
ACTIVE_MODEL_KEY = "imperial"
MODEL_WEIGHTS    = AVAILABLE_CONFIGS[ACTIVE_MODEL_KEY]["weights"]
DESCRIPTOR_FILE  = AVAILABLE_CONFIGS[ACTIVE_MODEL_KEY]["descriptors"]
# ─────────────────────────────────────────────────────────────────────────────

# =========================
# TRANSFORM
# =========================
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# =========================
# DESCRIPTOR STORE
# =========================
class DescriptorStore:
    def __init__(self, path=None):
        self.path = path or DESCRIPTOR_FILE
        self.db   = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                raw = json.load(f)
            self.db = {name: [np.array(e) for e in embs]
                       for name, embs in raw.items()}
            total = sum(len(v) for v in self.db.values())
            print(f"  [DB] Loaded {total} descriptor(s) for "
                  f"{len(self.db)} identity(s) from '{self.path}'")
        else:
            print(f"  [DB] No existing descriptors at '{self.path}' — starting fresh.")

    def save(self):
        raw = {name: [e.tolist() for e in embs]
               for name, embs in self.db.items()}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(raw, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _reload_from_file(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                raw = json.load(f)
            self.db = {name: [np.array(e) for e in embs]
                       for name, embs in raw.items()}

    def add(self, name: str, embedding: np.ndarray):
        if name not in self.db:
            self.db[name] = []
        self.db[name].append(embedding)
        self.save()
        self._reload_from_file()

    def get_mean_embeddings(self):
        result = {}
        for name, embs in self.db.items():
            mean_emb = np.mean(embs, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
            result[name] = mean_emb
        return result

    def list_identities(self):
        return {name: len(embs) for name, embs in self.db.items()}

    def delete(self, name: str):
        if name in self.db:
            del self.db[name]
            self.save()
            self._reload_from_file()
            return True
        return False


# =========================
# RETINAFACE DETECTOR (lazy loaded)
# =========================
_detector = None

def get_detector():
    global _detector
    if _detector is not None:
        return _detector
    print("  [Detector] Loading RetinaFace from buffalo_l...")
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    _detector = app
    print("  [Detector] RetinaFace ready.")
    return _detector


def detect_faces(frame_bgr):
    app   = get_detector()
    faces = app.get(frame_bgr)
    return faces


# =========================
# ARCFACE BACKBONE (lazy loaded)
# =========================
_backbone = None

def get_model():
    global _backbone
    if _backbone is not None:
        return _backbone
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(
            f"Model weights not found: '{MODEL_WEIGHTS}'\n"
            "Make sure the file is in this folder."
        )

    checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        embed_dim  = int(checkpoint.get("embed_dim", 512))
        state_dict = checkpoint["model"]
    else:
        embed_dim  = 512
        state_dict = checkpoint

    if any(k.startswith("embedder.") for k in state_dict.keys()):
        state_dict = {
            k[len("embedder."):]: v
            for k, v in state_dict.items()
            if k.startswith("embedder.")
        }

    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith(("stem.", "layer1.", "layer2.", "bn.", "fc.")):
        model = FaceEmbedder(embed_dim)
    else:
        model = FaceBackbone(embed_dim)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    _backbone = model
    print(f"  [Model] Loaded '{MODEL_WEIGHTS}'  embed_dim={embed_dim}  device={DEVICE}")
    return _backbone


def switch_model(key: str, store: "DescriptorStore") -> tuple[bool, "DescriptorStore"]:
    """
    Switch active model weights AND descriptor store at runtime.

    Args:
        key  : one of the keys in AVAILABLE_CONFIGS (case-insensitive)
        store: the currently active DescriptorStore instance
    Returns:
        (success: bool, store: DescriptorStore)
        On success the returned store is bound to the new descriptor file.
        On failure the original store is returned unchanged.
    """
    global _backbone, MODEL_WEIGHTS, DESCRIPTOR_FILE, ACTIVE_MODEL_KEY

    key = key.lower().strip()

    if key not in AVAILABLE_CONFIGS:
        print(f"  [Model] Unknown key '{key}'. "
              f"Available: {list(AVAILABLE_CONFIGS.keys())}")
        return False, store

    if key == ACTIVE_MODEL_KEY:
        print(f"  [Model] '{key}' is already the active model — no change.")
        return True, store

    cfg      = AVAILABLE_CONFIGS[key]
    new_weights = cfg["weights"]
    new_desc    = cfg["descriptors"]

    if not os.path.exists(new_weights):
        print(f"  [Model] Weights file not found: '{new_weights}' — switch aborted.")
        return False, store

    # Unload backbone so get_model() reloads on next inference
    _backbone        = None
    MODEL_WEIGHTS    = new_weights
    DESCRIPTOR_FILE  = new_desc
    ACTIVE_MODEL_KEY = key

    # Reload descriptor store for the new model
    new_store = DescriptorStore(path=new_desc)

    print(f"  [Model] Switched to '{key}'")
    print(f"          Weights     : {new_weights}")
    print(f"          Descriptors : {new_desc}")
    return True, new_store


# =========================
# EMBEDDING
# =========================
@torch.no_grad()
def extract_embedding(frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    aligned = face_align.norm_crop(frame_bgr, landmark=kps, image_size=IMG_SIZE)
    tensor  = preprocess(aligned).unsqueeze(0).to(DEVICE)
    emb     = get_model()(tensor)
    emb     = F.normalize(emb, dim=1)
    return emb.cpu().numpy()[0]


# =========================
# MATCHER
# =========================
def match(query_emb: np.ndarray, store: DescriptorStore):
    identities = store.get_mean_embeddings()
    if not identities:
        return "Unknown", 0.0

    best_name, best_score = "Unknown", -1.0
    skipped = 0
    for name, ref_emb in identities.items():
        if ref_emb.shape != query_emb.shape:
            skipped += 1
            continue
        score = float(np.dot(query_emb, ref_emb))
        if score > best_score:
            best_score, best_name = score, name

    if skipped > 0:
        print(f"  [Match] ⚠️  Skipped {skipped} descriptor(s) — "
              f"dimension mismatch (stored={ref_emb.shape[0]}, "
              f"query={query_emb.shape[0]}). "
              f"Re-enroll under the current model.")

    if best_name == "Unknown" or best_score < THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score


# =========================
# DRAW
# =========================
def draw_box(frame, x1, y1, x2, y2, label, score, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    bar_y1, bar_y2 = y1 - 30, y1
    cv2.rectangle(frame, (x1, bar_y1), (x2, bar_y2), color, -1)
    text = f"{label}  {score:.2f}"
    cv2.putText(frame, text, (x1 + 4, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

def draw_landmarks(frame, kps, color=(0, 255, 255)):
    for pt in kps:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, color, -1)


# =========================
# MENU HELPERS
# =========================
def print_header():
    print("\n" + "=" * 54)
    print("      ArcFace Face Verification System")
    print("      Detection  : RetinaFace (buffalo_l)")
    print(f"      Model      : {ACTIVE_MODEL_KEY}  ({MODEL_WEIGHTS})")
    print(f"      Descriptors: {DESCRIPTOR_FILE}")
    print("=" * 54)

def print_menu():
    cfg = AVAILABLE_CONFIGS[ACTIVE_MODEL_KEY]
    print("\n  [1]  Enroll new person")
    print("  [2]  Recognize faces (live)")
    print("  [3]  List enrolled identities")
    print("  [4]  Delete an identity")
    print(f"  [5]  Switch model  "
          f"(active: {ACTIVE_MODEL_KEY} | weights: {cfg['weights']} | db: {cfg['descriptors']})")
    print("  [0]  Exit")
    print()


# =========================
# ACTION: ENROLL
# =========================
def action_enroll(store: DescriptorStore):
    print("\n--- ENROLL ---")

    while True:
        name = input("  Enter the person's name: ").strip()
        if name:
            break
        print("  Name cannot be empty. Please try again.")

    already = store.db.get(name)
    if already:
        print(f"  '{name}' already has {len(already)} photo(s) enrolled.")
        choice = input("  Add more photos? (y/n): ").strip().lower()
        if choice != 'y':
            return

    print(f"\n  Name set to: '{name}'")
    print("  Loading detector & model...")
    get_detector()
    get_model()

    print("  Opening webcam...")
    print("  ► Press  C  to capture and save")
    print("  ► Press  R  to retake last capture")
    print("  ► Press  E  to finish enrollment\n")

    cap             = cv2.VideoCapture(0)
    captured_count  = 0
    status_msg      = "Align your face and press C"
    status_color    = (255, 200, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        disp  = frame.copy()

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 200, 0), 2)
            if face.kps is not None:
                draw_landmarks(disp, face.kps)

        cv2.rectangle(disp, (0, 0), (disp.shape[1], 42), (30, 30, 30), -1)
        cv2.putText(disp,
                    f"Enrolling: {name}  |  Captured: {captured_count}"
                    f"  |  Faces: {len(faces)}  |  Model: {ACTIVE_MODEL_KEY}",
                    (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        cv2.putText(disp, status_msg,
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1)

        cv2.imshow(f"Enroll: {name}  |  C=capture  R=retake  E=done", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('e'):
            break

        elif key == ord('c'):
            if len(faces) == 0:
                status_msg   = "No face detected — adjust position"
                status_color = (0, 0, 255)
                continue

            best_face = max(faces, key=lambda f: f.det_score)

            if best_face.kps is None:
                status_msg   = "No landmarks detected — try again"
                status_color = (0, 0, 255)
                continue

            embedding = extract_embedding(frame, best_face.kps)
            store.add(name, embedding)
            captured_count += 1
            status_msg      = f"✓ Captured #{captured_count}  (C=more  E=finish)"
            status_color    = (0, 220, 0)
            print(f"  ✓ Captured photo #{captured_count} for '{name}'")

        elif key == ord('r'):
            if name in store.db and store.db[name]:
                store.db[name].pop()
                store.save()
                captured_count  = max(0, captured_count - 1)
                status_msg      = "Last capture removed — try again"
                status_color    = (0, 165, 255)
                print("  ↩ Last capture removed.")
            else:
                status_msg   = "Nothing to retake"
                status_color = (0, 165, 255)

    cap.release()
    cv2.destroyAllWindows()

    if captured_count == 0:
        print(f"  No photos captured for '{name}'.")
    else:
        print(f"\n  ✅ Enrollment complete: '{name}' — {captured_count} photo(s) saved.")


# =========================
# ACTION: RECOGNIZE
# =========================
def action_recognize(store: DescriptorStore):
    print("\n--- RECOGNIZE ---")

    if not store.db:
        print("  No enrolled identities. Please enroll someone first.")
        return

    ids = store.list_identities()
    print(f"  Enrolled  : {', '.join(ids.keys())}")
    print(f"  Model     : {ACTIVE_MODEL_KEY}  ({MODEL_WEIGHTS})")
    print(f"  Descriptors: {DESCRIPTOR_FILE}")
    print("  Loading detector & model...")
    get_detector()
    get_model()
    print("  Opening webcam...  Press E to quit.\n")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)

            if face.kps is not None:
                embedding   = extract_embedding(frame, face.kps)
                name, score = match(embedding, store)
                color       = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                draw_box(frame, x1, y1, x2, y2, name, score, color)
                draw_landmarks(frame, face.kps, color)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, "No landmarks", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (30, 30, 30), -1)
        cv2.putText(frame,
                    f"Model: {ACTIVE_MODEL_KEY}  |  {len(ids)} enrolled"
                    f"  |  Faces: {len(faces)}  |  E=quit",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        cv2.imshow("Face Recognition  |  E to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  Recognition stopped.")


# =========================
# ACTION: LIST
# =========================
def action_list(store: DescriptorStore):
    print("\n--- ENROLLED IDENTITIES ---")
    print(f"  Descriptor file: {store.path}")
    ids = store.list_identities()
    if not ids:
        print("  No identities enrolled yet.")
        return
    print(f"\n  {'Name':<30} {'Photos':>6}")
    print("  " + "-" * 38)
    for name, count in ids.items():
        print(f"  {name:<30} {count:>6}")
    print(f"\n  Total: {len(ids)} identit{'y' if len(ids) == 1 else 'ies'}")


# =========================
# ACTION: DELETE
# =========================
def action_delete(store: DescriptorStore):
    print("\n--- DELETE IDENTITY ---")
    ids = store.list_identities()
    if not ids:
        print("  No identities to delete.")
        return

    print(f"\n  {'Name':<30} {'Photos':>6}")
    print("  " + "-" * 38)
    for n, count in ids.items():
        print(f"  {n:<30} {count:>6}")

    print()
    name = input("  Enter name to delete (or ENTER to cancel): ").strip()
    if not name:
        print("  Cancelled.")
        return

    if name not in store.db:
        print(f"  '{name}' not found.")
        return

    desc_count = len(store.db[name])
    confirm = input(
        f"  Delete '{name}' and all {desc_count} descriptor(s)? (y/n): "
    ).strip().lower()
    if confirm != 'y':
        print("  Cancelled.")
        return

    if store.delete(name):
        remaining = store.list_identities()
        if name not in remaining:
            print(f"  ✅ '{name}' and all {desc_count} descriptor(s) removed.")
            print(f"  Remaining identities: {len(remaining)}")
        else:
            print(f"  ⚠️  Something went wrong — '{name}' still found. Try again.")
    else:
        print(f"  '{name}' not found.")


# =========================
# ACTION: SWITCH MODEL
# =========================
def action_switch_model(store: DescriptorStore) -> DescriptorStore:
    print("\n--- SWITCH MODEL ---")
    print(f"\n  {'Key':<15} {'Weights File':<42} {'Descriptors':<38} {'Status'}")
    print("  " + "-" * 105)
    for key, cfg in AVAILABLE_CONFIGS.items():
        exists = "✓ found  " if os.path.exists(cfg["weights"]) else "✗ missing"
        active = "  ← active" if key == ACTIVE_MODEL_KEY else ""
        print(f"  {key:<15} {cfg['weights']:<42} {cfg['descriptors']:<38} {exists}{active}")

    print()
    choice = input(
        "  Enter model key to activate (or ENTER to cancel): "
    ).strip().lower()

    if not choice:
        print("  Cancelled.")
        return store

    success, new_store = switch_model(choice, store)
    if success:
        print(f"  ✅ Active model is now '{ACTIVE_MODEL_KEY}' "
              f"— will load on next inference.")
        return new_store
    else:
        print("  ❌ Switch failed. Check the key and file path above.")
        return store


# =========================
# MAIN
# =========================
def main():
    print_header()
    store = DescriptorStore(path=DESCRIPTOR_FILE)

    while True:
        print_menu()
        choice = input("  Select option: ").strip()

        if choice == "1":
            action_enroll(store)
        elif choice == "2":
            action_recognize(store)
        elif choice == "3":
            action_list(store)
        elif choice == "4":
            action_delete(store)
        elif choice == "5":
            store = action_switch_model(store)   # rebind on successful switch
        elif choice == "0":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid option. Please enter 1–5 or 0.")


if __name__ == "__main__":
    main()