# Face Recognition System

This project implements a facial recognition security system using ArcFace with a ResNet-18 backbone.  
The workflow includes model training, embedding extraction, evaluation, and face verification using two datasets:

- CASIA-WebFace
- Imperial College Dataset

---

# 📂 Dataset & Model Files

## Datasets

### CASIA-WebFace Dataset
Download:
[CASIA-WebFace Dataset](https://drive.google.com/file/d/138IQPsG54qHdd4YXk0QnLzoXqFrQ4QKy/view?usp=sharing)

Dataset folder:

```text
train_me/
├── train/
├── validation/
└── test/
```

---

### Imperial College Dataset
Download:
[Imperial College Dataset](https://drive.google.com/file/d/1_Jmyw_l3AhjsclfAeiMoTK0-YhXftjH3/view?usp=sharing)

Dataset folder:

```text
imperial_dataset/
├── train/
├── validation/
└── test/
```

---

## Models

Download trained models:

[Models](https://drive.google.com/drive/folders/1Na67zuvW13qQUgwmfrFBJvN27jvdYCNZ?usp=drive_link)

---

# 1. Model Training

## CASIA-WebFace Training

Run:

```bash
python train.py
```

The training script uses:

```text
train_me/
└── CASIA-WebFace/
```

After training, the best weights are saved:

```text
checkpoints/
└── best_model.pth
```

---

## Imperial Dataset Training

Run:

```bash
python train.py
```

using:

```text
imperial_dataset/
├── train/
├── validation/
└── test/
```

The trained model will also generate:

```text
checkpoints/
└── best_model.pth
```

---

# 2. Extract Face Embeddings

Run:

```bash
python Infer.py
```

The script extracts face embeddings from the training set.

Generated embeddings are used as enrolled face representations.

---

# 3. Testing & Evaluation

Run:

```bash
python testing.py
```

The system compares:

```
Training Set Embeddings
          |
          v
     Test Set Images
```

Evaluation metrics:

| Metric | Description |
|---|---|
| Accuracy | Recognition performance |
| FAR | False Acceptance Rate |
| FRR | False Rejection Rate |
| ROC | Receiver Operating Characteristic |
| EER | Equal Error Rate |

---

# 4. Facial Security System

Run:

```bash
Face_System
```

The system must be tested using both models:

| Dataset Model | JSON Descriptor |
|---|---|
| CASIA-WebFace | `face_descriptors_casia.json` |
| Imperial Dataset | `face_descriptors_imperial.json` |

Generated files:

```text
face_descriptors_casia.json
face_descriptors_imperial.json
```

These JSON files store facial descriptors used for identity verification.

---

# Workflow

```text
                 CASIA-WebFace
                       |
                       v
                  train.py
                       |
                       v
              best_model_casia.pth
                       |
                       v
                   Infer.py
                       |
                       v
              Face Embeddings


                 Imperial Dataset
                       |
                       v
                  train.py
                       |
                       v
           best_model_imperial.pth
                       |
                       v
                   Infer.py
                       |
                       v
              Face Embeddings


                       |
                       v

                 testing.py
                       |
                       v
              Performance Evaluation


                       |
                       v

                 Face_System
                       |
        +--------------+--------------+
        |                             |
        v                             v
face_descriptors_casia.json   face_descriptors_imperial.json
```
Ps.: In the research, Casia-WebFace uses m = 0.25, while Imperial College Dataset uses m = 0.50. m is the margin of penalty for ArcFace, which is the parameters for promoting inter-class separation and intra-class compactness
