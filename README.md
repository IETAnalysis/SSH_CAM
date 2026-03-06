# SSH-CAM: Fine-grained SSH Behavior Identification Framework

SSH-CAM (SSH Curriculum-Adaptive Mixup) is an industrial-grade framework designed for identifying specific user behaviors within encrypted SSH tunnels. It addresses real-world network challenges such as background noise, heartbeat packets, and interleaved traffic by combining **Sequence-Aware Structural Interpolation** with **Gain-Based Curriculum Learning**.

## 📑 Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Hyperparameters](#hyperparameters)
7. [Industrial Safeguards](#industrial-safeguards)
8. [Monitoring](#monitoring)

---

## 🔍 Overview
SSH-CAM implements a multi-stage training strategy:
1. **Heterogeneous Embedding**: Maps packet length, direction, and timing into a shared latent space via Gated Fusion.
2. **Structural Mixup**: Synthesizes mixed flows using an **Anchor Packet** mechanism to maintain temporal causality.
3. **Curriculum Scheduling**: Progressively increases traffic interference based on the model's training gain ($G$).
4. **Manifold Regularization**: Enforces geometric compactness in the feature space using a Gaussian Mixture Model (GMM).

---

## 🌟 Key Features
- **Backbone Flexibility**: Toggle between `Transformer` (Global context) and `GRU` (Lightweight sequential) via CLI.
- **Dual Loss Logic**: Choose between `GMM` (Euclidean distance manifold) and `Softmax` (Cross-Entropy).
- **Temporal Rectification**: Non-linear redistribution of inter-packet intervals to handle heavy-tailed network distributions.
- **Industrial Engineering**: 
    - Full CLI parameterization (no hidden defaults).
    - Multi-threaded data pre-fetching.
    - Automated numerical stability guards.

---

## 📂 Project Structure
```text
.
├── train.py             # Main entry point: Training loop & GMM loss logic
├── architecture.py      # Model Factory: Backbones, Fusion, & Temporal Encoders
├── dataset_engine.py    # Data Engine: Structural Mixup & Robust Loading
├── utils.py             # Operations: Curriculum Scheduler & Logging
└── README.md            # Documentation
```
