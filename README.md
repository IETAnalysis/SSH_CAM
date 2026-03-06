# SSH-CAM: Fine-grained SSH Behavior Identification Framework

SSH-CAM is a curriculum-guided framework for fine-grained SSH behavior identification at encrypted tunnel observation points, designed to accurately infer the dominant SSH behavior in the presence of co-existing interfering behaviors within the captured traffic. The framework constructs packet-level representations encoding both structural attributes and temporal dynamics, followed by sequence-level feature extraction. A Curriculum-Adaptive Mixup mechanism is introduced to progressively increase training difficulty through controlled structural interpolation between behavioral categories. The learned latent representations are further constrained by a Gaussian Mixture Model (GMM) to promote intra-class compactness and inter-class separability under interference conditions.


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
├── main.py            
├── model.py      
├── dataset.py    
├── utils.py             
└── README.md           
```

## 🛠️ Installation
Prerequisites
- **Python 3.8+
- **PyTorch 1.10+
- **CUDA Toolkit (Recommended)
Setup
```bash
pip install torch torchvision torchaudio
pip install numpy tqdm tensorboard scikit-learn
```

