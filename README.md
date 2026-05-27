# SSH-CAM: Fine-grained SSH Behavior Identification Framework

SSH-CAM is a curriculum-guided framework for fine-grained SSH behavior identification at encrypted tunnel observation points, designed to accurately infer the dominant SSH behavior in the presence of co-existing interfering behaviors within the captured traffic. The framework constructs packet-level representations encoding both structural attributes and temporal dynamics, followed by sequence-level feature extraction. A Curriculum-Adaptive Mixup mechanism is introduced to progressively increase training difficulty through controlled structural interpolation between behavioral categories. The learned latent representations are further constrained by a Gaussian Mixture Model (GMM) to promote intra-class compactness and inter-class separability under interference conditions.

## Dataset Overview

This repository provides a dataset for fine-grained SSH behavior identification under encrypted tunneling conditions. The dataset covers **10 SSH behavior categories** encapsulated within six representative proxy/tunneling protocols. 

To simulate realistic and challenging network environments, all encapsulated traffic is **end-to-end encrypted** (utilizing TLS or QUIC), making payload inspection infeasible. Multiple SSH behavioral flows may be interleaved within a single tunnel flow due to application-layer multiplexing.

### Traffic Statistics

| Protocol / Tunnel | Total Flows | Train Set | Eval Set | Description & Encryption Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Hysteria2 (`hy2`)** | 741 | 592 | 149 | UDP-based QUIC proxy (Inherently TLS 1.3 encrypted) |
| **Trojan (`tro`)** | 741 | 592 | 149 | TCP-based proxy (TLS encrypted, HTTPS camouflage) |
| **AnyTLS (`anytls`)** | 738 | 590 | 148 | TCP-based custom stealth proxy (TLS encrypted) |
| **HTTPS (`http`)** | 738 | 590 | 148 | Standard proxy over HTTP/1.1 (TLS encrypted) |
| **TUIC (`tuic`)** | 741 | 592 | 149 | UDP-based proxy over QUIC (Inherently TLS 1.3 encrypted) |
| **VMess (`vmess`)** | 741 | 592 | 149 | TCP-based proxy protocol (TLS encrypted payload) |
| **Total (6 Tunnels)** | **4,440** | **3,548** | **892** | 10 SSH behavior categories in total |

> **Note:** The evaluation set provided here serves as the base (zero-interference) traffic. It is used to dynamically generate the mixed-interference (`MI-0` to `MI-25`) evaluation sets for robustness testing.
<img width="2414" height="991" alt="image" src="https://github.com/user-attachments/assets/39874d55-58ba-4373-9d6b-719193fd5abe" />


## 🔍 Overview
SSH-CAM implements a multi-stage training strategy:
1. **Heterogeneous Embedding**: Maps packet length, direction, and timing into a shared latent space via Gated Fusion.
2. **Structural Mixup**: Synthesizes mixed flows using an **Anchor Packet** mechanism to maintain temporal causality.
3. **Curriculum Scheduling**: Progressively increases traffic interference based on the model's training gain ($G$).
4. **Manifold Regularization**: Enforces geometric compactness in the feature space using a Gaussian Mixture Model (GMM).

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

- Python 3.8+
- PyTorch 1.10+
- **CUDA Toolkit** (Recommended)

```bash
pip install torch torchvision torchaudio
pip install numpy tqdm tensorboard scikit-learn
