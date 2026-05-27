# SSH-CAM: Fine-Grained SSH Behavior Identification in Encrypted Tunnel Traffic using Curriculum-Adaptive Mixup

Encrypted tunnels protect privacy but obscure application-layer semantics, complicating fine-grained traffic analysis. When SSH traffic is encapsulated within tunnels, multiple behaviors coexist within a single flow, yet existing tunnel analysis methods target only protocol- or application-level identification. We present SSH-CAM, a curriculum-guided framework for inferring dominant SSH behaviors at encrypted tunnel observation points under coexisting interference. SSH-CAM constructs packet-level representations encoding structural and temporal attributes, applies sequence-level feature extraction, and introduces a Curriculum-Adaptive Mixup mechanism that progressively increases training difficulty via structural interpolation. A learnable Gaussian prototype constraint further enforces intra-class compactness and inter-class separation. Experiments across six tunneling protocols demonstrate that SSH-CAM consistently outperforms baselines under varying interference levels.

## Dataset Overview

This repository provides a dataset for fine-grained SSH behavior identification under encrypted tunneling conditions. The dataset covers **10 SSH behavior categories** encapsulated within six representative proxy/tunneling protocols. 

To simulate realistic and challenging network environments, all encapsulated traffic is **end-to-end encrypted** (utilizing TLS or QUIC), making payload inspection infeasible. Multiple SSH behavioral flows may be interleaved within a single tunnel flow due to application-layer multiplexing.

📥 **Dataset Download Link**: [Google Drive (SSH Encrypted Tunneling Dataset)](https://drive.google.com/drive/folders/1hC_5yCFaCc0fsvpzmmWHpyqGe9Y2Nkki?usp=sharing)

---

### Taxonomy of SSH Encrypted Traffic Behaviors

The 10 SSH behavior categories are meticulously designed to cover typical interactive and automated tasks, categorized into three main groups:

| Category | Behavior Type | Description | Traffic Pattern |
| :--- | :--- | :--- | :--- |
| **Administrative Querying** | (1) System Status | Retrieval of basic metadata such as user identity, OS version, and runtime status. | Extremely short duration; minimal data volume; predominantly request-response. |
| | (2) Network Topology | Enumeration of network interfaces, routing tables, and connection states. | Short duration; response payload slightly larger than system status. |
| | (3) File System Enumeration | Browsing directory structures and querying file attributes or disk usage. | Interactive output; highly structured burst patterns. |
| | (4) Configuration Access | Reading system or application configurations to locate key parameters/credentials. | Small-scale text transmission; output exhibits distinct semantic structure. |
| **Data Manipulation** | (5) Data Exfiltration | Transferring files from the target system to a remote server. | Sustained outbound data transmission; significant directionality. |
| | (6) Tool Implantation | Uploading files or payloads to the target system. | Sustained inbound data transmission; significant directionality. |
| | (7) Bulk Output | Direct output of large-scale files or log contents. | Continuous data output; relatively smooth transmission rate. |
| **Persistent Monitoring** | (8) Resource Monitoring | Periodic retrieval of system performance metrics. | Long duration; periodic or quasi-periodic small data bursts. |
| | (9) Log Auditing | Continuous listening for new entries in system or application logs. | Long duration; event-driven data output; irregular inter-arrival times. |
| | (10) Interactive Session | Maintaining a persistent interactive shell environment. | Long duration; significant bidirectional interaction; irregular traffic bursts. |

---

### Traffic Statistics

The tunnel distribution are detailed below:

| Protocol / Tunnel | Total Flows | Train Set | Eval Set | Description & Encryption Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Hysteria2 (`hy2`)** | 741 | 592 | 149 | UDP-based QUIC proxy (Inherently TLS 1.3 encrypted) |
| **Trojan (`tro`)** | 741 | 592 | 149 | TCP-based proxy (TLS encrypted, HTTPS camouflage) |
| **AnyTLS (`anytls`)** | 738 | 590 | 148 | TCP-based custom stealth proxy (TLS encrypted) |
| **HTTPS (`http`)** | 738 | 590 | 148 | Standard proxy over HTTP/1.1 (TLS encrypted) |
| **TUIC (`tuic`)** | 741 | 592 | 149 | UDP-based proxy over QUIC (Inherently TLS 1.3 encrypted) |
| **VMess (`vmess`)** | 741 | 592 | 149 | TCP-based proxy protocol (TLS encrypted payload) |
| **Total (6 Tunnels)** | **4,440** | **3,548** | **892** | 10 SSH behavior categories in total |

---



## 📂 Project Structure
```text
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
