# WDVAD: Weakly-supervised Dynamic Video Anomaly Detection


## 📌 Overview

- **Task**: Dynamic Video Anomaly Detection (DVAD)
- **Setting**: Weakly supervised learning under a federated learning paradigm
- **Key Idea**: Adapt anomaly semantics that evolve over time after deployment
- **Core Techniques**:
  - Federated learning (Flower framework)
  - MMD-based label self-correction
  - Temporal consistency–aware evaluation

---

## 📁 Repository Structure
WDVAD/
├── README.md # This file
├── ucf_crime/ # Experiments on the UCF-Crime dataset
│ ├── server.py # Federated learning server
│ ├── client.py # Federated learning client
│ ├── model.py # Model definition
│ ├── train.py # Training script
│ ├── test.py # Testing script
│ ├── ablation_study.py # Ablation study (used for paper results)
│ ├── standalone_eval.py # Independent evaluation with checkpoints
│ ├── generate_new_labels.py # MMD-based label self-correction
│ └── README.md # Dataset-specific instructions
│
└── shanghaitech_CADD/ # Experiments on the ShanghaiTech CADD dataset
├── server.py
├── client.py
├── model.py
├── train.py
├── test.py
├── ablation_study.py
├── generate_new_labels.py
└── README.md

Each dataset directory is **self-contained** and includes independent configuration files and usage instructions.

---

## 🎯 Supported Datasets

- **UCF-Crime**
  - Large-scale video anomaly detection benchmark
  - 13 anomaly categories

- **ShanghaiTech**
  - Campus surveillance anomaly detection dataset

 - **Uniform-DVAD(CADD)**

Please refer to the README file in each dataset directory for dataset preparation details.

---

## ⚙️ Environment Setup

Each dataset directory provides its own environment configuration.

Example:
```bash
conda env create -f environment.yml


🚀 Running Experiments

Each dataset supports three execution modes:

1. Federated Training (Main Pipeline)

Run the federated learning server:
python server.py

Run one or multiple clients:

python client.py --cid <client_id>

During training, the system automatically performs MMD-based label self-correction and computes evaluation metrics.


Ablation Study (Paper Results)

python ablation_study.py --cid <client_id> --dataset_type <dataset_type>

Standalone Evaluation

python standalone_eval.py --ckpt_path <path_to_checkpoint>

Anonymity Notice

This repository is released solely for anonymous peer review.

No author-identifying information is included

Code structure and naming are anonymized

Links to personal or institutional resources are intentionally omitted

The repository will be updated with full documentation and attribution upon acceptance.

## 🎯 Pretrained Model
- **UCF-Crime**
https://drive.google.com/file/d/1ppJpGlqSLy12kM2zjacL1h-wFcjCx1Qy/view?usp=sharing

- **ShanghaiTech**
https://drive.google.com/file/d/1Ab_5xiAdQzerJK2Q8ncoixaER8Zz8rqe/view?usp=drive_link

 - **Uniform-DVAD(CADD)**
https://drive.google.com/file/d/1kqvbotadPdO6Cpzt3ldl_JCX8p-KgTJC/view?usp=drive_link

📄 License

This code is released for academic research purposes only.
License details will be provided in the final version.
