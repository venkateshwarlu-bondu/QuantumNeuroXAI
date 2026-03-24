[![DOI](https://zenodo.org/badge/1190271986.svg)](https://doi.org/10.5281/zenodo.19200650)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

---

## Code Archive

GitHub repository: https://github.com/venkateshwarlu-bondu/QuantumNeuroXAI

Permanent archived release (Zenodo DOI): https://doi.org/10.5281/zenodo.19200650
# QuantumNeuroXAI

### An AI-Driven Hybrid Quantum Machine Learning Framework for EEG-Based Neurological Disorder Detection with Multi-Level Explainability

---

## Overview

**QuantumNeuroXAI** is a hybrid quantum-inspired and deep learning framework designed for robust and explainable EEG signal analysis. The system integrates classical deep learning models with quantum-inspired feature encoding to enable accurate neurological disorder detection and brain–computer interface (BCI) classification.

The framework provides **multi-level explainability**, offering insights at the signal, model, and quantum representation levels, making it suitable for both research and clinical applications.

---

## Key Features

* Hybrid Quantum + Deep Learning Architecture
* Support for multiple EEG datasets
* Time–frequency feature extraction (STFT-based)
* Attention-based temporal modeling (GRU/LSTM)
* Multi-task classification (binary and multi-class)
* Multi-level Explainability

  * Signal-level (channels, frequency bands)
  * Model-level (attention weights)
  * Quantum-level (feature sensitivity)
* Modular and extensible pipeline
* Ablation and cross-dataset evaluation support

---

## System Architecture

The framework follows a dual-branch architecture:

```
EEG Signal
   ↓
Preprocessing (Filtering + Segmentation + STFT)
   ↓
Tensor Representation [C × F × T]
   ↓
 ┌───────────────────────────────┐
 │       Dual Processing         │
 │                               │
 │  Quantum Branch               │
 │  - Amplitude Encoding         │
 │  - Phase Encoding             │
 │  - Feature Mapping            │
 │  - Measurement                │
 │                               │
 │  Classical Branch             │
 │  - Temporal CNN               │
 │  - Attention GRU/LSTM         │
 └──────────────┬────────────────┘
                ↓
         Fusion Layer
                ↓
         Prediction Head
                ↓
        Explainability Engine
```

---

## Repository Structure

```
QuantumNeuroXAI/
│
├── configs/                 # YAML configurations
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Preprocessed tensors
│   └── manifests/           # Metadata CSVs
│
├── src/
│   ├── datasets/            # Dataset loaders
│   ├── preprocessing/       # Signal processing
│   ├── quantum/             # Quantum-inspired modules
│   ├── models/              # Deep learning models
│   ├── training/            # Training & evaluation
│   ├── explainability/      # XAI modules
│   └── utils/               # Utilities
│
├── scripts/                 # Execution scripts
├── outputs/                 # Results and logs
├── notebooks/               # Experiments
│
├── requirements.txt
├── run.py
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/venkateshwarlu-bondu/QuantumNeuroXAI.git
cd QuantumNeuroXAI
```

Create environment:

```
conda create -n qnxai python=3.10
conda activate qnxai
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Dataset Preparation

Supported datasets:

* CHB-MIT → Seizure Detection
* BCI IV-2a → Motor Imagery Classification
* TUH EEG → Large-scale EEG analysis

Place datasets in:

```
data/raw/chbmit/
data/raw/bci2a/
data/raw/tuh_eeg/
```

---

## Quick Start (Demo Mode)

Run with synthetic data:

```
python scripts/00_make_synthetic_demo.py --dataset chbmit --n 20
```

Then execute:

```
python scripts/01_build_manifests.py --dataset chbmit
python scripts/02_preprocess_dataset.py --dataset chbmit --config configs/dataset_chbmit.yaml
python scripts/03_train_baseline.py --dataset chbmit \
  --config configs/global.yaml \
  --dataset-config configs/dataset_chbmit.yaml \
  --model-config configs/model_baseline.yaml
```

---

## Full Pipeline

1. Build dataset manifest

```
python scripts/01_build_manifests.py --dataset chbmit
```

2. Preprocess EEG data

```
python scripts/02_preprocess_dataset.py \
  --dataset chbmit \
  --config configs/dataset_chbmit.yaml
```

3. Train baseline model

```
python scripts/03_train_baseline.py
```

4. Train QuantumNeuroXAI model

```
python scripts/04_train_quantumneuroxai.py
```

5. Evaluation

```
python scripts/05_run_evaluation.py
```

6. Explainability

```
python scripts/06_generate_explanations.py
```

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* AUROC
* Sensitivity / Specificity
* Macro-F1

---

## Explainability Modules

### Signal-Level

* Channel importance
* Frequency band relevance
* Time-region saliency

### Model-Level

* Attention weight visualization
* Feature contribution analysis

### Quantum-Level

* Feature sensitivity
* Importance ranking

---

## Ablation Study

Run:

```
python scripts/07_run_ablation.py
```

Evaluates:

* CNN only
* CNN + Attention
* Quantum only
* Quantum + CNN
* Full Hybrid Model

---

## Model Output

Example:

```
{
  "prediction": 1,
  "probability": 0.92,
  "attention_weights": [...],
  "quantum_features": [...],
  "explanations": {...}
}
```

---

## Performance Highlights

* Improved AUROC and F1-score over classical baselines
* Reduced false positives
* Better cross-dataset generalization
* Enhanced interpretability

---

## Reproducibility

To reproduce experiments:

```
python run.py --config configs/global.yaml
```

---

## Future Work

* Real quantum hardware integration (Qiskit / Pennylane)
* Transformer-based EEG modeling
* Multimodal learning (EEG + clinical data)
* Real-time deployment systems

---

## Citation
If you are using this repository or the corresponding research work, please cite the following paper:
```
@article{QuantumNeuroXAI2026,
title={QuantumNeuroXAI: A Hybrid Quantum Machine Learning Framework for EEG-Based Neurological 
}
```

---

## License

This project is licensed under the MIT License.
See the LICENSE file for details.

---

