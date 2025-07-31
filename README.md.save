# 🚀 E(Q)AGNN-PPIS

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-IEEE_TSIPN%202025-blue)](https://www.biorxiv.org/content/10.1101/2024.10.06.616807v2)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

This is the official implementation of **E(Q)AGNN-PPIS: Attention Enhanced Equivariant Graph Neural Network for Protein-Protein Interaction Site Prediction**

## 📑 Table of Contents
- [Introduction](#-introduction)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dataset Structure](#-dataset-structure)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Pre-trained Models](#-pre-trained-models)
- [Citation](#-citation)
- [Contact](#-contact)

## 💡 Introduction

We introduce E(Q)AGNN-PPIS, an equivariant geometric graph neural network architecture that leverages geometric information, designed to focus on PPI site prediction. The proposed E(Q)AGNN-PPIS is the first method to leverage the expressive power of equivariant message passing, incorporating both scalar and vector features, while introducing an attention mechanism to selectively focus on the most relevant features and interactions during message passing in the PPI site prediction task.

![E(Q)AGNN-PPIS_framework](https://github.com/ainimesh/EQAGNN-PPIS/blob/main/Images/Model.png)

## 📦 Installation

### 💻 System Requirements 

For fast prediction and training process, we recommend using GPU with CUDA 12.1 support.

### 🔧 Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/ainimesh/EQAGNN-PPIS.git
cd EQAGNN-PPIS
```

2. **Create a virtual environment**
```bash
python -m venv eqagnn_env
source eqagnn_env/bin/activate  # On Windows: eqagnn_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### ⚙️ Dependencies

- Python 3.12.8
- PyTorch 2.4.0 with CUDA 12.1
- PyTorch Geometric 2.6.1 with CUDA 12.1
- NumPy
- Pandas 
- BioPython 1.84
- tqdm
- scikit-learn

### 🛠️ Optional: Feature Generation Requirements

To generate features for your own PDB files, you'll need:
- [BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90](https://www.uniprot.org/downloads)  
- [HH-suite](https://github.com/soedinglab/hh-suite) and [Uniclust30](https://uniclust.mmseqs.com/)  
- [DSSP](https://github.com/cmbi/dssp)  

## 🚀 Quick Start

### Testing with Pre-trained Models

```bash
# Test on Test_60 dataset
python test.py --dataset test_60

# Test on Test_315 dataset  
python test.py --dataset test_315

```

### Training from Scratch

```bash
# Basic training
python train.py --epochs 50 --val_dataset test_60

# Training with custom parameters
python train.py --epochs 100 --val_dataset test_315 --lr 0.001 --batch_size 2

# Resume training from checkpoint
python train.py --resume checkpoints/EQAGNN_latest.pt --epochs 20
```

## 📁 Dataset Structure

```
EQAGNN-PPIS/
├── main/
│   ├── Dataset/           # Fasta files
│   │   ├── Train_332.fa   # Training set
│   │   ├── Test_60.fa     # Test set 1
│   │   ├── Test_315.fa    # Test set 2
│   │   └── UBtest.fa      # Ubiquitin test set
│   ├── Features/          # Pre-computed features
│   │   ├── PSSM/         # L × 20 matrices
│   │   ├── HMM/          # L × 20 matrices
│   │   ├── DSSP/         # L × 14 matrices
│   │   └── Atomic/       # L × 7 matrices
│   ├── Input_adj/        # Adjacency matrices
│   └── Res_positions/    # Residue position files
├── model_trained/        # Pre-trained models
└── results/              # Evaluation results
```

### 📊 Dataset Statistics

| Dataset | Proteins | Residues | Positive | Negative | Ratio |
|---------|----------|----------|----------|----------|-------|
| Train_332 | 335 | 82,932 | 17,473 | 65,459 | 1:3.7 |
| Test_60 | 60 | 14,842 | 3,360 | 11,482 | 1:3.4 |
| Test_315 | 315 | 78,849 | 16,764 | 62,085 | 1:3.7 |
| UBtest_31 | 31 | 7,636 | 1,523 | 6,113 | 1:4.0 |

## 🏋️ Training

### Basic Training

```bash
python train.py --epochs 50 --val_dataset test_60
```

### Advanced Training Options

```bash
python train.py \
    --epochs 100 \
    --val_dataset test_315 \
    --lr 0.0005 \
    --batch_size 2 \
    --num_workers 4 \
    --early_stopping 10 \
    --exp_name my_experiment \
    --model_dir my_models \
    --results_dir my_results
```

## 📊 Evaluation

### Testing Individual Datasets

```bash
# Test specific dataset
python test.py --dataset test_60 --model_path path/to/model.pt

# Test with custom batch size
python test.py --dataset test_315 --batch_size 2 --num_workers 4
```

## 🎯 Pre-trained Models

We provide pre-trained models for each test dataset:

| Model | Test Dataset | Download |
|-------|--------------|----------|
| Best_EQAGNNModel_test_60.pt | Test_60 | [Download](model_trained/saved_models/Best_EQAGNNModel_test_60.pt) |
| Best_EQAGNNModel_test_315.pt | Test_315 | [Download](model_trained/saved_models/Best_EQAGNNModel_test_315_&_60.pt) |
| Best_EQAGNNModel_Ubtest.pt | UBtest | [Download](model_trained/saved_models/Best_EQAGNNModel_Ubtest.pt) |


### Batch Processing

For processing multiple experiments:

```bash
# Run multiple experiments with different parameters
bash run_experiments.sh
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{animesh2024eqagnn,
  title={E(Q)AGNN-PPIS: Attention Enhanced Equivariant Graph Neural Network for Protein-Protein Interaction Site Prediction},
  author={Animesh and Suvvada, Rishi and Bhowmick, Plaban Kumar and Mitra, Pralay},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## 👥 Contact

**Animesh**  
Email: animesh.sachan24794@kgpian.iitkgp.ac.in  
Indian Institute of Technology Kharagpur

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the authors of the following tools and databases used in this work:
- BLAST+ and UniRef90
- HH-suite and Uniclust30
- DSSP
- [PyG](https://github.com/pyg-team/pytorch_geometric)

---

<p align="center">
  Made with ❤️ at IIT Kharagpur.
</p>
