# :rocket: E(Q)AGNN-PPIS

This is the code repository for the paper E(Q)AGNN-PPIS:Attention Enhanced Equivariant Graph Neural Network for Protein-Protein Interaction Site Prediction

# :bulb: Introduction

We introduce E(Q)AGNN-PPIS, an equivariant geometric graph neural network architecture that leverages geometric information, designed to focus on PPI site prediction. The proposed E(Q)AGNN-PPIS is the first method to leverage the expressive power of equivariant message passing, incorporating both scalar and vector features, while introducing an attention mechanism to selectively focus on the most relevant features and interactions during message passing in the PPI site prediction task.
![E(Q)AGNN-PPIS_framework](https://github.com/ainimesh/EQAGNN-PPIS/blob/main/Images/Model.png)

## :package: Installation

### :computer: System Requirements 

For fast prediction and training process, we recommend using GPU. To use E(Q)AGNN-PPIS with GPU you will need following configuration:

### :gear: Virtual Environment Configuration

:snake: **Python Environment:**
1. Python 3.12.8
2. torch 2.4.0 with cuda 12.1
3. torch_geometric 2.6.1 with cuda 12.1
4. numpy
5. pandas 
6. biopython 1.84

## :hammer_and_wrench: Software and Database Requirements to produce the features yourself
To genertae the features of correspodning PDBs, you need to install the following three software and download the corresponding databases:  
[BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90](https://www.uniprot.org/downloads)  
[HH-suite](https://github.com/soedinglab/hh-suite) and [Uniclust30](https://uniclust.mmseqs.com/)  
[DSSP](https://github.com/cmbi/dssp)  

## :page_facing_up: How to Reproduce Our Work

:white_check_mark: We provide the datasets, pre-computed features, pre-trained models, and the training and evaluation codes for those interested in reproducing our paper.

### :file_folder: Dataset Structure

1. :books: **Datasets:** The datasets used in this study (Train_335-1, Test_60_0, Test_315-28 and UBtest_31-6) are stored in `./main/Dataset` in fasta format.
2. :bar_chart: **Features:** The distance maps(L × L) and normalized feature matrices PSSM(L × 20), HMM(L × 20), DSSP(L × 14), and Atomic Features(L × 7) are stored in `./main/Features`.
3. :gem: **Pre-trained Models:** The pre-trained E(Q)AGNN-PPIS full model can be found under `./model_trained`

### :star: Getting Started

The training and evaluation codes can be found in [here](https://github.com/ainimesh/EQAGNN-PPIS).

### :rocket: How to Run Test Files for Reproduction

**Step 1:** Navigate to project root
```bash
cd /path/to/EQAGNN-PPIS
```

**Step 2:** Run test script with your chosen dataset
```bash
# For Test_60 dataset
python test.py --dataset test_60

# For Test_315 dataset  
python test.py --dataset test_315

# For UBtest_31 dataset
python test.py --dataset ubtest_31
```

# :page_facing_up: citation

You can cite our work as:  
```bibtex
@article{animesh2024q,
  title={E(Q)AGNN-PPIS: Attention Enhanced Equivariant Graph Neural Network for Protein-Protein Interaction Site Prediction},
  author={Animesh and Suvvada, Rishi and Bhowmick, Plaban Kumar and Mitra, Pralay},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

Contact:  
Animesh (animesh.sachan24794@kgpian.iitkgp.ac.in)
