# :rocket: E(Q)AGNN-PPIS

This is the code repository for the paper E(Q)AGNN-PPIS:Attention Enhanced Equivariant Graph Neural Network for Protein-Protein Interaction Site Prediction

# Introduction

We introduce E(Q)AGNN-PPIS, an equivariant geometric graph neural network architecture that leverages geometric information, designed to focus on PPI site prediction. The proposed E(Q)AGNN-PPIS is the first method to leverage the expressive power of equivariant message passing, incorporating both scalar and vector features, while introducing an attention mechanism to selectively focus on the most relevant features and interactions during message passing in the PPI site prediction task.
![E(Q)AGNN-PPIS_framework](https://github.com/ainimesh/EQAGNN-PPIS/blob/main/Images/Model.png)

# Installation

## System Requirements 

For fast prediction and training process, we recommend using GPU. To use E(Q)AGNN-PPIS with GPU you will need following configuration:

## Virtual Environment Configuration

    1. Python 3.12.8
    2. torch 2.4.0 with cuda 12.1
    3. torch_geometric 2.6.1 with cuda 12.1
    4. numpy
    5. pandas 
    6. biopython 1.84

## Software and database requirement to generate feature yourself
To genertae the features of correspodning PDBs, you need to install the following three software and download the corresponding databases:  
[BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90](https://www.uniprot.org/downloads)  
[HH-suite](https://github.com/soedinglab/hh-suite) and [Uniclust30](https://uniclust.mmseqs.com/)  
[DSSP](https://github.com/cmbi/dssp)  

# How to reproduce our work
1. We provide the datasets, pre-computed features, pre-trained models, and the training and evaluation codes for those interested in reproducing our paper.
2. The datasets used in this study (Train_335, Test_60, Test_315 and UBtest_31) are stored in ./Dataset in fasta format.  
3. The distance maps(L * L) and normalized feature matrixes PSSM(L * 20), HMM(L * 20), DSSP(L * 14), and Atomic Features(L * 7) are stored in ./Feature.
4. The pre-trained E(Q)AGNN-PPIS full model can be found under ./Saved_Models

:star:The training and evaluation codes can be found in [here](https://github.com/ainimesh/EQAGNN-PPIS).

