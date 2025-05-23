import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
To process input fasta file and generating the required information

"""

default_path = "/home/animesh/PhD/ainimesh/PBSP/GPBSP/Dataset/Train_335_clean.fa"

# function to peeform the required task

def processing_fasta_file(file_path = default_path):

    """
    input: fasta file as input
    Output: pdb_ids, sequences, labels, input_files
    
    """

    # Initialize lists to store data
    pdb_ids = []
    sequences = []
    labels = []
    input_files = []

    # Read the custom FASTA-like file
    with open(file_path, "r") as file:
        lines = file.readlines()

        # Process the lines in groups of three
        for i in range(0, len(lines), 3):
            # Extract PDB ID with chain from the first line
            header_line = lines[i].strip()
            pdb_id_with_chain = header_line[1:]  # Remove the '>' character
            f_name = f"{pdb_id_with_chain[:4]}_{pdb_id_with_chain[-1]}.pdb"
            input_files.append(f_name)

            # Extract the amino acid sequence from the second line
            sequence_line = lines[i + 1].strip()

            # Extract labels per amino acid from the third line
            labels_line = lines[i + 2].strip()

            # Append the extracted data to the respective lists
            pdb_ids.append(pdb_id_with_chain)
            sequences.append(sequence_line)
            labels.append(labels_line)

    return pdb_ids, sequences, labels, input_files


#! -----------------------------------PMI BASED LOSS----------------------------
# CUSTOM FUNCTION FOR PMI BASED LOSS
def compute_priors(num1, num2, device):
    y_prior = torch.log(torch.tensor([num1+1e-8, num2+1e-8], requires_grad = False)).to(device)
    return y_prior

#! --------------------------------------  FoCAL Loss -------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(inputs, targets),# reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# # Example usage
# focal_loss = FocalLoss(alpha=1, gamma=2)

#!----------------------------------NORMALIZATION LAYER-----------------------------------
        
class NormLayer(nn.Module):
    def __init__(self, norm_mode, norm_scale):
        """
            mode:
              'None' : No normalization
              'PN'   : PairNorm
              'PN-SI'  : Scale-Individually version of PairNorm
              'PN-SCS' : Scale-and-Center-Simultaneously version of PairNorm
              'LN': LayerNorm
              'CN': ContraNorm
        """
        super(NormLayer, self).__init__()
        self.mode = norm_mode
        self.scale = norm_scale

    def forward(self, x, adj=None, tau=1.0):
        if self.mode == 'None':
            return x
        if self.mode == 'LN':
            x = x - x.mean(dim=1, keepdim=True)
            x = nn.functional.normalize(x, dim=1)

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'CN':
            norm_x = nn.functional.normalize(x, dim=1)
            sim = norm_x @ norm_x.T / tau
            # if adj.size(1) == 2:
            #     sim[adj[0], adj[1]] = -np.inf
            # else:
            sim.masked_fill_(adj > 1e-5, -np.inf)
            sim = nn.functional.softmax(sim, dim=1)
            x_neg = sim @ x
            x = (1 + self.scale) * x - self.scale * x_neg

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


#! ---------------------------------------------------------------------------------------