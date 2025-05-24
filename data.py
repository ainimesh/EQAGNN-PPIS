import pickle, os
import torch
import numpy as np
from utils import processing_fasta_file
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


import warnings
warnings.filterwarnings("ignore")

# Paths
Feature_Path = './main/Feature/'
Dataset_Path = './main/Dataset/'

## reading the file 
# train_332
pdb_ids_train, sequences_train, labels_train, input_files_train = processing_fasta_file(Dataset_Path + 'Train_332.fa')

# test_60
pdb_ids_test, sequences_test, labels_test, input_files_test = processing_fasta_file(Dataset_Path + 'Test_60.fa')

# test_315
pdb_ids_test_315, sequences_test_315, labels_test_315, input_files_test_315 = processing_fasta_file(Dataset_Path + 'Test_315.fa')


# defining a function to load features as per the requirements

def load_features(ID, Feature_Path, seq = False, pbert = False, pstruct = False, patom = False, all_feat = True):
    """
    #1. PSSM - sequential features 20D
    #2. HMM - sequential features 20D
    #3. DSSP - for structural features 14D
    #4. resAF - residue atomic features 7D
    #5. prot_bert - amino acid features generated from protein bert model 1024D

    seq = pssm+hmm (40)
    pbert = protein bert generated features (1024)
    patomic = seq + resAF (47)
    pstruct = seq + dssp (54)
    all_feat = seq + DSSP + resAF + pbert (61 + 1024)

    """

    # conditions to load the features as per the experiments
    pssm_feature = np.load(Feature_Path + "pssm/" + ID + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + ID + '.npy')
    dssp_feature = np.load(Feature_Path + "dssp/" + ID + '.npy')
    res_atom_feature = np.load(Feature_Path + "resAF/" + ID + '.npy')
    
    if pbert == False:

        if seq == True:
            node_features = np.concatenate([pssm_feature, hmm_feature], axis=1)
        elif patom == True:
            node_features = np.concatenate([pssm_feature, hmm_feature, res_atom_feature], axis=1)
        elif pstruct == True:
            node_features = np.concatenate([pssm_feature, hmm_feature, dssp_feature], axis=1)
        elif all_feat == True:
            node_features = np.concatenate([pssm_feature, hmm_feature, dssp_feature, res_atom_feature], axis=1)
    
    elif pbert == True:
        prot_bert_feature = np.load(Feature_Path + "prot_bert/" + ID + '.npy')
        if seq == True:
            node_features = np.concatenate([pssm_feature, hmm_feature, prot_bert_feature], axis=1)
        elif patom == True:
            node_features = np.concatenate([pssm_feature, hmm_feature, prot_bert_feature, res_atom_feature], axis=1)
        elif pstruct == True:
            node_features = np.concatenate([pssm_feature, hmm_feature, prot_bert_feature, dssp_feature], axis=1)
        elif all_feat == True:
            node_features = np.concatenate([pssm_feature, hmm_feature, prot_bert_feature, dssp_feature, res_atom_feature], axis=1)
    
    # returning the appropriate value
    return node_features.astype(np.float32)

def adj_matrix_to_edge_list(path,ID,threshold = 14):
    """
    Converts an adjacency matrix to an edge list.

    Args:
        adj_matrix: An adjacency matrix.

    Returns:
        A list of edges.
    """
    adj_matrix = torch.load(os.path.join(path,ID))

    edges = [[],[]]
    for i in range(adj_matrix.shape[0]):
        for j in range(i + 1, adj_matrix.shape[1]):
            if adj_matrix[i][j]<=threshold:
                edges[0].append(i)
                edges[1].append(j)
                #edges.append((i, j))

    return edges




class ProDataset(Dataset):
    def __init__(self, pdb_ids, sequences, labels, threshold, Res_Position_Path, Adj_path, Feat_path,
                seq = False, pbert = False, pstruct = False, patom = False, all_feat = True):
        self.IDs = pdb_ids
        self.labels = labels
        self.threshold = threshold
        self.sequences = sequences
        self.seq = seq
        self.pbert = pbert
        self.pstruct = pstruct
        self.patom = patom
        self.all_feat = all_feat
        self.Res_Position_Path = Res_Position_Path
        self.Adj_path = Adj_path
        self.Feat_path = Feat_path
        self.residue_pos = pickle.load(open(self.Res_Position_Path, 'rb'))
        self.dist = 15
    
    def __getitem__(self, index):

        ID = self.IDs[index]
        label = self.labels[index]
        label = np.array([int(i) for i in list(label)])
        sequence = self.sequences[index]
        # nodes_num = len(sequence)

        # to calculate edge features
        res_pos = self.residue_pos[ID]
        reference_res_psepos = res_pos[0]
        pos = res_pos - reference_res_psepos
        pos = torch.from_numpy(pos)


        # to load features and edge indexes
        node_features = load_features(ID, self.Feat_path, seq = self.seq, pbert = self.pbert, 
                                    pstruct = self.pstruct, patom = self.patom, all_feat = self.all_feat)
        
        node_features = torch.from_numpy(node_features)
        
        #!--- adding Pseudo-position embedding feature [PPEF]

        ppef = torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist
        node_features = torch.cat([node_features, ppef], dim=-1)
        # print(node_feature.shape)
        
        #!--

        # load edge indexes
        edge_indexes = adj_matrix_to_edge_list(self.Adj_path, f"{ID[:4]}_{ID[-1]}.adj.pt",
                                                        threshold = self.threshold)
        
        edge_index = torch.LongTensor(edge_indexes)
        coords = torch.FloatTensor(res_pos)
        y = torch.LongTensor(label)

        data = Data(atoms=node_features, edge_index=edge_index, pos=coords, y=y)
        data.edge_index = to_undirected(data.edge_index)

        return data

    def __len__(self):
        return len(self.labels)