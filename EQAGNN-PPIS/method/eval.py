import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import e3nn
from functools import partial
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import sys
sys.path.append('/home/rishi/rishi_mtp/main_code')

from .train_utils import run_experiment,eval
# from model import EGNNModel
from model import GVPGNNModel
from model import GVP_AAMPModel

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f"Using device: {device}")


#Changing os directory
os.chdir("/home/rishi/rishi_mtp")


####################################################################  DATA PREPROCESSING  #############################################################################

#Hyperparams for data importing
DIST_NORM = 15

file_test315_28 = 'final_feats/Test315-28_psepos_CA.pkl'
file_UBtest31_6 = 'final_feats/UBtest31-6_psepos_CA.pkl'

#load pickle file in a variable called psepos
import pickle

with open(file_test315_28, 'rb') as f:
    psepos_test315_28 = pickle.load(f)

with open(file_UBtest31_6, 'rb') as f:
    psepos_UBtest31_6 = pickle.load(f)


def adj_matrix_to_edge_list(adj_matrix):
  """
  Converts an adjacency matrix to an edge list.

  Args:
    adj_matrix: An adjacency matrix.

  Returns:
    A list of edges.
  """

  edges = [[],[]]
  for i in range(adj_matrix.shape[0]):
    for j in range(i + 1, adj_matrix.shape[1]):
      if adj_matrix[i][j]<=14:
        edges[0].append(i)
        edges[1].append(j)
        #edges.append((i, j))

  return edges


def load_data (pdb_list,label_list,what='test315'):
  dataset = []
  for i,protein in tqdm(enumerate(pdb_list)):
    try:
      pssm = np.load('final_feats/pssm/'+protein +'.npy')
      hmm = np.load('final_feats/hmm/'+protein +'.npy')
      dssp = np.load('final_feats/dssp/'+protein +'.npy')
      resaf = np.load('final_feats/resAF/'+protein +'.npy')
      
      
      #prot_bert = np.load('final_feats/prot_bert/'+protein +'.npy')
      
      


      if what=='test315-28':
        cords = psepos_test315_28[protein]#np.load('/content/drive/MyDrive/rishi_mtp/final_feats/cords/'+protein +'_cords.npy')
      elif what=='UBtest31-6':
        cords = psepos_UBtest31_6[protein]

      p = torch.FloatTensor(cords - cords[0])
      #print(p.shape)
      #psepos = np.sqrt(np.sum(p * p, dim=1)).reshape(-1) / DIST_NORM




      adj = np.load('final_feats/distance_map/'+protein +'.npy')

      #print(pssm.shape,hmm.shape,dssp.shape,cords.shape,adj.shape)

      #atoms = torch.FloatTensor(prot_bert)
      atoms = torch.FloatTensor(np.concatenate([pssm, hmm, dssp, resaf], axis = 1).astype(np.float64))
      atoms = torch.cat([atoms, torch.sqrt(torch.sum(p * p, dim=1)).unsqueeze(-1) / DIST_NORM], dim=-1)

      edge_index = torch.LongTensor(np.array(adj_matrix_to_edge_list(adj)))
      pos = torch.FloatTensor(cords)
      y = torch.LongTensor(label_list[i])

      data1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
      data1.edge_index = to_undirected(data1.edge_index)
      dataset.append(data1)
    except:
      print(protein)
  return dataset


test315_28_obj = pd.read_pickle(r'Dataset/Test_315-28.pkl')
test315_28_obj = pd.DataFrame(test315_28_obj)

UBtest31_6_obj = pd.read_pickle(r'Dataset/UBtest_31-6.pkl')
UBtest31_6_obj = pd.DataFrame(UBtest31_6_obj)

test315_28_dataset = load_data (list(test315_28_obj.columns),list(test315_28_obj.iloc[1]),'test315-28')
UBtest31_6_dataset = load_data (list(UBtest31_6_obj.columns),list(UBtest31_6_obj.iloc[1]),'UBtest31-6')

test315_28loader = DataLoader(test315_28_dataset, batch_size=1, shuffle=False)
UBtest31_6loader = DataLoader(UBtest31_6_dataset, batch_size=1, shuffle=False)

######################################################################################################################################################################

#Loading saved models
saved_model_path = 'saved_models/gvpatt_best-model-parameters_24.pt'
model_name = "gvp_att"
model = {
    "gvp": GVPGNNModel,
    "gvp_att": GVP_AAMPModel,
}[model_name](num_layers=8, in_dim=62, out_dim=2, s_dim=62, s_dim_edge=8)
model.load_state_dict(torch.load(saved_model_path))
model.to(device)

######################################################################################################################################################################
#Evaluating model on test sets

test_loader = test315_28loader
test_acc, test_auprc, test_mcc = eval(model, test_loader, device)
print(f'Test AUROC: {test_acc:.3f},Test AUPRC: {test_auprc:.3f},, Test MCC: {test_mcc:.3f}')
