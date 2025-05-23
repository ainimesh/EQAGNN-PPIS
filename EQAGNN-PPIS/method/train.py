
import sys
sys.path.append('/home/rishi/rishi_mtp/main_code')

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

os.chdir("/home/rishi/rishi_mtp/main_code")
print(os.listdir())
print(sys.path)


print("PyTorch version {}".format(torch.__version__))
print("PyG version {}".format(torch_geometric.__version__))
print("e3nn version {}".format(e3nn.__version__))

from .train_utils import run_experiment
# from models.egnn import EGNNModel
from model import GVPGNNModel
from model import GVP_AAMPModel
# from models.gvpgnn_aamp import GVPGNNModel as GVPGNNModel_AAMP


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f"Using device: {device}")


#Changing os directory
os.chdir("/home/rishi/rishi_mtp")





####################################################################  DATA PREPROCESSING  #############################################################################

#Hyperparams for data importing
DIST_NORM = 15

file_train = 'final_feats/Train335_psepos_CA.pkl'
file_test = 'final_feats/Test60_psepos_CA.pkl'

#load pickle file in a variable called psepos
import pickle

with open(file_train, 'rb') as f:
    psepos_train = pickle.load(f)

with open(file_test, 'rb') as f:
    psepos_test = pickle.load(f)


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


def load_data (pdb_list,label_list,what='train'):
  dataset = []
  for i,protein in tqdm(enumerate(pdb_list)):
    try:
      pssm = np.load('final_feats/pssm/'+protein +'.npy')
      hmm = np.load('final_feats/hmm/'+protein +'.npy')
      dssp = np.load('final_feats/dssp/'+protein +'.npy')
      resaf = np.load('final_feats/resAF/'+protein +'.npy')
      
      
      #prot_bert = np.load('final_feats/prot_bert/'+protein +'.npy')
      
      


      if what=='train':
        cords = psepos_train[protein]#np.load('/content/drive/MyDrive/rishi_mtp/final_feats/cords/'+protein +'_cords.npy')
      else:
        cords = psepos_test[protein]

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


train_obj = pd.read_pickle(r'Dataset/Train_335.pkl')
train_obj = pd.DataFrame(train_obj)

test_obj = pd.read_pickle(r'Dataset/Test_60.pkl')
test_obj = pd.DataFrame(test_obj)

train_dataset = load_data (list(train_obj.columns),list(train_obj.iloc[1]),'train')
test_dataset = load_data (list(test_obj.columns),list(test_obj.iloc[1]),'test')

dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

######################################################################################################################################################################

####################################################################  TRAINING  ######################################################################################

# # Set model
# model_name = "egnn"

# correlation = 4
# model = {
#     "egnn": EGNNModel,
#     "gvp": GVPGNNModel,
# }[model_name](num_layers=3, in_dim=62, out_dim=2, emb_dim=62, equivariant_pred= True)
# #(num_layers=2, in_dim=1085, out_dim=2, emb_dim=1085, equivariant_pred= True)

# best_val_acc, test_acc, train_time = run_experiment(
#     model,
#     dataloader,
#     val_loader,
#     test_loader,
#     n_epochs=1,
#     n_times=1,
#     device=device,
#     verbose=True
# )


# Set model
model_name = "gvp_att"
model_ = {
    "gvp": GVPGNNModel,
    "gvp_att": GVP_AAMPModel,
}[model_name](num_layers=8, in_dim=62, out_dim=2, s_dim=62, s_dim_edge=8)

best_val_acc, test_acc, train_time = run_experiment(
   
   
    model_,
    dataloader,
    val_loader,
    test_loader,
    n_epochs=40,
    n_times=1,
    device=device,
    verbose=True
)


######################################################################################################################################################################

