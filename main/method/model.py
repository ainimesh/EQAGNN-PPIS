##################
# Author: Rishi and Animesh
##################

import torch
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool

from .layers import LayerNorm, GVP, GVPConvLayer, _merge, AAMP_Layer
from .utils import RadialEmbeddingBlock


# Our E(Q)AGNN Model for PPI binding site prediction

class EQAGNN_Model(torch.nn.Module):
    """
    GVP-GNN model from "Equivariant Graph Neural Networks for 3D Macromolecular Structure".
    """
    def __init__(
        self,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        num_layers: int = 5,
        in_dim=1,
        out_dim=1,
        s_dim: int = 128,
        v_dim: int = 16,
        s_dim_edge: int = 32,
        v_dim_edge: int = 1,
        pool: str = "sum",
        residual: bool = True,
        equivariant_pred: bool = False
    ):
        """
        Initializes an instance of the GVPGNNModel class with the provided parameters.

        Parameters:
        - r_max (float): Maximum distance for Bessel basis functions (default: 10.0)
        - num_bessel (int): Number of Bessel basis functions (default: 8)
        - num_polynomial_cutoff (int): Number of polynomial cutoff basis functions (default: 5)
        - num_layers (int): Number of layers in the model (default: 5)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - s_dim (int): Dimension of the node state embeddings (default: 128)
        - v_dim (int): Dimension of the node vector embeddings (default: 16)
        - s_dim_edge (int): Dimension of the edge state embeddings (default: 32)
        - v_dim_edge (int): Dimension of the edge vector embeddings (default: 1)
        - pool (str): Global pooling method to be used (default: "sum")
        - residual (bool): Whether to use residual connections (default: True)
        - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)
        """
        super().__init__()
        
        self.r_max = r_max
        self.num_layers = num_layers
        self.equivariant_pred = equivariant_pred
        self.s_dim = s_dim
        self.v_dim = v_dim
        
        activations = (F.relu, None)
        _DEFAULT_V_DIM = (s_dim, v_dim)
        _DEFAULT_E_DIM = (s_dim_edge, v_dim_edge) 

        # Node embedding
        self.emb_in = torch.nn.Embedding(in_dim, 1)
        self.W_v = torch.nn.Sequential(
            LayerNorm((s_dim, 0)),
            GVP((s_dim, 0), _DEFAULT_V_DIM,
                activations=(None, None), vector_gate=True)
        )

        # Edge embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        self.W_e = torch.nn.Sequential(
            LayerNorm((self.radial_embedding.out_dim, 1)),
            GVP((self.radial_embedding.out_dim, 1), _DEFAULT_E_DIM, 
                activations=(None, None), vector_gate=True)
        )
        
        # Stack of GNN layers
        self.layers = torch.nn.ModuleList(
            AAMP_Layer(
                _DEFAULT_V_DIM, (s_dim_edge+1, v_dim_edge),
                activations=activations, vector_gate=True,
                residual=residual
            ) 
            for _ in range(num_layers)
        )
        
        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        if self.equivariant_pred:
            # Linear predictor for equivariant tasks using geometric features
            self.pred = torch.nn.Linear(s_dim + v_dim * 3, out_dim)
        else:
            # MLP predictor for invariant tasks using only scalar features
            self.pred = torch.nn.Sequential(
                torch.nn.Dropout(0.1),
                torch.nn.Linear(s_dim, s_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(s_dim, out_dim),
                torch.nn.Softmax()
            )
    
    def forward(self, batch):

        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    
        h_V = torch.as_tensor(torch.Tensor(batch.atoms), dtype=torch.float32)
        cos_2 = torch.nn.CosineSimilarity(dim=1) 
        cosine = cos_2(batch.pos[batch.edge_index[0]], batch.pos[batch.edge_index[1]]).unsqueeze(1)
        #print(cosine.shape)
        h_E = (
            self.radial_embedding(lengths), 
            torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze_(-2)
        )
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        h_E = (torch.cat([h_E[0],cosine], dim=-1), h_E[1])

        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)
        out = _merge(*h_V)
        
        if not self.equivariant_pred:
            # Select only scalars for invariant prediction
            out = out[:,:self.s_dim]
        
        return self.pred(out)




## GVP-GNN Model For PPI bindiung site prediction task
class GVPGNNModel(torch.nn.Module):
    """
    GVP-GNN model from "Equivariant Graph Neural Networks for 3D Macromolecular Structure".
    """
    def __init__(
        self,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        num_layers: int = 5,
        in_dim=1,
        out_dim=1,
        s_dim: int = 128,
        v_dim: int = 16,
        s_dim_edge: int = 32,
        v_dim_edge: int = 1,
        pool: str = "sum",
        residual: bool = True,
        equivariant_pred: bool = False
    ):
        """
        Initializes an instance of the GVPGNNModel class with the provided parameters.

        Parameters:
        - r_max (float): Maximum distance for Bessel basis functions (default: 10.0)
        - num_bessel (int): Number of Bessel basis functions (default: 8)
        - num_polynomial_cutoff (int): Number of polynomial cutoff basis functions (default: 5)
        - num_layers (int): Number of layers in the model (default: 5)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - s_dim (int): Dimension of the node state embeddings (default: 128)
        - v_dim (int): Dimension of the node vector embeddings (default: 16)
        - s_dim_edge (int): Dimension of the edge state embeddings (default: 32)
        - v_dim_edge (int): Dimension of the edge vector embeddings (default: 1)
        - pool (str): Global pooling method to be used (default: "sum")
        - residual (bool): Whether to use residual connections (default: True)
        - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)
        """
        super().__init__()
        
        self.r_max = r_max
        self.num_layers = num_layers
        self.equivariant_pred = equivariant_pred
        self.s_dim = s_dim
        self.v_dim = v_dim
        
        activations = (F.relu, None)
        _DEFAULT_V_DIM = (s_dim, v_dim)
        _DEFAULT_E_DIM = (s_dim_edge, v_dim_edge) 

        # Node embedding
        self.emb_in = torch.nn.Embedding(in_dim, 1)
        self.W_v = torch.nn.Sequential(
            LayerNorm((s_dim, 0)),
            GVP((s_dim, 0), _DEFAULT_V_DIM,
                activations=(None, None), vector_gate=True)
        )

        # Edge embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        self.W_e = torch.nn.Sequential(
            LayerNorm((self.radial_embedding.out_dim, 1)),
            GVP((self.radial_embedding.out_dim, 1), _DEFAULT_E_DIM, 
                activations=(None, None), vector_gate=True)
        )
        
        # Stack of GNN layers
        self.layers = torch.nn.ModuleList(
            GVPConvLayer(
                _DEFAULT_V_DIM, (s_dim_edge+1, v_dim_edge),
                activations=activations, vector_gate=True,
                residual=residual,
            )
            for _ in range(num_layers)
        )
        
        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        if self.equivariant_pred:
            # Linear predictor for equivariant tasks using geometric features
            self.pred = torch.nn.Linear(s_dim + v_dim * 3, out_dim)
        else:
            # MLP predictor for invariant tasks using only scalar features
            self.pred = torch.nn.Sequential(
                torch.nn.Dropout(0.1),
                torch.nn.Linear(s_dim, s_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(s_dim, out_dim),
                torch.nn.Softmax()
            )
    
    def forward(self, batch):

        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        
        h_V = torch.as_tensor(torch.Tensor(batch.atoms), dtype=torch.float32)
        cos_2 = torch.nn.CosineSimilarity(dim=1) 
        cosine = cos_2(batch.pos[batch.edge_index[0]], batch.pos[batch.edge_index[1]]).unsqueeze(1)

        h_E = (
            self.radial_embedding(lengths), 
            torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze_(-2)
        )
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        h_E = (torch.cat([h_E[0],cosine], dim=-1), h_E[1])
        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)
        
        out = _merge(*h_V)
        
        if not self.equivariant_pred:
            # Select only scalars for invariant prediction
            out = out[:,:self.s_dim]
        
        return self.pred(out)