###########################################################################################
# Authors: Animesh
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import torch
from .radial import BesselBasis, PolynomialCutoff



class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(self, r_max: float, num_bessel: int, num_polynomial_cutoff: int):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self, edge_lengths: torch.Tensor
    ):
        bessel = self.bessel_fn(edge_lengths)  
        cutoff = self.cutoff_fn(edge_lengths) 
        return bessel * cutoff 
