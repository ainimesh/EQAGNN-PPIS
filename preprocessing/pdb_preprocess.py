"""

Author: Animesh
Github: @ainimesh


This file is to Perform following tasks:

Input: Takes 1 pdb file as input at the time 

Tasks: 
        
        
1. Returns nodes dict which has nodes as keys and co-ordinates of that node as values.
    'create_nodes_dict_from_pdb(pdb_file_path, edge_type_CA = True, edge_type_avg = False, include_hetatoms = False)'
    Input: pdb_path
    returns: nodes_dict

2. Returns Edge list.
    pdb_graph_edges(nodes_dict, threshold)
    Input: Nodes_dictonaries and threshold to connect the edges
    Returns: edge_list

3. Pdb edge list if we take two graphs
    pdb_graph_edges_from_two_graphs(nodes1, nodes2, threshold):
    Input: nodes1_dict and nodes2_dict for 2 pdbs and threshold
    returns: Edge_list 

4. Function to return edge list required by torch_geometric
    pdb_torch_edge_list(edges)
    input: edge_list
    output: list of edges torch.tensor([[1,1..],[2,3,..]])


5. function to return Amino acid sequence and residue sequence.
    amn_res_sequence(input_pdb)
    Input: input_pdb_file
    Returns: two things (sequence as string and residue_id as list)

        


"""
# ------------------------------------------------------------------------------------------------------------------------
# Importing Required Libraries

import numpy as np
import os
from tqdm import tqdm
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import networkx as nx
from Bio.PDB.PDBIO import PDBIO
import sys
import torch
from torch_geometric.data import Data

# ----------------------------------------------------Defining Impo. things for Protein----------------------------------
# list of 20 proteins
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Dictionary for getting Residue symbols
ressymbl = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN':'Q', 'ARG':'R', 'SER': 'S','THR': 'T', 'VAL': 'V', 'TRP':'W', 'TYR': 'Y'}


# -----------------------------------------------------------------------------------------------------------------------


## ------------------------ Defining functions for performing various Preprocessing tasks -------------------

# writing a function to create graph 
"""
This function will create graph for given protein PDb file.

Input: PDB_path, edge_type, threshold, hetatoms
Output: edges, nodes

"""


def create_nodes_dict_from_pdb(pdb_file_path, edge_type_CA = True, edge_type_avg = False, include_hetatoms = False):
    parser = PDBParser()
    structure = parser.get_structure('pdb',pdb_file_path)
    model = structure[0]

    # Create node names and store them in a dictonary
    nodes_dict = {}

    # for every chain
    for chain in model:
        
        # IF WE WANT TO TAKE THE ONLY 'CA' ATOM OF THE RESIDUE THEN
        if edge_type_CA:
            for residue in chain:
                if include_hetatoms or residue.get_id()[0] == ' ':
                    residue_id = residue.get_id()[1]
                    amino_acid_name = residue.get_resname()
                    chain_id = chain.get_id()
                    node_name = f"{chain_id}:{residue_id}:{amino_acid_name}"
                    if node_name not in nodes_dict:
                        nodes_dict[node_name] = []

                    if 'CA' in residue:
                        ca_coord = residue['CA'].get_coord()
                        nodes_dict[node_name].append(ca_coord)
                    else:
                        avg_coords = np.zeros(3)
                        atom_count = 0
                        for atom in residue:
                            avg_coords += atom.get_coord()
                            atom_count += 1

                        avg_coords /= atom_count
                        nodes_dict[node_name].append(avg_coords)
                 
        
        # IF WE WANT TO TAKE THE ONLY 'AVERAGE OF ATOMS' OF THE RESIDUES THEN
        if edge_type_avg:
            for residue in chain:
                if include_hetatoms or residue.get_id()[0] == ' ':
                    residue_id = residue.get_id()[1]
                    amino_acid_name = residue.get_resname()
                    chain_id = chain.get_id()
                    node_name = f"{chain_id}:{residue_id}:{amino_acid_name}"

                    if node_name not in nodes_dict:
                        nodes_dict[node_name] = []

                    # calculate average co-ords
                    avg_coords = np.zeros(3)
                    atom_count = 0
                    for atom in residue:
                        avg_coords += atom.get_coord()
                        atom_count += 1

                    avg_coords /= atom_count
                    nodes_dict[node_name].append(avg_coords)



    # Create a mapping from keys to sequential values
    key_to_index = {key: index for index, key in enumerate(nodes_dict.keys())}


    # Resulting dictionary with keys mapped to sequential values
    mapped_nodes_dict = {key_to_index[key]: value for key, value in nodes_dict.items()}

      
    return nodes_dict, mapped_nodes_dict

#! ---------------------------------------Create nodes dict from pdb for "CA", "C" and side chain centroid ("SC") -----------

def create_nodes_dict_from_pdb_updated(pdb_file_path, edge_type_CA=False, edge_type_C=False, edge_type_SC=False, include_hetatoms=False):
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb_file_path)
    model = structure[0]

    # Create node names and store them in a dictionary
    nodes_dict = {}

    for chain in model:
        for residue in chain:
            if include_hetatoms or residue.get_id()[0] == ' ':
                residue_id = residue.get_id()[1]
                amino_acid_name = residue.get_resname()
                chain_id = chain.get_id()
                node_name = f"{chain_id}:{residue_id}:{amino_acid_name}"
                if node_name not in nodes_dict:
                    nodes_dict[node_name] = []

                if edge_type_CA and 'CA' in residue:
                    nodes_dict[node_name].append(residue['CA'].get_coord())
                elif edge_type_C and 'C' in residue:
                    nodes_dict[node_name].append(residue['C'].get_coord())
                elif edge_type_SC:
                    side_chain_atoms = [atom for atom in residue if atom.get_name() not in ["N", "CA", "C", "O"]]
                    if len(side_chain_atoms) > 0:
                        sc_coords = np.mean([atom.get_coord() for atom in side_chain_atoms], axis=0)
                        nodes_dict[node_name].append(sc_coords)
                    else:
                        # For Glycine or if no side chain atoms
                        avg_coords = np.mean([atom.get_coord() for atom in residue], axis=0)
                        nodes_dict[node_name].append(avg_coords)
                else:
                    # Default: Average of all atoms
                    avg_coords = np.mean([atom.get_coord() for atom in residue], axis=0)
                    nodes_dict[node_name].append(avg_coords)

    
     # Create a mapping from keys to sequential values
    key_to_index = {key: index for index, key in enumerate(nodes_dict.keys())}


    # Resulting dictionary with keys mapped to sequential values
    mapped_nodes_dict = {key_to_index[key]: value for key, value in nodes_dict.items()}
    
    return nodes_dict, mapped_nodes_dict

#! --------------------------------------------end---------------------------------------------------------------------------

## ----------------------------------function to get edge list of a graph ---------------------

'''
This function will take:

Input: Nodes_dict (or) Mapped_nodes_dict, and threshold to join an edge

Output: edges_list, distance, pytorch compatible edge_list

'''
# getting edges from node(key:value) dictonary of a graph
def pdb_graph_edges(nodes, threshold=10.0):

    # defining an empty edge list
    edges_list = []
    distance = []
    
    # Calculate the edges_list based on the distance between the selected atom coordinates
    for node1_name, node1_coords in nodes.items():
        # print('coords',node1_coords)
        # print('mean', np.mean(node1_coords, axis=0))
        for node2_name, node2_coords in nodes.items():
            # print(node2_coords)
            # print(np.mean(node2_coords, axis=0))
            if node1_name != node2_name:

                avg_dist = np.linalg.norm(np.mean(node1_coords, axis=0) - np.mean(node2_coords, axis=0))
                # avg_dist = np.linalg.norm(node1_coords - node2_coords)
                
                if avg_dist <= threshold:
                    edges_list.append((node1_name, node2_name))
                    distance.append(avg_dist)

    edge_index = []

    # Create lists to store source nodes and target nodes for each edge
    src_nodes = []
    tgt_nodes = []

    for edge in edges_list:
        src_node, tgt_node = edge
        src_nodes.append(src_node)
        tgt_nodes.append(tgt_node)

        
    edge_index.append(src_nodes)
    edge_index.append(tgt_nodes)

    

    return edges_list, torch.tensor(distance), torch.tensor(edge_index, dtype=torch.long)


## ! ------------------------------------- FOR PARTNER PROTEINS: function to get partner edges and labels -------

'''
This function will take 

Input: mapped nodes dictonary of graph 1 and graph2, threshold to define binding residues

Output: pos_edges_list, all_partner_edges, distance, labels tensor

'''
# getting edges from node(key:value) dictonary 

def pdb_graph_partner_edges(node_dict1, node_dict2, threshold=14.0):

    # defining an empty edge list

    all_partner_edges = []
    pos_edges_list = []
    distance = []
    
    # Calculate the edges_list based on the distance between the selected atom coordinates
    for node1_name, node1_coords in node_dict1.items():
        # print(node1_coords)
        # print(np.mean(node1_coords, axis=0))
        for node2_name, node2_coords in node_dict2.items():
            # print(node2_coords)
            # print(np.mean(node2_coords, axis=0))
            # if node1_name != node2_name:

            avg_dist = np.linalg.norm(np.mean(node1_coords, axis=0) - np.mean(node2_coords, axis=0))
            

            all_partner_edges.append((node1_name, node2_name))
            if avg_dist <= threshold:
                pos_edges_list.append((node1_name,node2_name))
                distance.append(avg_dist)
    labels = []
    for a in all_partner_edges:
        if a in pos_edges_list:
            labels.append(1)
        else:
            labels.append(0)



    return pos_edges_list, all_partner_edges, torch.tensor(distance), torch.tensor(labels)



## ------------------------------------- Function to get amino sequence and residue sequence from a PDB file--------

'''
This function will take 

Input: PDB file

Output: Amino acid sequence and residue_ids sequence

'''

def amn_res_sequence(input_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", input_pdb)
    amino_seq = ''
    residue_ids = []
    for model in structure:
          for chain in model:
            for residue in chain:
              residue_id = residue.get_id()[1]
              residue_ids.append(residue_id)
              if residue.get_resname() in ressymbl.keys():
                  amino_seq = amino_seq+ ressymbl[residue.get_resname()]
    return amino_seq, residue_ids



