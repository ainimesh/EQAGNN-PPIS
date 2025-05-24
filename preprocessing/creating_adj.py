from Bio import SeqIO
from pdb_preprocess import create_nodes_dict_from_pdb, create_nodes_dict_from_pdb_updated, pdb_graph_edges, pdb_graph_partner_edges, amn_res_sequence
import os
import torch
import numpy as np
# Specify the path to your custom FASTA-like file
custom_fasta_file = " "


# Initialize lists to store data
pdb_ids = []
sequences = []
labels = []
input_files = []

# Read the custom FASTA-like file
with open(custom_fasta_file, "r") as file:
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

input_directory = " "

output_directory = " "
os.makedirs(output_directory, exist_ok=True)


def adjacency_matrix(protein_nodes_dict):

    '''
    it will take the input as 'protein_nodes_list = {0:np.array([1,2,3])......}
    output = n*n matrix as adjacency matrix
    '''
    # finding number of nodes from dictonary: dictonary keys are the nodes
    num_nodes = len(protein_nodes_dict)
    # initializing empty adjacency matrix
    adj_matrix = np.zeros((num_nodes,num_nodes), dtype = np.float32)
    # calculating pairwise distances of all the nodes i.e. residues in the protein dictonary
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            # calculating euclidian distance between node i and j
            distance = np.linalg.norm(protein_nodes_dict[i][0] - protein_nodes_dict[j][0])

            # set the adjacency values (making undirected graph)
            adj_matrix[i][j] = distance
            adj_matrix[j][i] = distance
    
    return adj_matrix



for i in range(len(input_files)):
    nodes_dict, mapped_nodes_dict = create_nodes_dict_from_pdb_updated(os.path.join(input_directory, input_files[i]), edge_type_C=True)
    print(f"File {input_files[i]} processed")

    # computing adjacency matrix
    adj_matrix = adjacency_matrix(mapped_nodes_dict)
    torch.save(adj_matrix, os.path.join(output_directory, f"{input_files[i].split('.')[0]}.adj.pt"))
