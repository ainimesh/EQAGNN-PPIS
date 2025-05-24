
# -*- coding: utf-8 -*-
"""
@author: Animesh
"""

import requests
import time
import os

# Specify the path to your custom FASTA-like file
train_fasta_file = " "

# Output directory where the chain-specific PDB files will be saved
output_directory = " "
os.makedirs(output_directory, exist_ok=True)


def fetch_pdb_file(pdb_id):
    the_url = "https://files.rcsb.org/download/" + pdb_id
    page = requests.get(the_url)
    pdb_file = str(page.content)
    pdb_file = pdb_file.replace('\\n', '\n')
    return(pdb_file)
    
# Initialize lists to store data
# Read the custom FASTA-like file
with open(train_fasta_file, "r") as file:
    lines = file.readlines()
    # Process the lines in groups of three
    for i in range(0, len(lines), 3):
        # Extract PDB ID with chain from the first line
        header_line = lines[i].strip()
        pdb_id_with_chain = header_line[1:]  # Remove the '>' character
        pdb_id = pdb_id_with_chain[:4]
        chain = pdb_id_with_chain[-1]

        # fetch pdb file 
        pdbfile = fetch_pdb_file(pdb_id + ".pdb")
        
        # Save the entire PDB file
        pdb_file_name = os.path.join(output_directory, f"{pdb_id}.pdb")
        print("writing" + pdb_file_name)
        with open(pdb_file_name, "w") as fd:
            fd.write(pdbfile)

print("All Done!")