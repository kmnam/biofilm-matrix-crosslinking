"""
A simple script for running the main LAMMPS script (crosslink_tetrahedral.lammps)
according to preferences in the given input .json file.

Authors:
    Kee-Myoung Nam

Last updated:
    8/13/2024
"""

import sys
import json
import subprocess
import generate_molecules

# Parse the given .json file
json_filename = sys.argv[1]
with open(json_filename) as f:
    params = json.load(f)

# Generate initial configuration
outprefix = sys.argv[2]
init_filename = outprefix + '_init.data'
pdf_filename = outprefix + '_init.pdf'
rng_seed = int(sys.argv[3])
generate_molecules.main(json_filename, init_filename, pdf_filename, rng_seed)

# Write the LAMMPS input/output file paths to a file to be parsed by LAMMPS
input_filename = outprefix + '_paths.txt'
with open(input_filename, 'w') as f:
    f.write('{}\n'.format(init_filename))
    f.write('{}.lammpstrj\n'.format(outprefix))
    f.write('{}.bond\n'.format(outprefix))
    f.write('{}.angle\n'.format(outprefix))
    f.write('{}\n'.format(params['path_unbonded_template']))
    f.write('{}\n'.format(params['path_bonded_template']))
    f.write('{}\n'.format(params['path_forward_map']))

# Run LAMMPS 
subprocess.run([
    'lmp', '-i', 'crosslink_tetrahedral.lammps', '-v', 'VARS', input_filename
])


