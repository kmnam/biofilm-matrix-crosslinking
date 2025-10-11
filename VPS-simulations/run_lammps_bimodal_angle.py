"""
A simple script for running the main LAMMPS script (crosslink_tetrahedral.lammps)
according to preferences in the given input .json file.

Authors:
    Kee-Myoung Nam

Last updated:
    9/16/2025
"""

import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from generate import generate_polymers
from utils import plot_polymers, write_init_config

#########################################################################
def generate_init_config_polymers(json_filename, init_filename, pdf_filename,
                                  rng_seed):
    """
    Generate an initial configuration of polymers, according to the settings
    specified in the given .json file. 

    Parameters
    ----------
    json_filename : str
        Path to input .json file. 
    init_filename : str
        Path to output file, which specifies the polymer configuration as 
        a LAMMPS input file. 
    pdf_filename : str
        Path to output PDF file, showing the polymer configuration as 
        xy-, xz-, and yz-projections. 
    rng_seed : int
        Seed for random number generation. 
    """
    # Parse input .json file  
    with open(json_filename) as f:
        params = json.load(f)

    # Parse input parameters  
    n_polymers = params['n_polymers']
    polymer_length = params['polymer_length']
    bond_length = params['bond_length']
    monomer_mass = params['monomer_mass']
    lj_coefs = [params['lj_coefs']]
    bond_coefs = [params['fene_coefs']]
    angle_coefs = params['angle_coefs']
    try:
        dihedral_coefs = params['dihedral_coefs']
        for key in dihedral_coefs:
            if 'n' not in dihedral_coefs[key]:
                dihedral_coefs[key]['n'] = 1
            dihedral_coefs[key]['d'] = 1
    except KeyError:
        dihedral_coefs = None
    eps1 = params['inter_molecule_mindist']
    eps2 = params['intra_polymer_mindist']
    xmin = params['xmin']
    xmax = params['xmax']
    ymin = params['ymin']
    ymax = params['ymax']
    zmin = params['zmin']
    zmax = params['zmax']
    try:
        max_backtracks_per_polymer = params['max_backtracks_per_polymer']
    except KeyError:
        max_backtracks_per_polymer = 1000
    try:
        max_tries_per_bond = params['max_tries_per_bond']
    except KeyError:
        max_tries_per_bond = 1000

    # Initialize random number generator 
    rng = np.random.default_rng(rng_seed)

    # If there are multiple angle types, randomly assign angle types along
    # each polymer 
    if len(angle_coefs) > 1:
        angle_probs = [
            angle_coefs['type{}'.format(i + 1)]['prob']
            for i in range(len(angle_coefs))
        ]
        angle_types = np.zeros((n_polymers, polymer_length - 2), dtype=np.int64)
        for i in range(n_polymers):
            for j in range(polymer_length - 2):
                angle_types[i, j] = rng.choice(len(angle_coefs), p=angle_probs) + 1
    else:
        angle_types = np.ones((n_polymers, polymer_length - 2), dtype=np.int64)

    # Define von Mises distributions of bond angles and dihedral angles
    angle_dists = [
        lambda rng: rng.vonmises(
            angle_coefs['type{}'.format(i + 1)]['theta0'],
            angle_coefs['type{}'.format(i + 1)]['K']
        )
        for i in range(len(angle_coefs))
    ]
    if dihedral_coefs is None:
        dihedral_dists = None
    else:
        dihedral_dists = []
        for i in range(len(dihedral_coefs)):
            key = 'type{}'.format(i + 1)
            K = dihedral_coefs[key]['K']
            n = dihedral_coefs[key]['n']
            if n == 1:
                dihedral_dists.append(lambda rng: rng.vonmises(np.pi, K))
            elif n == 2:
                dihedral_dists.append(lambda rng: rng.vonmises(np.pi / 2, K))

    # Generate box and distance thresholds, adding a little padding along
    # all faces of the box
    xmin_ = 0.95 * xmin
    xmax_ = 0.95 * xmax
    ymin_ = 0.95 * ymin
    ymax_ = 0.95 * ymax
    zmin_ = 0.95 * zmin
    zmax_ = 0.95 * zmax

    # Generate polymers and crosslinkers
    polymers = generate_polymers(
        n_polymers, polymer_length, bond_length, angle_types, angle_coefs, rng,
        xmin, xmax, ymin, ymax, zmin, zmax, eps1, eps2, dihedral_dists=dihedral_dists,
        max_backtracks_per_polymer=max_backtracks_per_polymer, 
        max_tries_per_bond=max_tries_per_bond
    )

    # Plot the polymers and crosslinkers projected onto the xy-, xz-, and
    # yz-planes
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    plot_polymers(polymers, axes[0], dims=(0, 1))
    plot_polymers(polymers, axes[1], dims=(0, 2))
    plot_polymers(polymers, axes[2], dims=(1, 2))
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')
    axes[2].set_aspect('equal')
    plt.tight_layout()
    plt.savefig(pdf_filename)

    # Write the generated initial configuration to file
    dihedral_types = np.zeros((n_polymers, polymer_length - 3), dtype=np.int64)
    for i in range(n_polymers):
        for j in range(polymer_length - 3):
            # The (i, j)-th entry here is the dihedral angle along the segment
            # j-(j+1)-(j+2)-(j+3), which is determined by the angle types at 
            # j + 1 and j + 2
            #
            # Note that these entries correspond to angle_types[i, j] and 
            # angle_types[i, j + 1], respectively
            if angle_types[i, j] == 2 or angle_types[i, j + 1] == 2:
                dihedral_types[i, j] = 2
            else:
                dihedral_types[i, j] = 1
    write_init_config(
        polymers, [], bond_length, 'none', 0, monomer_mass, 0, lj_coefs,
        bond_coefs, ['fene'],
        [angle_coefs['type{}'.format(i + 1)] for i in range(len(angle_coefs))],
        angle_types,
        [dihedral_coefs['type{}'.format(i + 1)] for i in range(len(dihedral_coefs))],
        dihedral_types,
        xmin, xmax, ymin, ymax, zmin, zmax, init_filename
    )

#########################################################################
if __name__ == '__main__':
    # Parse the given .json file
    json_filename = sys.argv[1]
    with open(json_filename) as f:
        params = json.load(f)

    # Generate initial configuration
    outprefix = sys.argv[2]
    init_filename = outprefix + '_init.data'
    pdf_filename = outprefix + '_init.pdf'
    rng_seed = int(sys.argv[3])
    generate_init_config_polymers(
        json_filename, init_filename, pdf_filename, rng_seed
    )

    # Write the LAMMPS input/output file paths to a file to be parsed by LAMMPS
    rng = np.random.default_rng(rng_seed)
    input_filename = outprefix + '_paths.txt'
    with open(input_filename, 'w') as f:
        # Input file containing initial configurations 
        f.write('{}\n'.format(init_filename))
        # Output files 
        f.write('{}.lammpstrj\n'.format(outprefix))
        f.write('{}.bond\n'.format(outprefix))
        f.write('{}.angle\n'.format(outprefix))
        # Timestep and other simulation parameters 
        f.write('{:.10e}\n'.format(params['dt']))
        f.write('{:.10e}\n'.format(params['damp']))
        nsteps = int(params['t_final'] / params['dt'])
        nsteps_per_dump = int(params['dt_per_dump'] / params['dt'])
        f.write('{}\n'.format(nsteps))
        f.write('{}\n'.format(nsteps_per_dump))
        # Random seeds 
        f.write('{}\n'.format(rng.integers(0, 1000)))
        f.write('{}\n'.format(rng.integers(0, 1000)))
        f.write('{}\n'.format(rng.integers(0, 1000)))

    # Run LAMMPS
    subprocess.run([
        'lmp', '-i', 'bimodal_angles.lammps', '-v', 'VARS', input_filename
    ])

