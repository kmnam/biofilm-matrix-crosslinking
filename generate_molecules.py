"""
Randomly generates initial configurations of polymers and crosslinkers.

Authors:
    Kee-Myoung Nam

Last updated:
    8/13/2024
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from polymers import (
    generate_polymers,
    generate_atomic_crosslinkers,
    generate_tetrahedral_crosslinkers,
    plot_polymers,
    plot_crosslinkers
)
from lammps_utils import write_init_config

#########################################################################
if __name__ == '__main__':
    # Parse input arguments
    json_filename = sys.argv[1]
    outprefix = sys.argv[2]
    seed = int(sys.argv[3])

    # Parse input .json file  
    with open(json_filename) as f:
        params = json.load(f)
    n_polymers = params['n_polymers']
    polymer_length = params['polymer_length']
    bond_length = params['bond_length']
    theta_mean = params['init_angle_mean']
    theta_conc = params['init_angle_conc']
    monomer_mass = params['monomer_mass']
    crosslinker_style = params['crosslinker_style']
    n_crosslinkers = params['n_crosslinkers']
    crosslinker_radius = params['crosslinker_radius']
    crosslinker_mass = params['crosslinker_mass']
    if crosslinker_style == 'atomic':
        lj_coefs = [
            params['lj_coefs_11'], params['lj_coefs_12'], params['lj_coefs_22']
        ]
    else:    # crosslinker_style == 'tetrahedral'
        lj_coefs = [
            params['lj_coefs_11'], params['lj_coefs_12'], params['lj_coefs_13'],
            params['lj_coefs_22'], params['lj_coefs_23'], params['lj_coefs_33']
        ]
    if crosslinker_style == 'atomic':
        bond_coefs = [
            params['fene_coefs'], 
            params['harmonic_crosslink_coefs']
        ]
    else:    # crosslinker_style == 'tetrahedral'
        bond_coefs = [
            params['fene_coefs'], 
            params['harmonic_within_coefs'],
            params['harmonic_crosslink_coefs']
        ]
    angle_coefs = [
        params['angle_default_cosine_coefs'],
        params['angle_crosslinked_cosine_coefs']
    ]
    eps1 = params['inter_molecule_mindist']
    eps2 = params['intra_polymer_mindist']
    xmin = params['xmin']
    xmax = params['xmax']
    ymin = params['ymin']
    ymax = params['ymax']
    zmin = params['zmin']
    zmax = params['zmax']

    # Initialize random number generator 
    rng = np.random.default_rng(seed)

    # Define von Mises distribution of bond angles 
    angle_dist = lambda rng: rng.vonmises(theta_mean, theta_conc)

    # Generate box and distance thresholds, adding a little padding along
    # all faces of the box
    xmin_ = 0.95 * xmin
    xmax_ = 0.95 * xmax
    ymin_ = 0.95 * ymin
    ymax_ = 0.95 * ymax
    zmin_ = 0.95 * zmin
    zmax_ = 0.95 * zmax

    # Generate polymers and crosslinkers
    print('... generating polymers')
    polymers = generate_polymers(
        n_polymers, polymer_length, bond_length, angle_dist, rng, xmin, xmax,
        ymin, ymax, zmin, zmax, eps1, eps2
    )
    print('... generating crosslinkers')
    if crosslinker_style == 'atomic':
        crosslinkers = generate_atomic_crosslinkers(
            polymers, n_crosslinkers, rng, xmin, xmax, ymin, ymax, zmin, zmax,
            crosslinker_radius + eps1,     # Minimum separation from each monomer
            2 * crosslinker_radius + eps1  # Minimum separation from every other crosslinker
        )
    elif crosslinker_style == 'tetrahedral':
        crosslinkers = generate_tetrahedral_crosslinkers(
            polymers, n_crosslinkers, crosslinker_radius, rng, xmin, xmax,
            ymin, ymax, zmin, zmax, eps1, eps1
        )
    else:
        raise RuntimeError(
            'Invalid crosslinker style specified: {}'.format(crosslinker_style)
        )

    # Plot the polymers and crosslinkers projected onto the xy-, xz-, and
    # yz-planes
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    plot_polymers(polymers, axes[0], dims=(0, 1))
    plot_polymers(polymers, axes[1], dims=(0, 2))
    plot_polymers(polymers, axes[2], dims=(1, 2))
    plot_crosslinkers(crosslinkers, axes[0], dims=(0, 1))
    plot_crosslinkers(crosslinkers, axes[1], dims=(0, 2))
    plot_crosslinkers(crosslinkers, axes[2], dims=(1, 2))
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')
    axes[2].set_aspect('equal')
    plt.tight_layout()
    plt.savefig(outprefix + '.pdf')

    # Write the generated initial configuration to file 
    write_init_config(
        polymers, crosslinkers, bond_length, crosslinker_style, crosslinker_radius,
        monomer_mass, crosslinker_mass, lj_coefs, bond_coefs, angle_coefs, rng,
        xmin, xmax, ymin, ymax, zmin, zmax, outprefix + '.data'
    )

