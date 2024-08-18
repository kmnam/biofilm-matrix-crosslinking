"""
Randomly generates initial configurations of polymers and crosslinkers.

Authors:
    Kee-Myoung Nam

Last updated:
    8/18/2024
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
    plot_crosslinkers,
    write_init_config
)

#########################################################################
def main(json_filename, init_filename, pdf_filename, rng_seed):
    # Parse input .json file  
    with open(json_filename) as f:
        params = json.load(f)
    n_polymers = params['n_polymers']
    polymer_length = params['polymer_length']
    bond_length = params['bond_length']
    monomer_mass = params['monomer_mass']
    crosslinker_style = params['crosslinker_style']
    n_crosslinkers = params['n_crosslinkers']
    crosslinker_mass = params['crosslinker_mass']
    if crosslinker_style == 'atomic':
        lj_coefs = [
            params['lj_coefs_11'], params['lj_coefs_12'], params['lj_coefs_22']
        ]
        bond_coefs = [
            params['fene_coefs'], 
            params['harmonic_crosslink_coefs']
        ]
        bond_styles = ['fene', 'harmonic']
        crosslinker_radius = 0.0
    else:    # crosslinker_style == 'tetrahedral'
        lj_coefs = [
            params['lj_coefs_11'], params['lj_coefs_12'], params['lj_coefs_13'],
            params['lj_coefs_22'], params['lj_coefs_23'], params['lj_coefs_33']
        ]
        bond_coefs = [
            params['fene_coefs'], 
            params['harmonic_within_coefs'],
            params['harmonic_crosslink_coefs']
        ]
        bond_styles = ['fene', 'harmonic', 'harmonic']
        crosslinker_radius = params['harmonic_within_coefs']['R0']
    angle_coefs = [
        params['angle_polymer_cosine_coefs'],
        params['angle_crosslinker_cosine_coefs'],
        params['angle_polymer_crosslinked_cosine_coefs']
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
    rng = np.random.default_rng(rng_seed)

    # Define von Mises distribution of polymer bond angles 
    angle_dist = lambda rng: rng.vonmises(
        params['angle_polymer_cosine_coefs']['theta0'],
        params['angle_polymer_cosine_coefs']['K']
    )

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
    plt.savefig(pdf_filename)

    # Write the generated initial configuration to file 
    write_init_config(
        polymers, crosslinkers, bond_length, crosslinker_style, crosslinker_radius,
        monomer_mass, crosslinker_mass, lj_coefs, bond_coefs, bond_styles,
        angle_coefs, xmin, xmax, ymin, ymax, zmin, zmax, init_filename
    )


#########################################################################
if __name__ == '__main__':
    json_filename = sys.argv[1]
    outprefix = sys.argv[2]
    rng_seed = int(sys.argv[3])
    main(json_filename, outprefix + '.data', outprefix + '.pdf', rng_seed)
