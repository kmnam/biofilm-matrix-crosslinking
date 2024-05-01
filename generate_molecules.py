"""
Authors:
    Kee-Myoung Nam

Last updated:
    4/13/2024
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
from lammps_utils import (
    write_box_dims,
    write_masses,
    write_lj_coefs,
    write_bond_coefs,
    write_molecule_coords,
    write_molecule_bonds
)

#########################################################################
if __name__ == '__main__':
    # Parse input arguments and JSON file
    infilename = sys.argv[1]
    outprefix = sys.argv[2]
    seed = int(sys.argv[3])
    data = ''
    with open(infilename) as f:
        data = f.read()
    params = json.loads(data)
    n_polymers = params['n_polymers']
    polymer_length = params['polymer_length']
    bond_length = params['bond_length']
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
            params['harmonic_crosslink_coefs'],
            params['harmonic_within_coefs']
        ]
    eps1 = params['inter_molecule_mindist']
    eps2 = params['intra_polymer_mindist']
    xmin = params['xmin']
    xmax = params['xmax']
    ymin = params['ymin']
    ymax = params['ymax']
    zmin = params['zmin']
    zmax = params['zmax']
    rng = np.random.default_rng(seed)

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
        n_polymers, polymer_length, bond_length, rng, xmin, xmax, ymin, ymax,
        zmin, zmax, eps1, eps2
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

    # Write output file
    if crosslinker_style == 'atomic':
        header = (
            'Test system of {} polymers of length {} and {} atomic '
            'crosslinkers, bond length = {}, crosslinker radius = {}\n\n'.format(
                n_polymers, polymer_length, n_crosslinkers, bond_length,
                crosslinker_radius
            )
        )
        para1 = '{} atoms\n{} bonds\n0 angles\n0 dihedrals\n0 impropers\n\n'.format(
            n_polymers * polymer_length + n_crosslinkers,
            n_polymers * (polymer_length - 1)
        )
        para2 = (
            '2 atom types\n2 bond types\n0 angle types\n0 dihedral types\n'
            '0 improper types\n\n'
        )
        para3 = write_box_dims(xmin, xmax, ymin, ymax, zmin, zmax) + '\n\n'
        para4 = 'Masses\n\n{}\n\n'.format(
            write_masses([monomer_mass, crosslinker_mass])
        )
        para5 = 'PairIJ Coeffs\n\n{}\n\n'.format(write_lj_coefs(lj_coefs))
        para6 = 'Bond Coeffs\n\n{}\n\n'.format(write_bond_coefs(bond_coefs))
        para7 = 'Atoms\n\n{}\n'.format(
            write_molecule_coords(polymers, crosslinkers, rng)
        )
        para8 = 'Bonds\n\n{}\n'.format(
            write_molecule_bonds(polymers, crosslinkers)
        )
    else:    # crosslinker_style == 'tetrahedral'
        header = (
            'Test system of {} polymers of length {} and {} tetrahedral '
            'crosslinkers, bond length = {}, crosslinker radius = {}\n\n'.format(
                n_polymers, polymer_length, n_crosslinkers, bond_length,
                crosslinker_radius
            )
        )
        para1 = '{} atoms\n{} bonds\n0 angles\n0 dihedrals\n0 impropers\n\n'.format(
            n_polymers * polymer_length + 5 * n_crosslinkers,
            n_polymers * (polymer_length - 1) + 4 * n_crosslinkers
        )
        para2 = (
            '3 atom types\n3 bond types\n0 angle types\n0 dihedral types\n'
            '0 improper types\n\n'
        )
        para3 = write_box_dims(xmin, xmax, ymin, ymax, zmin, zmax) + '\n\n'
        para4 = 'Masses\n\n{}\n\n'.format(
            write_masses([monomer_mass, crosslinker_mass / 5, crosslinker_mass / 5])
        )
        para5 = 'PairIJ Coeffs\n\n{}\n\n'.format(write_lj_coefs(lj_coefs))
        para6 = 'Bond Coeffs\n\n{}\n\n'.format(write_bond_coefs(bond_coefs))
        para7 = 'Atoms\n\n{}\n'.format(
            write_molecule_coords(polymers, crosslinkers, rng)
        )
        para8 = 'Bonds\n\n{}\n'.format(
            write_molecule_bonds(polymers, crosslinkers)
        )
    with open(outprefix + '.data', 'w') as f:
        f.write(header + para1 + para2 + para3 + para4 + para5 + para6 + para7 + para8)

