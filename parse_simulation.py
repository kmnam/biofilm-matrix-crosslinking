"""
Authors:
    Kee-Myoung Nam

Last updated:
    4/25/2024
"""
import numpy as np
from lammps_utils import parse_simulation

if __name__ == '__main__':
    sim = parse_simulation(
        #'systems/test_react/test_init_0.data',
        #'systems/test_react/test_react_0.lammpstrj',
        #'systems/test_react/test_react_0.bond'
        'systems/tetrahedral_20p1000c/init.data',
        'systems/tetrahedral_20p1000c/tetrahedral_20p1000c.lammpstrj',
        'systems/tetrahedral_20p1000c/tetrahedral_20p1000c.bond'
    )
    polymer_bond_type = 1
    crosslink_type = 3
    for i in range(len(sim.times)):
        print(i)
        polymer_bonds = np.vstack((sim.bonds[i] == polymer_bond_type).nonzero()).T
        bond_lengths = np.array([
            np.linalg.norm(sim.coords[i][atom1, :] - sim.coords[i][atom2, :])
            for atom1, atom2 in polymer_bonds
        ])
        print(bond_lengths.min(), bond_lengths.max())
        print(polymer_bonds.shape[0])
        crosslinks = np.vstack((sim.bonds[i] == crosslink_type).nonzero()).T
        for j in range(crosslinks.shape[0]):
            atom1, atom2 = crosslinks[j, :]
            type1, type2 = sim.atom_types[atom1], sim.atom_types[atom2]
            bond_length = np.linalg.norm(
                sim.coords[i][atom1, :] - sim.coords[i][atom2, :]
            )
            print(atom1, atom2, type1, type2, sim.coords[i][atom1, :], sim.coords[i][atom2, :], bond_length)
        print(crosslinks.shape[0])
