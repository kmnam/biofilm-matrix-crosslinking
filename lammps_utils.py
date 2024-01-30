"""
Authors:
    Kee-Myoung Nam

Last updated:
    1/29/2024
"""
import numpy as np

#########################################################################
def write_molecule_coords(polymers, crosslinkers, rng, polymer_type=1,
                          crosslinker_type=2, crosslinker_sticky_type=3,
                          fmt=':.6f'):
    """
    Return a string containing the given polymer and crosslinker coordinates
    to be outputted into a LAMMPS data file.

    Parameters
    ----------
    polymers : list
        List of Polymer objects.
    crosslinkers : list
        List of TetrahedralCrosslinker objects.
    rng : numpy.random.Generator
        Random number generator (for choosing sticky atoms).
    polymer_type : int
        Atom type for each monomer. 
    crosslinker_type : int
        Atom type for each non-sticky crosslinker atom.
    crosslinker_sticky_type : int
        Atom type for each sticky crosslinker atom.
    fmt : str
        Format string for each coordinate.

    Returns
    -------
    Output string to be included in a LAMMPS data file.
    """
    outstr = ''
    atom_id = 1    # Number the atoms in the string as 1, 2, 3, ...

    # Write the polymer coordinates first ...
    for i, polymer in enumerate(polymers):
        molecule_id = i + 1
        for j in range(polymer.length):
            # For each atom, write the line:
            # [atom_id] [molecule_id] [atom_type] [xcoord] [ycoord] [zcoord]
            outstr += ('{} {} {} {' + fmt + '} {' + fmt + '} {' + fmt + '}\n').format(
                atom_id, molecule_id, polymer_type,
                polymer.coords[j, 0],
                polymer.coords[j, 1],
                polymer.coords[j, 2]
            )
            atom_id += 1

    # ... then write the crosslinker coordinates
    for i, crosslinker in enumerate(crosslinkers):
        molecule_id = len(polymers) + i + 1
        sticky_id = rng.integers(1, 5)   # Choose a peripheral atom to be sticky
        for j in range(5):
            # For each atom, write the line:
            # [atom_id] [molecule_id] [atom_type] [xcoord] [ycoord] [zcoord]
            atom_type = crosslinker_type if j != sticky_id else crosslinker_sticky_type
            outstr += ('{} {} {} {' + fmt + '} {' + fmt + '} {' + fmt + '}\n').format(
                atom_id, molecule_id, atom_type,
                crosslinker.coords[j, 0],
                crosslinker.coords[j, 1],
                crosslinker.coords[j, 2]
            )
            atom_id += 1

    return outstr

#########################################################################
def write_molecule_bonds(polymers, crosslinkers, polymer_bond_type=1,
                         crosslinker_bond_type=2):
    """
    Return a string containing the bonds within the given set of polymers 
    and crosslinkers to be outputted into a LAMMPS data file.

    Parameters
    ----------
    polymers : list
        List of Polymer objects.
    crosslinkers : list
        List of TetrahedralCrosslinker objects.
    polymer_bond_type : int
        Type of bond within each polymer. 
    crosslinker_bond_type : int
        Type of bond within each crosslinker.

    Returns
    -------
    Output string to be included in a LAMMPS data file.  
    """
    outstr = ''
    bond_id = 1    # Number the bonds in the string as 1, 2, 3, ...
    
    # Write the polymer bonds first ... 
    first_atom_id = 1
    for i, polymer in enumerate(polymers):
        last_atom_id = first_atom_id + polymer.length
        atom_ids = list(range(first_atom_id, last_atom_id + 1))
        for j in range(polymer.length - 1):
            # For each bond, write the line: 
            # [bond_id] [bond_type] [first_atom_id] [second_atom_id]
            outstr += '{} {} {} {}\n'.format(
                bond_id, polymer_bond_type, atom_ids[j], atom_ids[j+1] 
            )
            bond_id += 1
        first_atom_id = last_atom_id + 1

    # ... then write the crosslinker bonds 
    first_atom_id = sum(polymer.length for polymer in polymers) + 1
    for i, crosslinker in enumerate(crosslinkers):
        for j in range(1, 5):
            # For each bond, write the line:
            # [bond_id] [bond_type] [first_atom_id] [second_atom_id]
            outstr += '{} {} {} {}\n'.format(
                bond_id, crosslinker_bond_type, first_atom_id, first_atom_id + j
            )
            bond_id += 1
        first_atom_id += 5

    return outstr

