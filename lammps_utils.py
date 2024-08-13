"""
Utility functions for writing and analyzing LAMMPS data. 

Authors:
    Kee-Myoung Nam

Last updated:
    8/13/2024
"""
import numpy as np

#########################################################################
def dist_periodic(x, y, xmin, ymin, zmin, xmax, ymax, zmax):
    """
    Get the distance between x and y, assuming periodic boundary conditions.

    Parameters
    ----------
    x : numpy.ndarray
        First query point.
    y : numpy.ndarray
        Second query point.
    xmin, xmax : float, float
        Minimum and maximum x-coordinates.
    ymin, ymax : float, float
        Minimum and maximum y-coordinates.
    zmin, zmax : float, float
        Minimum and maximum z-coordinates.

    Returns
    -------
    Distance between x and y, assuming periodic boundary conditions.
    """
    delta = [xmax - xmin, ymax - ymin, zmax - zmin]
    total = 0.0
    for i in range(3):
        dist = np.abs(x[i] - y[i])
        if dist > delta[i] - dist:
            dist = delta[i] - dist
        total += dist ** 2
    return np.sqrt(total)

#########################################################################
def write_box_dims(xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Return a string containing the given box dimensions to be outputted
    into a LAMMPS data file.

    Parameters
    ----------
    xmin, xmax : float, float
        Minimum and maximum x-coordinates. 
    ymin, ymax : float, float
        Minimum and maximum y-coordinates.
    zmin, zmax : float, float
        Minimum and maximum z-coordinates.

    Returns
    -------
    Output string. 
    """
    return '{} {} xlo xhi\n{} {} ylo yhi\n{} {} zlo zhi'.format(
        xmin, xmax, ymin, ymax, zmin, zmax
    )

#########################################################################
def write_masses(masses):
    """
    Return a string containing the given masses to be outputted into a 
    LAMMPS data file.

    Parameters
    ----------
    masses : list of floats
        List of masses. 

    Returns
    -------
    Output string.
    """
    return '\n'.join(
        ['{} {:.3f}'.format(i + 1, mass) for i, mass in enumerate(masses)]
    )

#########################################################################
def write_lj_coefs(lj_coefs):
    """
    Return a string containing the given Lennard-Jones coefficients to 
    be outputted into a LAMMPS data file. 

    Parameters
    ----------
    lj_coefs : list of dicts
        List of dictionaries containing the Lennard-Jones coefficients
        ('eps', 'sigma', 'cutoff') for each pair of atom types.

    Returns
    -------
    Output string.
    """
    lines = []
    k = 0
    n_atom_types = (2 if len(lj_coefs) == 3 else 3)
    for i in range(1, n_atom_types + 1):
        for j in range(i, n_atom_types + 1):
            lines.append(
                '{} {} {:.10f} {:.10f} {:.10f}'.format(
                    i, j, lj_coefs[k]['eps'], lj_coefs[k]['sigma'], lj_coefs[k]['cutoff']
                )
            )
            k += 1

    return '\n'.join(lines)

#########################################################################
def write_bond_coefs(bond_coefs):
    """
    Return a string containing the given bond energy coefficients to 
    be outputted into a LAMMPS data file. 

    The first set of coefficients is assumed to specify the intra-polymer
    FENE bond energy parameters.

    Parameters
    ----------
    bond_coefs : list of dicts
        List of dictionaries containing the bond energy coefficients
        ('K', 'R0', 'eps', 'sigma' for FENE bonds; 'K', 'R0' for harmonic
        bonds) for each bond type. 

    Returns
    -------
    Output string.
    """
    lines = []
    n_bond_types = len(bond_coefs)
    for i in range(n_bond_types):
        if i == 0:
            line = '{} fene {:.10f} {:.10f} {:.10f} {:.10f}'.format(
                i + 1, bond_coefs[i]['K'], bond_coefs[i]['R0'],
                bond_coefs[i]['eps'], bond_coefs[i]['sigma']
            )
        else:
            line = '{} harmonic {:.10f} {:.10f}'.format(
                i + 1, bond_coefs[i]['K'], bond_coefs[i]['R0']
            )
        lines.append(line)

    return '\n'.join(lines)

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
        List of AtomicCrosslinker or TetrahedralCrosslinker objects.
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
    Output string.
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
    try:
        is_tetrahedral = (crosslinkers[0].coords.shape[0] == 5)
    except IndexError:   # If there are no crosslinkers, an exception is raised
        pass
    for i, crosslinker in enumerate(crosslinkers):
        molecule_id = len(polymers) + i + 1
        # If crosslinkers are tetrahedral, choose a peripheral atom to be sticky
        if is_tetrahedral:
            sticky_id = rng.integers(1, 5)
        else:
            sticky_id = None
        for j in range(crosslinker.coords.shape[0]):
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
        List of AtomicCrosslinker or TetrahedralCrosslinker objects.
    polymer_bond_type : int
        Type of bond within each polymer. 
    crosslinker_bond_type : int
        Type of bond within each crosslinker.

    Returns
    -------
    Output string.  
    """
    outstr = ''
    bond_id = 1    # Number the bonds in the string as 1, 2, 3, ...
    
    # Write the polymer bonds first ... 
    first_atom_id = 1
    for i, polymer in enumerate(polymers):
        last_atom_id = first_atom_id + polymer.length - 1
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
        for j in range(1, crosslinker.coords.shape[0]):
            # For each bond, write the line:
            # [bond_id] [bond_type] [first_atom_id] [second_atom_id]
            outstr += '{} {} {} {}\n'.format(
                bond_id, crosslinker_bond_type, first_atom_id, first_atom_id + j
            )
            bond_id += 1
        first_atom_id += crosslinker.coords.shape[0]

    return outstr

#########################################################################
def write_init_config(polymers, crosslinkers, bond_length, crosslinker_radius,
                      monomer_mass, crosslinker_mass, lj_coefs, bond_coefs,
                      rng, xmin, xmax, ymin, ymax, zmin, zmax, outfilename):
    """
    Write the given initial configuration of polymer and crosslinker
    coordinates to file.

    Parameters
    ----------
    polymers : list
        List of Polymer objects.
    crosslinkers : list
        List of AtomicCrosslinker or TetrahedralCrosslinker objects.
    TODO Fill this in
    """
    n_polymers = len(polymers)
    polymer_length = len(polymers[0])
    n_crosslinkers = len(crosslinkers)

    # If the crosslinkers are single atoms ... 
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
    # If the crosslinkers are tetrahedral ... 
    else:
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
    with open(outfilename, 'w') as f:
        f.write(header + para1 + para2 + para3 + para4 + para5 + para6 + para7 + para8)

