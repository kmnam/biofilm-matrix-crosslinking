"""
Utility functions for writing and analyzing LAMMPS data. 

Authors:
    Kee-Myoung Nam

Last updated:
    8/14/2024
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
def write_masses(masses, fmt=':.3f'):
    """
    Return a string containing the given masses to be outputted into a 
    LAMMPS data file.

    Parameters
    ----------
    masses : list of floats
        List of masses.
    fmt : str
        Format string for each floating-point value. 

    Returns
    -------
    Output string.
    """
    return '\n'.join([
        ('{} {' + fmt + '}').format(i + 1, mass) for i, mass in enumerate(masses)
    ])

#########################################################################
def write_lj_coefs(type_i, type_j, lj_coefs, fmt=':.6f'):
    """
    Return a one-line string containing the given Lennard-Jones coefficients
    to be outputted into a LAMMPS data file. 

    Parameters
    ----------
    type_i : int
        Index of first atom type. 
    type_j : int
        Index of second atom type. 
    lj_coefs : dict
        Dict containing the Lennard-Jones coefficients ('eps', 'sigma', 
        'cutoff') for a pair of atom types.
    fmt : str
        Format string for each floating-point value. 

    Returns
    -------
    Output string.
    """
    return ('{} {} {' + fmt + '} {' + fmt + '} {' + fmt + '}\n').format(
        type_i, type_j, lj_coefs['eps'], lj_coefs['sigma'], lj_coefs['cutoff']
    )

#########################################################################
def write_bond_coefs(bond_idx, bond_coefs, bond_type='harmonic', write_type=True,
                     fmt=':.6f'):
    """
    Return a string containing the given FENE bond energy coefficients to 
    be outputted into a LAMMPS data file. 

    Parameters
    ----------
    bond_idx : int
        Index of bond type. 
    bond_coefs : dict
        Dict containing the bond energy coefficients.
    bond_type : str
        String indicating the bond type; should be either 'fene' or
        'harmonic'.
    write_type : bool
        If True, write the bond type.
    fmt : str
        Format string for each floating-point value. 

    Returns
    -------
    Output string.
    """
    if bond_type == 'harmonic':
        return ('{} {}{' + fmt + '} {' + fmt + '}\n').format(
            bond_idx, bond_type + ' ' if write_type else '', bond_coefs['K'],
            bond_coefs['R0']
        )
    elif bond_type == 'fene':
        return ('{} {}{' + fmt + '} {' + fmt + '} {' + fmt + '} {' + fmt + '}\n').format(
            bond_idx, bond_type + ' ' if write_type else '', bond_coefs['K'],
            bond_coefs['R0'], bond_coefs['eps'], bond_coefs['sigma']
        )
    else:
        raise ValueError('Unrecognized bond type')

#########################################################################
def write_angle_coefs(angle_idx, angle_coefs, angle_type='cosine/delta',
                      write_type=True, fmt=':.6f'):
    """
    Return a string containing the given angle energy coefficients to 
    be outputted into a LAMMPS data file. 

    Parameters
    ----------
    angle_idx : int
        Index of angle type.
    angle_coefs : dict
        Dict containing the angle energy coefficients.
    angle_type : str
        String indicating the angle type; should be 'cosine' or 'cosine/delta'.
    write_type : bool
        If True, write the angle type.
    fmt : str
        Format string for each floating-point value. 

    Returns
    -------
    Output string.
    """
    if angle_type == 'cosine':
        return ('{} {}{' + fmt + '}\n').format(
            angle_idx, angle_type + ' ' if write_type else '', angle_coefs['K']
        )
    elif angle_type == 'cosine/delta':
        return ('{} {}{' + fmt + '} {' + fmt + '}\n').format(
            angle_idx, angle_type + ' ' if write_type else '', angle_coefs['K'],
            angle_coefs['theta0']
        )
    else:
        raise ValueError('Unrecognized angle type')

#########################################################################
def write_coords(polymers, crosslinkers, rng, polymer_type=1, crosslinker_type=2,
                 crosslinker_sticky_type=3, fmt=':.6f'):
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
        for j in range(len(polymer)):
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
def write_bonds(polymers, crosslinkers, polymer_bond_type=1, crosslinker_bond_type=2):
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
        Bond type within each polymer. 
    crosslinker_bond_type : int
        Bond type within each crosslinker.

    Returns
    -------
    Output string.  
    """
    outstr = ''
    bond_id = 1    # Number the bonds in the string as 1, 2, 3, ...
    
    # Write the polymer bonds first ... 
    first_atom_id = 1
    for i, polymer in enumerate(polymers):
        last_atom_id = first_atom_id + len(polymer) - 1
        atom_ids = list(range(first_atom_id, last_atom_id + 1))
        for j in range(len(polymer) - 1):
            # For each bond, write the line: 
            # [bond_id] [bond_type] [first_atom_id] [second_atom_id]
            outstr += '{} {} {} {}\n'.format(
                bond_id, polymer_bond_type, atom_ids[j], atom_ids[j+1] 
            )
            bond_id += 1
        first_atom_id = last_atom_id + 1

    # ... then write the crosslinker bonds 
    first_atom_id = sum(len(polymer) for polymer in polymers) + 1
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
def write_angles(polymers, angle_type=1):
    """
    Return a string containing the angles within the given set of polymers 
    to be outputted into a LAMMPS data file. 

    Parameters
    ----------
    polymers : list
        List of Polymer objects.
    angle_type : int
        Angle type within each polymer.

    Returns 
    -------
    Output string.
    """
    outstr = ''
    angle_id = 1    # Number the angles in the string as 1, 2, 3, ...
    
    # Write the polymer angles ... 
    first_atom_id = 1
    for i, polymer in enumerate(polymers):
        last_atom_id = first_atom_id + len(polymer) - 1
        atom_ids = list(range(first_atom_id, last_atom_id + 1))
        for j in range(1, len(polymer) - 1):
            # For each angle, write the line: 
            # [angle_id] [angle_type] [first_atom_id] [second_atom_id] [third_atom_id]
            outstr += '{} {} {} {} {}\n'.format(
                angle_id, angle_type, atom_ids[j-1], atom_ids[j], atom_ids[j+1]
            )
            angle_id += 1
        first_atom_id = last_atom_id + 1

    return outstr

#########################################################################
def write_init_config(polymers, crosslinkers, bond_length, crosslinker_style,
                      crosslinker_radius, monomer_mass, crosslinker_mass,
                      lj_coefs, bond_coefs, bond_types, angle_coefs, angle_types,
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
    bond_length : int
        Polymer bond lengths in the initial configuration. 
    crosslinker_style : str
        Crosslinker style, either 'atomic' or 'tetrahedral'.
    crosslinker_radius : float
        Crosslinker radius. 
    monomer_mass : float
        Monomer mass. 
    crosslinker_mass : float
        Crosslinker mass. 
    lj_coefs : list of dicts
        Lennard-Jones potential coefficients for each pair of atom types.
    bond_coefs : list of dicts
        Bond 
        
    TODO Fill this in
    """
    n_polymers = len(polymers)
    polymer_length = len(polymers[0])
    n_crosslinkers = len(crosslinkers)
    text = ''

    # If the crosslinkers are single atoms ... 
    if crosslinker_style == 'atomic':
        header = (
            'Test system of {} polymers of length {} and {} atomic '
            'crosslinkers, bond length = {}, crosslinker radius = {}\n\n'.format(
                n_polymers, polymer_length, n_crosslinkers, bond_length,
                crosslinker_radius
            )
        )
        text += '{} atoms\n{} bonds\n{} angles\n0 dihedrals\n0 impropers\n\n'.format(
            n_polymers * polymer_length + n_crosslinkers,
            n_polymers * (polymer_length - 1),
            n_polymers * (polymer_length - 2)
        )
        text += (
            '2 atom types\n2 bond types\n2 angle types\n0 dihedral types\n'
            '0 improper types\n\n'
        )
        text += write_box_dims(xmin, xmax, ymin, ymax, zmin, zmax) + '\n\n'
        text += 'Masses\n\n{}\n\n'.format(
            write_masses([monomer_mass, crosslinker_mass])
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
        text += '{} atoms\n{} bonds\n{} angles\n0 dihedrals\n0 impropers\n\n'.format(
            n_polymers * polymer_length + 5 * n_crosslinkers,
            n_polymers * (polymer_length - 1) + 4 * n_crosslinkers,
            n_polymers * (polymer_length - 2)
        )
        text += (
            '3 atom types\n3 bond types\n2 angle types\n0 dihedral types\n'
            '0 improper types\n\n'
        )
        text += write_box_dims(xmin, xmax, ymin, ymax, zmin, zmax) + '\n\n'
        text += 'Masses\n\n{}\n\n'.format(
            write_masses([monomer_mass, crosslinker_mass / 5, crosslinker_mass / 5])
        )
    
    # Write the Lennard-Jones coefficients
    #
    # A dict of coefficients should be present for each pair of atom types 
    text += 'PairIJ Coeffs\n\n'
    n_pairs = len(lj_coefs)
    n_atom_types = int((-1 + np.sqrt(1 + 8 * n_pairs)) / 2)
    k = 0
    for i in range(1, n_atom_types + 1):
        for j in range(i, n_atom_types + 1):
            text += write_lj_coefs(i, j, lj_coefs[k])
            k += 1
    text += '\n'

    # Write the bond potential coefficients
    #
    # A dict of coefficients should be present for every bond type
    text += 'Bond Coeffs\n\n'
    for i in range(len(bond_coefs)):
        text += write_bond_coefs(
            i + 1, bond_coefs[i], bond_types[i], write_type=True
        )
    text += '\n'

    # Write the angle potential coefficients 
    #
    # A dict of coefficients should be present for every angle type
    text += 'Angle Coeffs\n\n'
    for i in range(len(angle_coefs)):
        text += write_angle_coefs(
            i + 1, angle_coefs[i], angle_types[i], write_type=False
        )
    text += '\n'

    # Write atomic coordinates, bonds, and angles  
    text += 'Atoms\n\n{}\n'.format(write_coords(polymers, crosslinkers, rng))
    text += 'Bonds\n\n{}\n'.format(write_bonds(polymers, crosslinkers))
    text += 'Angles\n\n{}\n'.format(write_angles(polymers))

    with open(outfilename, 'w') as f:
        f.write(header + text)

