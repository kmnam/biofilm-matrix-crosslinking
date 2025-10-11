"""
Utility functions for plotting polymer configurations and writing LAMMPS 
input files. 

Authors:
    Kee-Myoung Nam

Last updated:
    10/10/2025
"""
import numpy as np
from polymers import Polymer, AtomicCrosslinker, TetrahedralCrosslinker

#########################################################################
#                         POTENTIAL FUNCTIONS                           #
#########################################################################
def lj(r, eps, sigma, cutoff):
    """
    Compute the Lennard-Jones potential for the given distance with the 
    given parameters.
    """
    if r > cutoff:
        return 0.0
    else:
        sr1, sr2 = (sigma / cutoff) ** 6, (sigma / r) ** 6
        shift = -4 * eps * (sr1 * sr1 - sr1)
        return 4 * eps * (sr2 * sr2 - sr2) + shift

#########################################################################
def bond_harmonic(r, K, r0):
    """
    Compute the harmonic bond potential for the given bond length with the
    given parameters.
    """
    d = r - r0
    return K * d * d

#########################################################################
def bond_fene(r, K, R0, eps, sigma):
    """
    Compute the FENE bond potential for the given bond length with the given
    parameters. 
    """
    U = -0.5 * K * R0 * R0 * np.log(1 - (r / R0) ** 2)
    rmin = (2 ** (1 / 6)) * sigma
    if r < rmin:
        sr = (sigma / r) ** 6
        U += 4 * eps * (sr * sr - sr) + eps

    return U

#########################################################################
def angle_cosine(theta, K):
    """
    Compute the cosine angle potential for the given bond angle with the
    given parameters.
    """
    return K * (1 + np.cos(theta))

#########################################################################
def angle_cosine_delta(theta, K, theta0):
    """
    Compute the cosine/delta angle potential for the given bond angle with
    the given parameters.
    """
    return K * (1 - np.cos(theta - theta0))

#########################################################################
def dihedral_harmonic(phi, K, d, n):
    """
    Compute the harmonic dihedral potential for the given dihedral angle
    with the given parameters. 
    """
    return K * (1 + d * np.cos(n * phi))

#########################################################################
#           FUNCTIONS FOR PLOTTING POLYMERS AND CROSSLINKERS            #
#########################################################################
def plot_polymers(polymers, ax, dims=(0, 1)):
    """
    Plot the given polymers on the given axes along the given dimensions. 

    The dimensions `dims` can be set to a tuple of two ints, each of which 
    specifies whether the x (0), y (1), or z (2) coordinates are plotted.

    Parameters
    ----------
    polymers : list of `Polymer` objects
        List of `Polymer` objects to be plotted.
    ax : `matplotlib.pyplot.Axes`
        Axes object.  
    dims : tuple of two ints
        Indicates the dimensions that should be plotted (x, y, or z).

    Returns
    -------
    Modified Axes object. 
    """
    if not (type(dims) == tuple and len(dims) == 2):
        raise ValueError('dims should be a tuple of length 2')
    if not (dims[0] in [0, 1, 2] and dims[1] in [0, 1, 2]):
        raise ValueError('Invalid value given for dims: {}'.format(dims))

    for polymer in polymers:
        ax.plot(polymer.coords[:, dims[0]], polymer.coords[:, dims[1]], marker='.')

    return ax

#########################################################################
def plot_crosslinkers(crosslinkers, ax, dims=(0, 1)):
    """
    Plot the given crosslinkers on the given axes along the given dimensions. 

    The dimensions `dims` can be set to a tuple of two ints, each of which 
    specifies whether the x (0), y (1), or z (2) coordinates are plotted.

    Parameters
    ----------
    crosslinkers : list
        List of `AtomicCrosslinker` or `TetrahedralCrosslinker` objects to
        be plotted.
    ax : `matplotlib.pyplot.Axes`
        Axes object.  
    dims : tuple of two ints
        Indicates the dimensions that should be plotted (x, y, or z).

    Returns
    -------
    Modified Axes object. 
    """
    if not (type(dims) == tuple and len(dims) == 2):
        raise ValueError('dims should be a tuple of length 2')
    if not (dims[0] in [0, 1, 2] and dims[1] in [0, 1, 2]):
        raise ValueError('Invalid value given for dims: {}'.format(dims))

    for crosslinker in crosslinkers:
        # Plot each crosslinker center 
        ax.plot(
            [crosslinker.coords[0, dims[0]]], [crosslinker.coords[0, dims[1]]],
            marker='x'
        )

    return ax

#########################################################################
#               FUNCTIONS FOR WRITING LAMMPS INPUT FILES                #
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
def write_masses(masses, fmt=':.10f'):
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
def write_lj_coefs(type_i, type_j, lj_coefs, fmt=':.10f'):
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
        type_i,
        type_j,
        lj_coefs['eps'],
        lj_coefs['sigma'],
        lj_coefs['cutoff']
    )

#########################################################################
def write_bond_coefs(bond_idx, bond_coefs, bond_style, write_style=True,
                     fmt=':.10f'):
    """
    Return a string containing the given FENE bond energy coefficients to 
    be outputted into a LAMMPS data file. 

    Parameters
    ----------
    bond_idx : int
        Bond index.
    bond_coefs : dict
        Dict containing the bond energy coefficients.
    bond_style : str
        String indicating the bond style; should be either 'fene' or
        'harmonic'.
    write_style : bool
        If True, write the bond style.
    fmt : str
        Format string for each floating-point value. 

    Returns
    -------
    Output string.
    """
    if bond_style == 'harmonic':
        return ('{} {}{' + fmt + '} {' + fmt + '}\n').format(
            bond_idx,
            'harmonic ' if write_style else '',
            bond_coefs['K'],
            bond_coefs['R0']
        )
    elif bond_style == 'fene':
        return ('{} {}{' + fmt + '} {' + fmt + '} {' + fmt + '} {' + fmt + '}\n').format(
            bond_idx,
            'fene ' if write_style else '',
            bond_coefs['K'],
            bond_coefs['R0'],
            bond_coefs['eps'],
            bond_coefs['sigma']
        )
    else:
        raise ValueError('Unrecognized bond style')

#########################################################################
def write_angle_coefs(angle_idx, angle_coefs, write_style=True, fmt=':.10f'):
    """
    Return a string containing the given angle energy coefficients to 
    be outputted into a LAMMPS data file.

    The angle style is assumed to be cosine/delta. 

    Parameters
    ----------
    angle_idx : int
        Angle index.
    angle_coefs : dict
        Dict containing the angle energy coefficients.
    write_style : bool
        If True, write the angle style.
    fmt : str
        Format string for each floating-point value. 

    Returns
    -------
    Output string.
    """
    return ('{} {}{' + fmt + '} {' + fmt + '}\n').format(
        angle_idx,
        'cosine/delta ' if write_style else '',
        angle_coefs['K'],
        angle_coefs['theta0'] * 180 / np.pi   # LAMMPS takes degrees as input
    )

#########################################################################
def write_dihedral_coefs(dihedral_idx, dihedral_coefs, write_style=True,
                         fmt=':.10f'):
    """
    Return a string containing the given dihedral energy coefficients to 
    be outputted into a LAMMPS data file.

    The dihedral style is assumed to be harmonic. 

    Parameters
    ----------
    dihedral_idx : int 
        Dihedral index.
    dihedral_coefs : dict
        Dict containing the dihedral energy coefficients.
    write_style : bool
        If True, write the dihedral style.
    fmt : str
        Format string for each floating-point value. 

    Returns
    -------
    Output string.
    """
    return ('{} {}{' + fmt + '} {:d} {:d}\n').format(
        dihedral_idx,
        'harmonic ' if write_style else '',
        dihedral_coefs['K'],
        dihedral_coefs['d'],
        dihedral_coefs['n']
    )

#########################################################################
def write_coords(polymers, crosslinkers, polymer_atom_type=_polymer_atom_type,
                 crosslinker_atom_type=_crosslinker_atom_type,
                 crosslinker_sticky_atom_type=_crosslinker_sticky_atom_type,
                 fmt=':.10f'):
    """
    Return a string containing the given polymer and crosslinker coordinates
    to be outputted into a LAMMPS data file.

    Parameters
    ----------
    polymers : list
        List of `Polymer` objects.
    crosslinkers : list
        List of `AtomicCrosslinker` or `TetrahedralCrosslinker` objects.
    polymer_atom_type : int
        Atom type for each monomer. 
    crosslinker_atom_type : int
        Atom type for each non-sticky crosslinker atom.
    crosslinker_sticky_atom_type : int
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
                atom_id, molecule_id, polymer_atom_type,
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
        for j in range(crosslinker.coords.shape[0]):
            # For each atom, write the line:
            # [atom_id] [molecule_id] [atom_type] [xcoord] [ycoord] [zcoord]
            outstr += ('{} {} {} {' + fmt + '} {' + fmt + '} {' + fmt + '}\n').format(
                atom_id, molecule_id, crosslinker.atom_types[j],
                crosslinker.coords[j, 0],
                crosslinker.coords[j, 1],
                crosslinker.coords[j, 2]
            )
            atom_id += 1

    return outstr

#########################################################################
def write_bonds(polymers, crosslinkers, polymer_bond_type=_polymer_bond_type,
                crosslinker_bond_type=_crosslinker_bond_type):
    """
    Return a string containing the bonds within the given set of polymers 
    and crosslinkers to be outputted into a LAMMPS data file.

    Parameters
    ----------
    polymers : list
        List of `Polymer` objects.
    crosslinkers : list
        List of `AtomicCrosslinker` or `TetrahedralCrosslinker` objects.
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
def write_angles(polymers, crosslinkers, polymer_angle_types,
                 crosslinker_angle_type=_crosslinker_angle_type):
    """
    Return a string containing the angles within the given set of polymers 
    to be outputted into a LAMMPS data file. 

    Parameters
    ----------
    polymers : list
        List of `Polymer` objects.
    crosslinkers : list
        List of `AtomicCrosslinker` or `TetrahedralCrosslinker` objects.
    polymer_angle_types : 2-D `numpy.ndarray`
        Angle types along each polymer. 
    crosslinker_angle_type : int
        Angle type within each crosslinker. 

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
                angle_id, polymer_angle_types[i, j - 1],
                atom_ids[j - 1], atom_ids[j], atom_ids[j + 1]
            )
            angle_id += 1
        first_atom_id = last_atom_id + 1

    # ... then write the crosslinker angles  
    first_atom_id = sum(len(polymer) for polymer in polymers) + 1
    for i, crosslinker in enumerate(crosslinkers):
        for j in range(1, crosslinker.coords.shape[0]):
            for k in range(j + 1, crosslinker.coords.shape[0]):
                # For each angle, write the line:
                # [angle_id] [angle_type] [first_atom_id] [second_atom_id] [third_atom_id]
                outstr += '{} {} {} {} {}\n'.format(
                    angle_id, crosslinker_angle_type,
                    first_atom_id + j,
                    first_atom_id,
                    first_atom_id + k
                )
                angle_id += 1
        first_atom_id += crosslinker.coords.shape[0]

    return outstr

#########################################################################
def write_dihedrals(polymers, polymer_dihedral_types):
    """
    Return a string containing the dihedral angles within the given set of
    polymers to be outputted into a LAMMPS data file. 

    Parameters
    ----------
    polymers : list
        List of `Polymer` objects.
    polymer_dihedral_type : int
        Dihedral angle type within each polymer.

    Returns 
    -------
    Output string.
    """
    outstr = ''
    dihedral_id = 1    # Number the angles in the string as 1, 2, 3, ...
    
    # Write the dihedral angles ... 
    first_atom_id = 1
    for i, polymer in enumerate(polymers):
        last_atom_id = first_atom_id + len(polymer) - 1
        atom_ids = list(range(first_atom_id, last_atom_id + 1))
        for j in range(2, len(polymer) - 1):
            # For each angle, write the line: 
            # [dihedral_id] [dihedral_type] [first_atom_id] [second_atom_id] [third_atom_id] [fourth_atom_id]
            outstr += '{} {} {} {} {} {}\n'.format(
                dihedral_id, polymer_dihedral_types[i, j - 2], 
                atom_ids[j - 2], atom_ids[j - 1], atom_ids[j], atom_ids[j + 1]
            )
            dihedral_id += 1
        first_atom_id = last_atom_id + 1

    return outstr

#########################################################################
def write_init_config(polymers, crosslinkers, bond_length, crosslinker_style,
                      crosslinker_radius, monomer_mass, crosslinker_mass,
                      lj_coefs, bond_coefs, bond_styles, angle_coefs,
                      polymer_angle_types, dihedral_coefs, polymer_dihedral_types,
                      xmin, xmax, ymin, ymax, zmin, zmax, outfilename):
    """
    Write the given initial configuration of polymer and crosslinker
    coordinates to file.

    Parameters
    ----------
    polymers : list
        List of `Polymer` objects.
    crosslinkers : list
        List of `AtomicCrosslinker` or `TetrahedralCrosslinker` objects.
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
        Bond potential coefficients.
    bond_styles : list
        Bond styles, each of which should be 'fene' or 'harmonic'.
    angle_coefs : list of dicts
        Angle potential coefficients. There should be two dicts, one for the
        polymers and another for the crosslinkers.
    polymer_angle_types : 

    dihedral_coefs : dict
        Dihedral potential coefficients. None if no potential is to be enforced.
    polymer_dihedral_types : 

    xmin, xmax : float, float
        Minimum and maximum x-coordinates.
    ymin, ymax : float, float
        Minimum and maximum y-coordinates.
    zmin, zmax : float, float
        Minimum and maximum z-coordinates.
    outfilename : str
        Output filename. 
    """
    n_polymers = len(polymers)
    polymer_length = len(polymers[0])
    n_crosslinkers = len(crosslinkers)
    text = ''

    # If there are no crosslinkers ... 
    if crosslinker_style == 'none':
        header = (
            'Test system of {} polymers of length {}, bond length = {}\n\n'.format(
                n_polymers, polymer_length, bond_length
            )
        )
        text += '{} atoms\n{} bonds\n{} angles\n{} dihedrals\n0 impropers\n\n'.format(
            n_polymers * polymer_length + n_crosslinkers,
            n_polymers * (polymer_length - 1),
            n_polymers * (polymer_length - 2),
            0 if dihedral_coefs is None else n_polymers * (polymer_length - 3)
        )
        text += (
            '1 atom types\n1 bond types\n{} angle types\n{} dihedral types\n'
            '0 improper types\n\n'.format(
                len(angle_coefs),
                0 if dihedral_coefs is None else len(dihedral_coefs)
            )
        )
        text += write_box_dims(xmin, xmax, ymin, ymax, zmin, zmax) + '\n\n'
        text += 'Masses\n\n1 {}\n\n'.format(monomer_mass)
    # If the crosslinkers are single atoms ... 
    elif crosslinker_style == 'atomic':
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
            n_polymers * (polymer_length - 2) + 6 * n_crosslinkers
        )
        text += (
            '3 atom types\n3 bond types\n3 angle types\n0 dihedral types\n'
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
            i + 1, bond_coefs[i], bond_styles[i], write_style=(len(bond_coefs) > 1)
        )
    text += '\n'

    # Write the angle potential coefficients 
    #
    # A dict of coefficients should be present for every angle type
    text += 'Angle Coeffs\n\n'
    for i in range(len(angle_coefs)):
        text += write_angle_coefs(i + 1, angle_coefs[i], write_style=False)
    text += '\n'

    # Write the dihedral potential coefficients
    #
    # A dict of coefficients should be present for every angle type
    if dihedral_coefs is not None:
        text += 'Dihedral Coeffs\n\n'
        for i in range(len(dihedral_coefs)):
            text += write_dihedral_coefs(i + 1, dihedral_coefs[i], write_style=False)
        text += '\n'

    # Write atomic coordinates, bonds, and angles  
    text += 'Atoms\n\n{}\n'.format(write_coords(polymers, crosslinkers))
    text += 'Bonds\n\n{}\n'.format(write_bonds(polymers, crosslinkers))
    text += 'Angles\n\n{}\n'.format(
        write_angles(polymers, crosslinkers, polymer_angle_types)
    )
    if dihedral_coefs is not None:
        text += 'Dihedrals\n\n{}\n'.format(
            write_dihedrals(polymers, polymer_dihedral_types)
        )

    with open(outfilename, 'w') as f:
        f.write(header + text)

