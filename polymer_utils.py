"""
Utility functions for defining initial conformations of polymers of fixed
bond length and crosslinkers. 

Authors:
    Kee-Myoung Nam

Last updated:
    1/22/2024
"""
import numpy as np
import matplotlib.pyplot as plt

#########################################################################
def random_dir(rng):
    """
    Generate a random unit vector in 3-D space.

    Parameters
    ----------
    rng : `numpy.random.Generator`
        Random number generator.

    Returns
    -------
    Random unit vector in 3-D space. 
    """
    v = rng.normal(size=(3,))
    return v / np.linalg.norm(v)

#########################################################################
def generate_polymers(n, m, bond_length, rng, xmin, xmax, ymin, ymax, zmin,
                      zmax, eps1, eps2):
    """
    Generate a set of `n` random coils, each containing `m` monomers and
    possessing the given bond length, in the given 3-D box. 

    Parameters
    ----------
    n : int
        Number of polymers.
    m : int
        Number of monomers per polymer.
    bond_length : float
        Bond length.
    rng : `numpy.random.Generator`
        Random number generator.
    xmin : float
        Minimum x-coordinate. 
    xmax : float
        Maximum x-coordinate. 
    ymin : float
        Minimum y-coordinate. 
    ymax : float
        Maximum y-coordinate. 
    zmin : float
        Minimum z-coordinate. 
    zmax : float
        Maximum z-coordinate.
    eps1 : float
        Inter-polymer distance threshold for sampling.
    eps2 : float
        Intra-polymer distance threshold for sampling.

    Returns
    -------
    A 3-D array of shape `(n, m, 3)` containing the polymer coordinates.
    """
    vmin = np.array([xmin, ymin, zmin])
    dims = np.array([xmax, ymax, zmax]) - vmin

    coords = np.zeros((n, m, 3))
    for i in range(n):
        # Initialize the i-th polymer
        polymer = np.zeros((m, 3))

        # Get previously sampled polymer coordinates
        if i > 0:
            prev_coords = coords[:i, :, :].reshape((i * m, 3))
        else:
            prev_coords = None

        # Sample a random starting point within the box, resampling until 
        # the point is sufficiently distant from all previously sampled points
        p = rng.random((3,)) * dims + vmin
        if i > 0:
            dist_to_p = np.linalg.norm(prev_coords - p, axis=1).min()
            while dist_to_p < eps1:
                p = rng.random((3,)) * dims + vmin
                dist_to_p = np.linalg.norm(prev_coords - p, axis=1).min()
        polymer[0, :] = p

        # Sample the next monomer position, resampling until:
        # (1) the point lies within the box,
        # (2) the point is sufficiently distant from all previously sampled
        #     polymers, and
        # (3) the point is sufficiently distant from all previously sampled
        #     monomers within the current polymer
        for j in range(m - 1):
            q = polymer[j, :] + bond_length * random_dir(rng)
            q_in_box = (
                q[0] >= xmin and q[0] <= xmax and q[1] >= ymin and q[1] <= ymax
                and q[2] >= zmin and q[2] <= zmax
            )
            if i > 0:
                dist_to_q_inter = np.linalg.norm(prev_coords - q, axis=1).min()
            else:
                dist_to_q_inter = np.inf
            dist_to_q_intra = np.linalg.norm(polymer[:j+1, :] - q, axis=1).min()
            while not (q_in_box and dist_to_q_inter >= eps1 and dist_to_q_intra >= eps2):
                q = polymer[j, :] + bond_length * random_dir(rng)
                q_in_box = (
                    q[0] >= xmin and q[0] <= xmax and q[1] >= ymin and q[1] <= ymax
                    and q[2] >= zmin and q[2] <= zmax
                )
                if i > 0:
                    dist_to_q_inter = np.linalg.norm(prev_coords - q, axis=1).min()
                else:
                    dist_to_q_inter = np.inf
                dist_to_q_intra = np.linalg.norm(polymer[:j+1, :] - q, axis=1).min()
            polymer[j+1, :] = q

        # Add polymer coordinates
        coords[i, :, :] = polymer

    return coords

#########################################################################
def plot_polymers(coords, ax, dims=(0, 1)):
    """
    Plot the given polymers on the given axes along the given dimensions. 

    The dimensions `dims` can be set to a tuple of two ints, each of which 
    specifies whether the x (0), y (1), or z (2) coordinates are plotted.
    """
    if not (type(dims) == tuple and len(dims) == 2):
        raise ValueError('dims should be a tuple of length 2')
    if not (dims[0] in [0, 1, 2] and dims[1] in [0, 1, 2]):
        raise ValueError('Invalid value given for dims: {}'.format(dims))

    for i in range(coords.shape[0]):
        ax.plot(coords[i, :, dims[0]], coords[i, :, dims[1]], marker='.')

    return ax

#########################################################################
def generate_crosslinker_centers(polymer_coords, n, rng, xmin, xmax, ymin,
                                 ymax, zmin, zmax, eps):
    """
    Generate a set of `n` crosslinkers to superimpose on the given set of 
    polymers in the given 3-D box.

    The generated points are the centers of the crosslinker molecules.

    Parameters
    ----------
    polymer_coords : `numpy.ndarray`
        Pre-generated 3-D array of polymer coordinates.
    n : int
        Number of crosslinkers.
    rng : `numpy.random.Generator`
        Random number generator.
    xmin : float
        Minimum x-coordinate. 
    xmax : float
        Maximum x-coordinate. 
    ymin : float
        Minimum y-coordinate. 
    ymax : float
        Maximum y-coordinate. 
    zmin : float
        Minimum z-coordinate. 
    zmax : float
        Maximum z-coordinate.
    eps : float
        Crosslinker radius. 

    Returns
    -------
    A 2-D array of shape `(n, 3)` containing the crosslinker center
    coordinates.
    """
    vmin = np.array([xmin, ymin, zmin])
    dims = np.array([xmax, ymax, zmax]) - vmin

    # Reshape the polymer coordinates and keep track of the sampled 
    # crosslinker coordinates in the same array
    sampled_coords = np.vstack((polymer_coords.reshape((-1, 3)), np.zeros((n, 3))))

    j = polymer_coords.shape[0] * polymer_coords.shape[1]
    for i in range(n):
        # Sample a point within the box, resampling until the point is 
        # sufficiently distant from all previously sampled points
        p = rng.random((3,)) * dims + vmin
        dist_to_p = np.linalg.norm(sampled_coords - p, axis=1).min()
        while dist_to_p < eps:
            p = rng.random((3,)) * dims + vmin
            dist_to_p = np.linalg.norm(sampled_coords - p, axis=1).min()
        sampled_coords[j+i, :] = p

    return sampled_coords[j:, :]

#########################################################################
def write_molecule_coords(polymer_coords, crosslinker_coords, polymer_type=1,
                          crosslinker_type=2, fmt=':.6f'):
    """
    Return a string containing the given polymer and crosslinker coordinates
    to be outputted into a LAMMPS data file.

    Parameters
    ----------
    polymer_coords : `numpy.ndarray`
        3-D array of polymer coordinates.
    crosslinker_coords : `numpy.ndarray`
        3-D array of crosslinker coordinates.
    polymer_type : int
        Type of polymer atom. 
    crosslinker_type : int
        Type of crosslinker atom. 
    fmt : str
        Format string for each coordinate.

    Returns
    -------
    Output string to be included in a LAMMPS data file.  
    """
    outstr = ''
    atom_id = 1

    # Write the polymer coordinates first ...
    n, m, _ = polymer_coords.shape
    for i in range(n):
        molecule_id = i + 1
        for j in range(m):
            outstr += ('{} {} {} {' + fmt + '} {' + fmt + '} {' + fmt + '}\n').format(
                atom_id, molecule_id, polymer_type,
                polymer_coords[i, j, 0],
                polymer_coords[i, j, 1],
                polymer_coords[i, j, 2]
            )
            atom_id += 1

    # ... then write the crosslinker coordinates
    n, m, _ = crosslinker_coords.shape
    for i in range(n):
        molecule_id = polymer_coords.shape[0] + i + 1
        for j in range(m):
            outstr += ('{} {} {} {' + fmt + '} {' + fmt + '} {' + fmt + '}\n').format(
                atom_id, molecule_id, crosslinker_type,
                crosslinker_coords[i, j, 0],
                crosslinker_coords[i, j, 1],
                crosslinker_coords[i, j, 2]
            )
            atom_id += 1

    return outstr

#########################################################################
def write_molecule_bonds(polymer_coords, crosslinker_coords, polymer_bond_type=1,
                         crosslinker_bond_type=2):
    """
    Return a string containing the bonds within the given set of polymers 
    and crosslinkers to be outputted into a LAMMPS data file.

    Parameters
    ----------
    polymer_coords : `numpy.ndarray`
        3-D array of polymer coordinates.
    crosslinker_coords : `numpy.ndarray`
        3-D array of crosslinker coordinates.
    polymer_bond_type : int
        Type of bond within each polymer. 
    crosslinker_bond_type : int
        Type of bond within each crosslinker.

    Returns
    -------
    Output string to be included in a LAMMPS data file.  
    """
    outstr = ''
    bond_id = 1
    
    # Write the polymer coordinates first ...
    n, m, _ = polymer_coords.shape
    for i in range(n):
        atom_ids = [i * m + j + 1 for j in range(m)]
        bond_suffixes = [
            '{} {} {}'.format(polymer_bond_type, atom_ids[j], atom_ids[j+1])
            for j in range(m - 1)
        ]
        for suffix in bond_suffixes:
            outstr += '{} {}\n'.format(bond_id, suffix)
            bond_id += 1

    # ... then write the crosslinker coordinates
    polymer_total = n * m
    n, m, _ = crosslinker_coords.shape
    for i in range(n):
        atom_ids = [polymer_total + i * m + j + 1 for j in range(m)]
        bond_suffixes = [
            '{} {} {}'.format(crosslinker_bond_type, atom_ids[j], atom_ids[j+1])
            for j in range(m - 1)
        ]
        for suffix in bond_suffixes:
            outstr += '{} {}\n'.format(bond_id, suffix)
            bond_id += 1

    return outstr

