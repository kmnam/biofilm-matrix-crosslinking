"""
Utility functions and classes for defining initial conformations of polymers
of fixed bond length and tetrahedral crosslinkers.

Authors:
    Kee-Myoung Nam

Last updated:
    8/18/2024
"""
import numpy as np
import matplotlib.pyplot as plt

_polymer_atom_type = 1
_crosslinker_atom_type = 2
_crosslinker_sticky_atom_type = 3
_polymer_bond_type = 1
_crosslinker_bond_type = 2
_polymer_crosslinker_bond_type = 3
_polymer_angle_type = 1
_crosslinker_angle_type = 2
_polymer_crosslinked_angle_type = 3

#########################################################################
#                           HELPER FUNCTIONS                            #
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
def dist_periodic(x, y, xmin, xmax, ymin, ymax, zmin, zmax):
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
def generate_orthonormal_basis_3d(u, rng):
    """
    Generate an orthonormal basis that contains the given vector, which is
    assumed to be unit-length, in three dimensions.

    This is done by:
    1) sampling a random unit vector, subtracting its projection onto v (as
       in Gram-Schmidt), then normalizing to get v,
    3) then getting the cross product of u and v and normalizing to get w.  

    Then u, v, and w form an orthonormal basis. 
    """
    # First sample a random unit vector ... 
    v = random_dir(rng)

    # ... then project it onto u and normalize ... 
    v -= np.dot(v, u) * u
    v /= np.linalg.norm(v)

    # ... then get the cross product of u and v and normalize
    w = np.cross(u, v)
    w /= np.linalg.norm(w)

    # Return the two new vectors 
    return v, w

#########################################################################
def generate_next_atom(atom1_coords, atom2_coords, length, theta, rng):
    """
    Given the positions of two bonded atoms within a polymer, the length
    of the next bond, and the angle formed by the next bond, generate 
    a candidate position for the next atom.  
    """
    # Get the distance vector and direction from atom 1 to atom 2
    u = atom2_coords - atom1_coords
    u /= np.linalg.norm(u)

    # Randomly sample an orthonormal basis that contains the 2-1 direction
    # vector
    #
    # The other two vectors in this basis span the plane normal to the 2-1
    # direction vector (up to translation) 
    v, w = generate_orthonormal_basis_3d(-u, rng)

    # Rotate the direction vector from atom 2 to atom 1 about atom 2 by 
    # the given angle within the plane normal to w, which must contain u
    #
    # To do this, we rotate the vector (1, 0, 0) in the xy-plane, and
    # perform a change of basis in which x <-> -u, y <-> v, and z <-> w
    #
    # This yields a vector that is orthogonal to w and has the desired 
    # angle from -u 
    rot = np.array([
         [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]
    ])
    trans = np.hstack((-u.reshape((-1, 1)), v.reshape((-1, 1)), w.reshape((-1, 1))))
    u_new = trans @ (np.append(rot @ np.array([1, 0]), 0))

    # Get the position of the next atom
    return atom2_coords + length * u_new

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
#                           MOLECULE CLASSES                            #
#########################################################################
class Molecule:
    """
    A basic Molecule class that incorporates all attributes and functions
    common to both polymers and crosslinkers.

    This class is not meant to be used by itself, but rather as a base 
    class for `Polymer`, `AtomicCrosslinker`, and `TetrahedralCrosslinker`. 
    """
    def __init__(self, coords=None, atom_types=None, bonds=None):
        """
        Initialize a Molecule object with the given monomer coordinates.

        Parameters
        ----------
        coords : numpy.ndarray
            Array of atomic coordinates.
        atom_types : int or numpy.ndarray
            Array of atom types. (If a single int, then all atoms are assumed
            to have the same type.)
        bonds : dict
            Dict of bonds in the molecule. Each key is a pair of atom indices
            (zero-indexed), (i, j) with i < j, and the corresponding value is
            the bond type.
        """
        if coords is None:
            self.coords = np.zeros((0, 3), dtype=np.float64)
        else:
            self.coords = coords

        # Check that bonds are valid (each key should refer to atoms that
        # exist in the molecule)
        if bonds is None:
            self.bonds = {}
        else:
            for i, j in bonds:
                if i >= j or i >= self.coords.shape[0] or j >= self.coords.shape[0]:
                    raise RuntimeError('Invalid bonds specified')
            self.bonds = bonds

        # Check that atom types have been correctly specified, given the
        # coordinates 
        if coords is not None and atom_types is None:
            raise RuntimeError('Atom types must be specified if coordinates are specified')
        elif atom_types is None:
            self.atom_types = np.array([], dtype=np.int64)
        elif isinstance(atom_types, int):
            self.atom_types = atom_types * np.ones(self.coords.shape[0], dtype=np.int64)
        else:
            self.atom_types = np.array(atom_types, dtype=np.int64)
            if self.atom_types.shape[0] != self.coords.shape[0]:
                raise RuntimeError('Invalid array of atom types specified')

    #####################################################################
    def center_of_mass(self):
        """
        Return the center of mass of the Molecule object.

        Returns
        -------
        Center of mass of the atoms. 
        """
        return np.mean(self.coords, axis=0)

    #####################################################################
    def min_deviation(self, p):
        """
        Return the minimum deviation from the given point of the Molecule
        object, i.e., the distance to the closest atom.

        Parameters
        ----------
        p : numpy.ndarray
            Input point. 

        Returns
        -------
        Distance between input point and the closest atom.
        """
        return np.linalg.norm(self.coords - p, axis=1).min()

    #####################################################################
    def max_deviation(self, p):
        """
        Return the maximum deviation from the given point of the Molecule
        object, i.e., the distance to the furthest atom.

        Parameters
        ----------
        p : numpy.ndarray
            Input point. 

        Returns
        -------
        Distance between input point and the furthest atom.
        """
        return np.linalg.norm(self.coords - p, axis=1).max()

#########################################################################
class Polymer(Molecule):
    """
    A basic polymer class. 
    """
    def __init__(self, coords=None, atom_types=None):
        """
        Initialize a `Polymer` object with the given monomer coordinates.

        Note that a `Polymer` object can be empty. 

        The monomers in the `Polymer` object are assumed to be consecutive, 
        and the bonds in the `Polymer` object are accordingly generated.

        Parameters
        ----------
        coords : numpy.ndarray
            Array of monomer coordinates.
        atom_types : numpy.ndarray
            Atom types for the monomers.
        """
        super().__init__(coords, atom_types, bonds=None)
        if coords is None:
            self.length = 0
        else:
            self.length = coords.shape[0]
        self.bonds = {}
        for i in range(self.length - 2):
            self.bonds[(i, i + 1)] = _polymer_bond_type

    #####################################################################
    def __len__(self):
        """
        Return the length of the `Polymer` object. 

        Returns
        -------
        Length of the `Polymer` object. 
        """
        return self.length

    #####################################################################
    def append(self, p, atom_type):
        """
        Add a monomer to the end of the `Polymer` object.

        Parameters
        ----------
        p : numpy.ndarray
            Array of coordinates for the new monomer.
        atom_type : int 
            Atom type for the new monomer. 
        """
        self.length += 1
        self.coords = np.vstack((self.coords, p.reshape(1, -1)))
        self.atom_types = np.append(self.atom_types, atom_type)
        self.bonds[(self.length - 2, self.length - 1)] = _polymer_bond_type

    #####################################################################
    def pop(self):
        """
        Remove the last monomer from the `Polymer` object.
        """
        self.length -= 1
        self.coords = self.coords[:self.length, :]
        self.atom_types = self.atom_types[:self.length]
        self.bonds.pop((self.length - 1, self.length))

    #####################################################################
    def clear(self):
        """
        Clear the `Polymer` object.
        """
        self.length = 0
        self.coords = np.zeros((0, 3), dtype=np.float64)
        self.atom_types = np.zeros((0,), dtype=np.int64)
        self.bonds = {}

#########################################################################
class AtomicCrosslinker(Molecule):
    """
    A basic atomic crosslinker class. 
    """
    def __init__(self, p, atom_type):
        """
        Define the crosslinker at the given point.

        Parameters
        ----------
        p : numpy.ndarray
            Input point.
        atom_type : int
            Atom type. 
        """
        super().__init__(np.array(p).reshape(1, -1), atom_type)

    #####################################################################
    def translate(self, x, y, z):
        """
        Translate the crosslinker by the given x-, y-, and z-increments.

        Parameters
        ----------
        x : float
            x-increment. 
        y : float
            y-increment.
        z : float
            z-increment.
        """
        self.coords[0, 0] += x
        self.coords[0, 1] += y
        self.coords[0, 2] += z

#########################################################################
class TetrahedralCrosslinker(Molecule):
    """
    A basic tetrahedral crosslinker class.

    Each crosslinker consists of 5 vertices (one in the center, 4 on 
    the boundary) and 4 edges (each emanating from the center to a vertex
    on the boundary), as in a methane molecule.  
    """
    def __init__(self, center, radius, atom_types):
        """
        Define the coordinates of a crosslinker at the given center with
        the given radius.

        Parameters
        ----------
        center : numpy.ndarray
            Input center.
        radius : float
            Input radius.
        atom_types : numpy.ndarray
            Array of atom types. 
        """
        # First define at the origin, then translate to the given center
        self.coords = np.zeros((5, 3), dtype=np.float64)
        self.coords[0, :] = [0, 0, 0]
        self.coords[1, :] = [0, 0, 1]
        self.coords[2, :] = [np.sqrt(8 / 9), 0, -1 / 3]
        self.coords[3, :] = [-np.sqrt(2 / 9), np.sqrt(2 / 3), -1 / 3]
        self.coords[4, :] = [-np.sqrt(2 / 9), -np.sqrt(2 / 3), -1 / 3]
        self.coords *= radius
        self.coords += center
        self.radius = radius
        if isinstance(atom_types, int):
            self.atom_types = atom_types * np.ones(self.coords.shape[0], dtype=np.int64)
        else:
            self.atom_types = np.array(atom_types, dtype=np.int64)
            if self.atom_types.shape[0] != self.coords.shape[0]:
                raise RuntimeError('Invalid array of atom types')
        self.bonds = {(0, i): _crosslinker_bond_type for i in range(1, 5)}

    #####################################################################
    def translate(self, x, y, z):
        """
        Translate the crosslinker by the given x-, y-, and z-increments.

        Parameters
        ----------
        x : float
            x-increment. 
        y : float
            y-increment.
        z : float
            z-increment.
        """
        self.coords[:, 0] += x
        self.coords[:, 1] += y
        self.coords[:, 2] += z

    #####################################################################
    def rotate(self, alpha, beta, gamma):
        """
        Rotate the crosslinker by the given Tait-Bryan angles.

        Parameters
        ----------
        alpha : float
            Angle to rotate about z-axis. 
        beta : float
            Angle to rotate about y-axis.
        gamma : float
            Angle to rotate about x-axis.
        """
        # Translate the crosslinker to the origin
        center = self.coords[0, :]
        self.coords[:, 0] -= center[0]
        self.coords[:, 1] -= center[1]
        self.coords[:, 2] -= center[2]

        # Rotate by alpha about z-axis, by beta about y-axis, and by gamma
        # about x-axis 
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(gamma), -np.sin(gamma)],
                [0, np.sin(gamma), np.cos(gamma)]
            ],
            dtype=np.float64
        )
        Ry = np.array(
            [
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]
            ],
            dtype=np.float64
        )
        Rz = np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1]
            ],
            dtype=np.float64
        )
        for i in range(self.coords.shape[0]):
            self.coords[i, :] = Rz @ Ry @ Rx @ self.coords[i, :].T

        # Translate the crosslinker back to the original center
        self.coords[:, 0] += center[0]
        self.coords[:, 1] += center[1]
        self.coords[:, 2] += center[2]

#########################################################################
#          FUNCTIONS FOR GENERATING POLYMERS AND CROSSLINKERS           #
#########################################################################
def generate_polymers(n, polymer_length, bond_length, angle_dist, rng, xmin,
                      xmax, ymin, ymax, zmin, zmax, eps1, eps2,
                      atom_type=_polymer_atom_type, max_seed_per_polymer=1000,
                      max_backtrack_per_polymer=100, max_tries_per_bond=100):
    """
    Generate a set of `n` random coils as `Polymer` objects in the given
    3-D box. 

    Parameters
    ----------
    n : int
        Number of polymers.
    polymer_length : int
        Number of monomers per polymer.
    bond_length : float
        Bond length.
    angle_dist : function
        A callable that takes the given random number generator and samples
        a bond angle between each triplet of atoms in each polymer. 
    rng : `numpy.random.Generator`
        Random number generator.
    xmin, xmax : float, float
        Minimum and maximum x-coordinates. 
    ymin, ymax : float, float
        Minimum and maximum y-coordinates. 
    zmin, zmax : float, float
        Minimum and maximum z-coordinates. 
    eps1 : float
        Inter-polymer distance threshold for sampling.
    eps2 : float
        Intra-polymer distance threshold for sampling.
    atom_type : int
        Atom type for each atom in each `Polymer` object. 
    max_seed_per_polymer : int
        Maximum number of starting monomer positions to be sampled for any
        `Polymer` object. If this number is exceeded, a `RuntimeError` is
        raised.
    max_backtrack_per_polymer : int
        Maximum number of backtracking events for any `Polymer` object. This 
        function backtracks along a `Polymer` object (i.e., removes the
        previously sampled monomer position) if it samples more than
        `max_tries_per_bond` positions for the next monomer. If the number of
        backtracking events for a `Polymer` object is exceeded, then the
        function "starts over" by sampling a new starting monomer position.
    max_tries_per_bond : int
        Maximum number of sampling attempts for the (i+1)-th monomer from the
        i-th monomer in a `Polymer` object. If this number is exceeded, the 
        function backtracks along the `Polymer` object, i.e., resamples the
        i-th monomer position. 

    Returns
    -------
    A list of generated `Polymer` objects.
    """
    # Get the dimensions of the box 
    vmin = np.array([xmin, ymin, zmin])
    vmax = np.array([xmax, ymax, zmax])
    dims = vmax - vmin

    # Define a function that tests whether a point lies within the box 
    within_box = lambda p: ((p >= vmin).all() and (p <= vmax).all())

    # Maintain a list of Polymer objects 
    polymers = []

    # Number of times the polymer has been seeded (its 0-th monomer position
    # determined)
    n_seed = 0

    # Number of times a subsequent monomer position has been sampled
    n_tries = 0

    # Number of backtracks while sampling a given polymer 
    n_backtrack = 0

    # Generate the i-th polymer ... 
    for i in range(n):
        polymer_i = Polymer(coords=None, atom_types=None)
        j = 0
        while j < polymer_length:
            # If j == 0, sample a random starting point that is not too close
            # to any previously generated polymer
            if j == 0:
                # Sample a monomer position until an acceptable one is chosen
                p = rng.random((3,)) * dims + vmin
                n_seed += 1
                near_inter = any(
                    polymer.min_deviation(p) < eps1 for polymer in polymers
                )
                while near_inter and n_seed <= max_seed_per_polymer:
                    p = rng.random((3,)) * dims + vmin
                    near_inter = any(
                        polymer.min_deviation(p) < eps1 for polymer in polymers
                    )
                    n_seed += 1
                # If the maximum number of attempts at seeding the polymer has
                # been exceeded, raise an Exception
                if n_seed > max_seed_per_polymer:
                    raise RuntimeError('Exceeded max number of seeds for polymer')
                # Otherwise, append the new monomer 
                polymer_i.append(p, atom_type)
                j += 1
            # If j == 1, sample a second monomer position by the given bond
            # length in any direction
            elif j == 1:
                # Sample a monomer position until an acceptable one is chosen
                p = polymer_i.coords[0, :] + bond_length * random_dir(rng)
                n_tries += 1
                near_inter = any(
                    polymer.min_deviation(p) < eps1 for polymer in polymers
                )
                while (
                    (not within_box(p) or near_inter) and
                    n_backtrack <= max_backtrack_per_polymer and
                    n_tries <= max_tries_per_bond
                ):
                    p = polymer_i.coords[0, :] + bond_length * random_dir(rng)
                    near_inter = any(
                        polymer.min_deviation(p) < eps1 for polymer in polymers
                    )
                    n_tries += 1
                # If the maximum number of backtracks per polymer has been 
                # exceeded, then start fresh 
                if n_backtrack > max_backtrack_per_polymer:
                    polymer_i.clear()
                    j = 0
                    n_backtrack = 0
                    n_tries = 0
                # If the maximum number of attempts at sampling this monomer
                # position has been exceeded, backtrack by one monomer 
                if n_tries > max_tries_per_bond:
                    polymer_i.pop()
                    j -= 1
                    n_backtrack += 1
                    n_tries = 0
                # Otherwise, append the new monomer
                else:
                    polymer_i.append(p, atom_type)
                    j += 1
                    n_tries = 0
            # Otherwise, sample the j-th monomer positions, resampling until: 
            # (1) the point lies within the box 
            # (2) the point is sufficiently distant from all previously sampled 
            #     polymers, and 
            # (3) the point is sufficiently distant from all previously sampled 
            #     monomers within the given polymer
            else:
                # Sample a monomer position until an acceptable one is chosen
                p = generate_next_atom(
                    polymer_i.coords[j-2, :], polymer_i.coords[j-1, :],
                    bond_length, angle_dist(rng), rng
                )
                near_inter = any(
                    polymer.min_deviation(p) < eps1 for polymer in polymers
                )
                near_intra = (polymer_i.min_deviation(p) < eps2)
                n_tries += 1
                while (
                    (not within_box(p) or near_inter or near_intra) and
                    n_backtrack <= max_backtrack_per_polymer and
                    n_tries <= max_tries_per_bond
                ):
                    p = generate_next_atom(
                        polymer_i.coords[j-2, :], polymer_i.coords[j-1, :],
                        bond_length, angle_dist(rng), rng
                    )
                    near_inter = any(
                        polymer.min_deviation(p) < eps1 for polymer in polymers
                    )
                    near_intra = (polymer_i.min_deviation(p) < eps2)
                    n_tries += 1
                # If the maximum number of backtracks per polymer has been 
                # exceeded, then start fresh 
                if n_backtrack > max_backtrack_per_polymer:
                    polymer_i.clear()
                    j = 0
                    n_backtrack = 0
                    n_tries = 0
                # If the maximum number of attempts at sampling this monomer
                # position has been exceeded, backtrack by one monomer 
                if n_tries > max_tries_per_bond:
                    polymer_i.pop()
                    j -= 1
                    n_backtrack += 1
                    n_tries = 0
                # Otherwise, append the new monomer
                else:
                    polymer_i.append(p, atom_type)
                    j += 1
                    n_tries = 0

        # Keep track of generated polymer 
        polymers.append(polymer_i)

    return polymers

#########################################################################
def generate_atomic_crosslinkers(polymers, n, rng, xmin, xmax, ymin, ymax, zmin,
                                 zmax, eps1, eps2, atom_type=_crosslinker_atom_type):
    """
    Generate a set of `n` atomic crosslinkers to superimpose on the given
    set of polymers in the given 3-D box.

    Parameters
    ----------
    polymers : list
        Pre-generated list of `Polymer` objects.
    n : int
        Number of crosslinkers.
    rng : `numpy.random.Generator`
        Random number generator.
    xmin, xmax : float, float
        Minimum and maximum x-coordinates. 
    ymin, ymax : float, float
        Minimum and maximum y-coordinates. 
    zmin, zmax : float, float
        Minimum and maximum z-coordinates. 
    eps1 : float
        Minimum separation between each crosslinker and every monomer.
    eps2 : float
        Minimum separation between crosslinkers.
    atom_type : int
        Atom type for the sole atom in each `AtomicCrosslinker`. 

    Returns
    -------
    A list of `AtomicCrosslinker` objects.
    """
    # Get the dimensions of the box 
    vmin = np.array([xmin, ymin, zmin])
    vmax = np.array([xmax, ymax, zmax])
    dims = vmax - vmin

    # Maintain a list of AtomicCrosslinker objects
    crosslinkers = []

    for i in range(n):
        # Sample a point within the box, resampling until the point is 
        # sufficiently distant from all polymers and previously sampled 
        # crosslinkers
        p = rng.random((3,)) * dims + vmin
        near_polymers = any(
            poly.min_deviation(p) < eps1 for poly in polymers)
        near_crosslinkers = any(
            cross.min_deviation(p) < eps2 for cross in crosslinkers
        )
        while near_polymers or near_crosslinkers:
            p = rng.random((3,)) * dims + vmin
            near_polymers = any(poly.min_deviation(p) < eps1 for poly in polymers)
            near_crosslinkers = any(
                cross.min_deviation(p) < eps2 for cross in crosslinkers
            )

        # Generate a crosslinker at the sampled point
        crosslinker = AtomicCrosslinker(p, atom_type)
        crosslinkers.append(crosslinker)

    return crosslinkers

#########################################################################
def generate_tetrahedral_crosslinkers(polymers, n, radius, rng, xmin, xmax,
                                      ymin, ymax, zmin, zmax, eps1, eps2,
                                      atom_type=_crosslinker_atom_type,
                                      sticky_atom_type=_crosslinker_sticky_atom_type):
    """
    Generate a set of `n` tetrahedral crosslinkers to superimpose on the
    given set of polymers in the given 3-D box.

    Parameters
    ----------
    polymers : list
        Pre-generated list of `Polymer` objects.
    n : int
        Number of crosslinkers.
    radius : float
        Crosslinker radius.
    rng : `numpy.random.Generator`
        Random number generator.
    xmin, xmax : float, float
        Minimum and maximum x-coordinates. 
    ymin, ymax : float, float
        Minimum and maximum y-coordinates. 
    zmin, zmax : float, float
        Minimum and maximum z-coordinates. 
    eps1 : float
        Minimum separation between each crosslinker circumsphere and every
        monomer.  
    eps2 : float
        Minimum separation between each crosslinker circumsphere and every 
        other crosslinker atom.

    Returns
    -------
    A list of `TetrahedralCrosslinker` objects.
    """
    # Make the box a bit smaller, as we are sampling merely the crosslinker
    # centers and not the peripheral atoms 
    vmin = np.array([xmin + radius, ymin + radius, zmin + radius])
    vmax = np.array([xmax - radius, ymax - radius, zmax - radius])
    dims = vmax - vmin

    # Maintain a list of TetrahedralCrosslinker objects
    crosslinkers = []

    threshold1 = radius + eps1
    threshold2 = radius + eps2
    for i in range(n):
        # Sample a point within the box, resampling until the point is 
        # sufficiently distant from all polymers and previously sampled 
        # crosslinkers
        p = rng.random((3,)) * dims + vmin
        near_polymers = any(
            polymer.min_deviation(p) < threshold1 for polymer in polymers
        )
        near_crosslinkers = any(
            cross.min_deviation(p) < threshold2 for cross in crosslinkers
        )
        while near_polymers or near_crosslinkers:
            p = rng.random((3,)) * dims + vmin
            near_polymers = any(
                polymer.min_deviation(p) < threshold1 for polymer in polymers
            )
            near_crosslinkers = any(
                cross.min_deviation(p) < threshold2 for cross in crosslinkers
            )

        # Generate a tetrahedral crosslinker at the origin, designate one of
        # the peripheral atoms as sticky, then rotate and translate
        atom_types = atom_type * np.ones(5, dtype=np.int64)
        atom_types[rng.choice(np.arange(1, 5))] = sticky_atom_type
        crosslinker = TetrahedralCrosslinker([0, 0, 0], radius, atom_types)
        crosslinker.rotate(
            rng.random() * 2 * np.pi,
            rng.random() * 2 * np.pi,
            rng.random() * 2 * np.pi
        )
        crosslinker.translate(p[0], p[1], p[2])
        crosslinkers.append(crosslinker)

    return crosslinkers

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
def write_angles(polymers, crosslinkers, polymer_angle_type=_polymer_angle_type,
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
    polymer_angle_type : int
        Angle type within each polymer.
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
                angle_id, polymer_angle_type,
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
def write_init_config(polymers, crosslinkers, bond_length, crosslinker_style,
                      crosslinker_radius, monomer_mass, crosslinker_mass,
                      lj_coefs, bond_coefs, bond_styles, angle_coefs,
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
            i + 1, bond_coefs[i], bond_styles[i], write_style=True
        )
    text += '\n'

    # Write the angle potential coefficients 
    #
    # A dict of coefficients should be present for every angle type
    text += 'Angle Coeffs\n\n'
    for i in range(len(angle_coefs)):
        text += write_angle_coefs(i + 1, angle_coefs[i], write_style=False)
    text += '\n'

    # Write atomic coordinates, bonds, and angles  
    text += 'Atoms\n\n{}\n'.format(write_coords(polymers, crosslinkers))
    text += 'Bonds\n\n{}\n'.format(write_bonds(polymers, crosslinkers))
    text += 'Angles\n\n{}\n'.format(write_angles(polymers, crosslinkers))

    with open(outfilename, 'w') as f:
        f.write(header + text)

