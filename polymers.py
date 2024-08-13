"""
Utility functions and classes for defining initial conformations of polymers
of fixed bond length and tetrahedral crosslinkers. 

Authors:
    Kee-Myoung Nam

Last updated:
    8/13/2024
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
class Molecule:
    """
    A basic Molecule class that incorporates all attributes and functions
    common to both polymers and crosslinkers.
    """
    def __init__(self, coords=None):
        """
        Initialize a Molecule object with the given monomer coordinates.

        Parameters
        ----------
        coords : numpy.ndarray
            Array of monomer coordinates.
        """
        if coords is None:
            self.coords = np.zeros((0, 3), dtype=np.float64)
        else:
            self.coords = coords

    #####################################################################
    def center_of_mass(self):
        """
        Return the center of mass of the Polymer object.

        Returns
        -------
        Center of mass of the monomers. 
        """
        return np.mean(self.coords, axis=0)

    #####################################################################
    def min_deviation(self, p):
        """
        Return the minimum deviation from the given point of the Polymer 
        object, i.e., the distance to the closest monomer.

        Parameters
        ----------
        p : numpy.ndarray
            Input point. 

        Returns
        -------
        Distance between input point and the closest monomer. 
        """
        return np.linalg.norm(self.coords - p, axis=1).min()

    #####################################################################
    def max_deviation(self, p):
        """
        Return the maximum deviation from the given point of the Polymer 
        object, i.e., the distance to the furthest monomer. 

        Parameters
        ----------
        p : numpy.ndarray
            Input point. 

        Returns
        -------
        Distance between input point and the furthest monomer. 
        """
        return np.linalg.norm(self.coords - p, axis=1).max()

#########################################################################
class Polymer(Molecule):
    """
    A basic Polymer class. 
    """
    def __init__(self, coords=None):
        """
        Initialize a Polymer object with the given monomer coordinates.

        Parameters
        ----------
        coords : numpy.ndarray
            Array of monomer coordinates.
        """
        super(Polymer, self).__init__(coords)
        if coords is None:
            self.length = 0
        else:
            self.length = coords.shape[0]

    #####################################################################
    def __len__(self):
        """
        Return the length of the Polymer object. 

        Returns
        -------
        Length of the Polymer object. 
        """
        return self.length

    #####################################################################
    def append(self, p):
        """
        Add a monomer to the end of the Polymer object.

        Parameters
        ----------
        p : numpy.ndarray
            Array of coordinates for the new monomer. 
        """
        self.length += 1
        self.coords = np.vstack((self.coords, p.reshape(1, -1)))

    #####################################################################
    def radius(self):
        """
        Return the radius of the Polymer object, defined as the maximum 
        deviation from the center of mass. 

        Note that this is not the radius of gyration of the polymer. 

        Returns
        -------
        Radius of the polymer. 
        """
        return self.max_deviation(self.center_of_mass())

    #####################################################################
    def contact_matrix(self, polymer):
        """
        Return the distance matrix between two Polymer objects.

        Parameters
        ----------
        polymer : Polymer
            Input polymer. 

        Returns
        -------
        Distance matrix between monomers in the two polymers. 
        """
        dist = np.zeros((self.length, len(polymer)), dtype=np.float64)

        # The (i, j)-th entry is the distance between monomer i in self 
        # and monomer j in the other polymer
        for i in range(self.length):
            for j in range(len(polymer)):
                dist[i, j] = np.linalg.norm(
                    self.coords[i, :] - polymer.coords[j, :]
                )

        return dist

#########################################################################
class AtomicCrosslinker(Molecule):
    """
    A basic atomic crosslinker class. 
    """
    def __init__(self, p):
        """
        Define the crosslinker at the given point.

        Parameters
        ----------
        p : numpy.ndarray
            Input point. 
        """
        super(AtomicCrosslinker, self).__init__(np.array(p).reshape(1, -1))

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
    def __init__(self, center, radius):
        """
        Define the coordinates of a crosslinker at the given center with
        the given radius.

        Parameters
        ----------
        center : numpy.ndarray
            Input center.
        radius : float
            Input radius.
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
def generate_polymers(n, polymer_length, bond_length, angle_dist, rng, xmin,
                      xmax, ymin, ymax, zmin, zmax, eps1, eps2):
    """
    Generate a set of `n` random coils as Polymer objects in the given
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

    Returns
    -------
    A list of generated Polymer objects.
    """
    # Get the dimensions of the box 
    vmin = np.array([xmin, ymin, zmin])
    vmax = np.array([xmax, ymax, zmax])
    dims = vmax - vmin

    # Define a function that tests whether a point lies within the box 
    within_box = lambda p: ((p >= vmin).all() and (p <= vmax).all())

    # Maintain a list of Polymer objects 
    polymers = []

    # Generate the i-th polymer ... 
    for i in range(n):
        polymer_i = Polymer()

        # Sample a random starting point that is not too close to any
        # previously generated polymer
        p = rng.random((3,)) * dims + vmin
        while any(polymer.min_deviation(p) < eps1 for polymer in polymers):
            p = rng.random((3,)) * dims + vmin
        polymer_i.append(p)

        # Sample a second monomer position by the given bond length in 
        # any direction 
        p = polymer_i.coords[0, :] + bond_length * random_dir(rng)
        near_inter = any(polymer.min_deviation(p) < eps1 for polymer in polymers)
        while (not within_box(p)) or near_inter:
            p = polymer_i.coords[0, :] + bond_length * random_dir(rng)
            near_inter = any(polymer.min_deviation(p) < eps1 for polymer in polymers)
        polymer_i.append(p)

        # Sample all subsequent monomer positions, resampling until: 
        # (1) the point lies within the box 
        # (2) the point is sufficiently distant from all previously sampled 
        #     polymers, and 
        # (3) the point is sufficiently distant from all previously sampled 
        #     monomers within the given polymer 
        for j in range(2, polymer_length):
            q = generate_next_atom(
                polymer_i.coords[j-2, :], polymer_i.coords[j-1, :], bond_length,
                angle_dist(rng), rng
            )
            near_inter = any(polymer.min_deviation(q) < eps1 for polymer in polymers)
            near_intra = (polymer_i.min_deviation(q) < eps2)
            while (not within_box(q)) or near_inter or near_intra:
                q = generate_next_atom(
                    polymer_i.coords[j-2, :], polymer_i.coords[j-1, :], bond_length,
                    angle_dist(rng), rng
                )
                near_inter = any(polymer.min_deviation(q) < eps1 for polymer in polymers)
                near_intra = (polymer_i.min_deviation(q) < eps2)
            polymer_i.append(q)

        # Keep track of generated polymer 
        polymers.append(polymer_i)

    return polymers

#########################################################################
def generate_atomic_crosslinkers(polymers, n, rng, xmin, xmax, ymin, ymax,
                                 zmin, zmax, eps1, eps2):
    """
    Generate a set of `n` atomic crosslinkers to superimpose on the given
    set of polymers in the given 3-D box.

    Parameters
    ----------
    polymers : list
        Pre-generated list of Polymer objects.
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

    Returns
    -------
    A list of AtomicCrosslinker objects.
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
        near_polymers = any(poly.min_deviation(p) < eps1 for poly in polymers)
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
        crosslinker = AtomicCrosslinker(p)
        crosslinkers.append(crosslinker)

    return crosslinkers

#########################################################################
def generate_tetrahedral_crosslinkers(polymers, n, radius, rng, xmin, xmax,
                                      ymin, ymax, zmin, zmax, eps1, eps2):
    """
    Generate a set of `n` tetrahedral crosslinkers to superimpose on the
    given set of polymers in the given 3-D box.

    Parameters
    ----------
    polymers : list
        Pre-generated list of Polymer objects.
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
    A list of TetrahedralCrosslinker objects.
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
        near_polymers = any(poly.min_deviation(p) < threshold1 for poly in polymers)
        near_crosslinkers = any(
            cross.min_deviation(p) < threshold2 for cross in crosslinkers
        )
        while near_polymers or near_crosslinkers:
            p = rng.random((3,)) * dims + vmin
            near_polymers = any(poly.min_deviation(p) < threshold1 for poly in polymers)
            near_crosslinkers = any(
                cross.min_deviation(p) < threshold2 for cross in crosslinkers
            )

        # Generate a tetrahedral crosslinker at the origin, then rotate,
        # then translate
        crosslinker = TetrahedralCrosslinker([0, 0, 0], radius)
        crosslinker.rotate(
            rng.random() * 2 * np.pi,
            rng.random() * 2 * np.pi,
            rng.random() * 2 * np.pi
        )
        crosslinker.translate(p[0], p[1], p[2])
        crosslinkers.append(crosslinker)

    return crosslinkers

#########################################################################
def plot_polymers(polymers, ax, dims=(0, 1), with_bounding_spheres=False):
    """
    Plot the given polymers on the given axes along the given dimensions. 

    The dimensions `dims` can be set to a tuple of two ints, each of which 
    specifies whether the x (0), y (1), or z (2) coordinates are plotted.

    Parameters
    ----------
    polymers : list of `Polymer` objects
        List of Polymer objects to be plotted.
    ax : `matplotlib.pyplot.Axes`
        Axes object.  
    dims : tuple of two ints
        Indicates the dimensions that should be plotted (x, y, or z).
    with_bounding_spheres : bool
        If True, plot the circle that bounds each polymer.

    Returns
    -------
    Modified Axes object. 
    """
    if not (type(dims) == tuple and len(dims) == 2):
        raise ValueError('dims should be a tuple of length 2')
    if not (dims[0] in [0, 1, 2] and dims[1] in [0, 1, 2]):
        raise ValueError('Invalid value given for dims: {}'.format(dims))

    for polymer in polymers:
        # Plot each polymer ... 
        ax.plot(polymer.coords[:, dims[0]], polymer.coords[:, dims[1]], marker='.')

        # ... along with the bounding sphere if desired
        if with_bounding_spheres:
            center = polymer.center_of_mass()
            radius = polymer.radius()
            theta = np.linspace(0, 2 * np.pi, 50)
            ax.plot(
                center[dims[0]] + radius * np.cos(theta),
                center[dims[1]] + radius * np.sin(theta),
                c='black'
            )

    return ax

#########################################################################
def plot_crosslinkers(crosslinkers, ax, dims=(0, 1)):
    """
    Plot the given crosslinkers on the given axes along the given dimensions. 

    The dimensions `dims` can be set to a tuple of two ints, each of which 
    specifies whether the x (0), y (1), or z (2) coordinates are plotted.

    Parameters
    ----------
    crosslinkers : list of `AtomicCrosslinker` or `TetrahedralCrosslinker` objects
        List of AtomicCrosslinker or TetrahedralCrosslinker objects to be plotted.
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

