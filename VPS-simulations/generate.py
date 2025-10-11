"""
Functions for generating new polymer/crosslinker configurations. 

Authors:
    Kee-Myoung Nam

Last updated:
    10/10/2025
"""
import numpy as np
from scipy.stats import circmean
from polymers import Polymer, AtomicCrosslinker, TetrahedralCrosslinker

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
def get_dihedral(r1, r2, r3, r4):
    """
    Compute the dihedral angle along the given four-atom segment. 
    """
    u1 = r2 - r1
    u2 = r3 - r2
    u3 = r4 - r3
    return np.arctan2(
        np.linalg.norm(u2) * np.dot(u1, np.cross(u2, u3)), 
        np.dot(np.cross(u1, u2), np.cross(u2, u3))
    )

#########################################################################
def generate_next_atom(atom1_coords, atom2_coords, length, angle, rng):
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
         [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]
    ])
    trans = np.hstack((-u.reshape((-1, 1)), v.reshape((-1, 1)), w.reshape((-1, 1))))
    u_new = trans @ (np.append(rot @ np.array([1, 0]), 0))

    # Get the position of the next atom
    return atom2_coords + length * u_new

#########################################################################
def generate_next_atom_dihedral(atom1_coords, atom2_coords, atom3_coords, length,
                                angle, dihedral, rng, force_plus=False):
    """
    Given the positions of three bonded atoms within a polymer (atoms 1-3),
    the length of the next bond (3 to 4), the angle formed by the next bond
    (between 2-3 and 3-4), and the dihedral angle along the segment, generate
    one of the two candidate positions for atom 4. 
    """
    # Get the distance vector and direction from atom 2 to atom 3
    u1 = atom2_coords - atom1_coords
    u2 = atom3_coords - atom2_coords
    x = u2 / np.linalg.norm(u2)

    # Get the cross product between u1 and u2, which is normal to the plane
    # containing atoms 1, 2, 3
    z = np.cross(u1, u2)
    z /= np.linalg.norm(z)

    # Get the cross product between z and x; this yields a right-handed
    # orthonormal basis (x, y, z)
    y = np.cross(z, x)

    # Get the position of atom 4
    sign = (1 if force_plus or rng.random() < 0.5 else -1)
    w = -np.cos(angle) * x + np.sin(angle) * (np.cos(dihedral) * y + sign * np.sin(dihedral) * z)
    
    return atom3_coords + length * w

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
#          FUNCTIONS FOR GENERATING POLYMERS AND CROSSLINKERS           #
#########################################################################
def generate_polymers(n, polymer_length, bond_length, angle_types, angle_coefs,
                      rng, xmin, xmax, ymin, ymax, zmin, zmax, eps1, eps2,
                      dihedral_dists=None, atom_type=_polymer_atom_type,
                      max_seed_per_polymer=1000, max_backtracks_per_polymer=100,
                      max_tries_per_bond=100):
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
    angle_types : 2-D `numpy.ndarray`
        Angle types (1, 2, ...) for each internal monomer along each polymer. 
    angle_dists : list of callables 
        A callable for each angle type, which takes the given random number
        generator as input and samples a bond angle at the corresponding 
        monomer. 
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
    dihedral_dists : list of callables 
        A callable for each dihedral type, which takes the given random number
        generator as input and samples a dihedral angle along the corresponding
        four-atom segment. 
    atom_type : int
        Atom type for each atom in each `Polymer` object. 
    max_seed_per_polymer : int
        Maximum number of starting monomer positions to be sampled for any
        `Polymer` object. If this number is exceeded, a `RuntimeError` is
        raised.
    max_backtracks_per_polymer : int
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

    # ------------------------------------------------------------------- #
    # Define a function for determining whether a monomer position is acceptable
    def is_acceptable(polymer_i, p):
        # Is p too close to any other polymer? 
        near_inter = (
            len(polymers) > 0 and
            any(polymer.min_deviation(p) < eps1 for polymer in polymers)
        )

        # Is p too close to any other monomer within the polymer? 
        near_intra = (len(polymer_i) > 0 and polymer_i.min_deviation(p) < eps2)

        # Is p within the box? 
        return ((not near_inter) and (not near_intra) and within_box(p))

    # ------------------------------------------------------------------- #
    # Define a function for seeding a polymer (adding the zeroth monomer)
    def add_zeroth_monomer(polymer_i, n_seed=0):
        # Sample a monomer position until an acceptable one is chosen
        p = rng.random((3,)) * dims + vmin
        n_seed += 1
        accept = is_acceptable(polymer_i, p)
        while not accept:
            p = rng.random((3,)) * dims + vmin
            n_seed += 1
            accept = is_acceptable(polymer_i, p)
            # If the maximum number of attempts at seeding the polymer
            # has been exceeded, raise an Exception
            if n_seed > max_seed_per_polymer:
                raise RuntimeError('Exceeded max number of seeds for polymer')

        # Append the monomer 
        polymer_i.append(p, atom_type)
        return polymer_i, n_seed

    # ------------------------------------------------------------------- #
    # Define a function for adding the first monomer 
    def add_first_monomer(polymer_i):
        # Sample a monomer position until an acceptable one is chosen
        p = polymer_i.coords[0, :] + bond_length * random_dir(rng)
        n_tries = 1
        accept = is_acceptable(polymer_i, p)
        while not accept:
            p = polymer_i.coords[0, :] + bond_length * random_dir(rng)
            n_tries += 1
            accept = is_acceptable(polymer_i, p)
            # Quit if we have reached the maximum number of tries
            if n_tries > max_tries_per_bond:
                break

        if n_tries > max_tries_per_bond:
            return None, False
        else:
            polymer_i.append(p, atom_type)    # Append the monomer
            return polymer_i, True

    # ------------------------------------------------------------------- #
    # Define a function for adding the subsequent monomers
    def add_monomer(polymer_i, i, j):
        # Sample a monomer position until an acceptable one is chosen
        #
        # If j == 2, then we constrain the new monomer according to
        # the 0-1-2 bond angle
        #
        # If j > 2, then we constrain the new monomer according to 
        # the (j-2)-(j-1)-j bond angle *and* (if desired) the (j-3)-
        # (j-2)-(j-1)-j dihedral angle
        angle_type = angle_types[i, j - 2]
        angle = rng.vonmises(
            angle_coefs['type{}'.format(angle_type)]['theta0'],
            angle_coefs['type{}'.format(angle_type)]['K']
        )
        if j == 2 or dihedral_dists is None:
            p = generate_next_atom(
                polymer_i.coords[j-2, :], polymer_i.coords[j-1, :],
                bond_length, angle, rng
            )
        else:
            # The dihedral along the segment (j-3)-(j-2)-(j-1)-j is 
            # determined by the angle types at j - 2 and j - 1
            #
            # Note that these correspond to the entries angle_types[i, j - 3]
            # and angle_types[i, j - 2], respectively
            if angle_types[i, j - 3] == 2 or angle_types[i, j - 2] == 2:
                dihedral_type = 2
            else:
                dihedral_type = 1
            dihedral = dihedral_dists[dihedral_type - 1](rng)
            p = generate_next_atom_dihedral(
                polymer_i.coords[j-3, :], polymer_i.coords[j-2, :],
                polymer_i.coords[j-1, :], bond_length, angle,
                dihedral, rng
            )
        n_tries = 1
        accept = is_acceptable(polymer_i, p)
        while not accept:
            # Keep trying until an acceptable monomer is sampled 
            angle_type = angle_types[i, j - 2]
            angle = rng.vonmises(
                angle_coefs['type{}'.format(angle_type)]['theta0'],
                angle_coefs['type{}'.format(angle_type)]['K']
            )
            if j == 2 or dihedral_dists is None:
                p = generate_next_atom(
                    polymer_i.coords[j-2, :], polymer_i.coords[j-1, :],
                    bond_length, angle, rng
                )
            else:
                # The dihedral along the segment (j-3)-(j-2)-(j-1)-j is 
                # determined by the angle types at j - 2 and j - 1
                #
                # Note that these correspond to the entries angle_types[i, j - 3]
                # and angle_types[i, j - 2], respectively
                if angle_types[i, j - 3] == 2 or angle_types[i, j - 2] == 2:
                    dihedral_type = 2
                else:
                    dihedral_type = 1
                dihedral = dihedral_dists[dihedral_type - 1](rng)
                p = generate_next_atom_dihedral(
                    polymer_i.coords[j-3, :], polymer_i.coords[j-2, :],
                    polymer_i.coords[j-1, :], bond_length, angle, 
                    dihedral, rng
                )
            n_tries += 1
            accept = is_acceptable(polymer_i, p)
            # Quit if we have reached the maximum number of tries
            if n_tries > max_tries_per_bond:
                break

        if n_tries > max_tries_per_bond:
            return None, False
        else:
            polymer_i.append(p, atom_type)    # Append the monomer
            return polymer_i, True

    # ------------------------------------------------------------------- #
    # Generate the i-th polymer ...
    for i in range(n):
        print('... generating polymer {}'.format(i))
        polymer_i = Polymer(coords=None, atom_types=None)
        j = 0
        n_seed = 0
        n_backtrack = 0
        while j < polymer_length:
            print('... adding monomer {}, {} backtracks'.format(j, n_backtrack))
            # If j == 0, sample a random starting point that is not too close
            # to any previously generated polymer
            if j == 0:
                polymer_i, n_seed = add_zeroth_monomer(polymer_i, n_seed)
                j += 1

            # If j == 1, sample a second monomer position by the given bond
            # length in any direction
            elif j == 1:
                polymer_new, status = add_first_monomer(polymer_i)
                # If the maximum number of attempts at sampling this monomer
                # position has been exceeded, backtrack by one monomer 
                if not status: 
                    polymer_i.pop()
                    j -= 1
                    n_backtrack += 1
                # Otherwise, append the new monomer
                else:
                    polymer_i = polymer_new
                    j += 1

            # Otherwise, sample the j-th monomer position
            else:
                polymer_new, status = add_monomer(polymer_i, i, j)
                # If the maximum number of attempts at sampling this monomer
                # position has been exceeded, backtrack by one monomer 
                if not status: 
                    if j == 2:
                        polymer_i.pop()
                        j -= 1
                    elif j == 3:
                        polymer_i.pop()
                        polymer_i.pop()
                        j -= 2
                    else:
                        polymer_i.pop()
                        polymer_i.pop()
                        polymer_i.pop()
                        j -= 3
                    n_backtrack += 1
                # Otherwise, append the new monomer
                else:
                    polymer_i = polymer_new
                    j += 1
            
            # If the maximum number of backtracks per polymer has been 
            # exceeded, then start fresh 
            if n_backtrack > max_backtracks_per_polymer:
                polymer_i.clear()
                j = 0
                n_backtrack = 0

        # Keep track of generated polymer 
        polymers.append(polymer_i)

    assert (polymer.distance_matrix().min() >= eps2 for polymer in polymers)
    for i in range(len(polymers)):
        for j in range(i + 1, len(polymers)):
            assert polymers[i].min_deviation_from_molecule(polymers[j]) >= eps1
    for polymer in polymers:
        for i in range(1, len(polymer)):
            p = polymer.coords[i, :]
            q = polymer.coords[i - 1, :]
            assert np.abs(np.linalg.norm(p - q) - bond_length) < 1e-8 
        angles = []
        for i in range(1, len(polymer) - 1):
            p = polymer.coords[i - 1, :] - polymer.coords[i, :]
            q = polymer.coords[i + 1, :] - polymer.coords[i, :]
            theta = np.arccos(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))
            angles.append(theta)
        print(np.min(angles), np.max(angles), circmean(angles, high=np.pi, low=0))

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

    for polymer in polymers:
        for crosslinker in crosslinkers:
            assert polymer.min_deviation_from_molecule(crosslinker) >= eps1
    for i in range(len(crosslinkers)):
        for j in range(i + 1, len(crosslinkers)):
            assert crosslinkers[i].min_deviation_from_molecule(crosslinkers[j]) >= eps2

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

    for polymer in polymers:
        for crosslinker in crosslinkers:
            assert polymer.min_deviation_from_molecule(crosslinker) >= eps1
    for i in range(len(crosslinkers)):
        for j in range(i + 1, len(crosslinkers)):
            assert crosslinkers[i].min_deviation_from_molecule(crosslinkers[j]) >= eps2

    return crosslinkers

