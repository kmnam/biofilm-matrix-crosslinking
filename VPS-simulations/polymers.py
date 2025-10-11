"""
Utility functions and classes for defining initial conformations of polymers
of fixed bond length and tetrahedral crosslinkers.

Authors:
    Kee-Myoung Nam

Last updated:
    10/10/2025
"""
import numpy as np

_polymer_atom_type = 1
_crosslinker_atom_type = 2
_crosslinker_sticky_atom_type = 3
_polymer_bond_type = 1
_crosslinker_bond_type = 2
_polymer_crosslinker_bond_type = 3
_polymer_angle_type = 1
_crosslinker_angle_type = 2
_polymer_crosslinked_angle_type = 3
_polymer_dihedral_type = 1

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
    def distance_matrix(self):
        """
        Return the pairwise distance matrix among the atoms in the Molecule
        object.

        Returns
        -------
        Pairwise distance matrix among the atoms in the Molecule object. 
        """
        dist = np.zeros((self.coords.shape[0], self.coords.shape[0]))

        # Only fill in the upper triangle of the matrix 
        for i in range(self.coords.shape[0]):
            for j in range(i + 1, self.coords.shape[0]):
                dist[i, j] = np.linalg.norm(self.coords[i, :] - self.coords[j, :])

        return dist

    #####################################################################
    def min_deviation_from_molecule(self, mol):
        """
        Return the minimum deviation between each atom in the Molecule object
        and each atom in the given Molecule object.

        Parameters
        ----------
        mol : `Molecule`
            Input Molecule object.

        Returns
        -------
        Distance between self and given `Molecule`. 
        """
        return np.min([
            self.min_deviation(mol.coords[j, :]) for j in range(mol.coords.shape[0])
        ])

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

