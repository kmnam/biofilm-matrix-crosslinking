"""
Authors:
    Kee-Myoung Nam

Last updated:
    4/29/2024
"""
import numpy as np
from scipy.sparse import csr_matrix

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
class Simulation:
    """
    A basic wrapper for a LAMMPS simulation.
    """
    def __init__(self, n_atoms, atom_types, atom_masses, molecule_ids, 
                 xmin, xmax, ymin, ymax, zmin, zmax):
        """
        Initialize a Simulation object. 
        """
        # Number of atoms in the simulation
        self.n_atoms = n_atoms

        # Array of atom type IDs for each atom 
        self.atom_types = atom_types

        # Dict of atom masses corresponding to each atom type 
        self.atom_masses = atom_masses

        # Array of molecule IDs for each atom 
        self.molecule_ids = molecule_ids

        # Simulation domain bounds 
        self.xlim = (xmin, xmax)
        self.ylim = (ymin, ymax)
        self.zlim = (zmin, zmax)

        # Lists of atomic configurations, bonding configurations, and their
        # corresponding timepoints
        self.n_frames = 0
        self.times = []
        self.coords = []          # List of 3-column arrays
        self.bonds = []           # List of sparse matrices indicating bond types 
        self.bond_lengths = []    # List of sparse matrices indicating bond lengths

    #####################################################################
    def append_frame(self, time, coords, bonds, bond_lengths):
        """
        Append a frame to the Simulation object.
        """
        self.times.append(time)
        self.coords.append(coords)
        self.bonds.append(bonds)
        self.bond_lengths.append(bond_lengths)

#########################################################################
def parse_init(init_filename):
    """
    Parse the initial configuration file for a simulation. 
    """
    # Parse the initial configuration file ... 
    with open(init_filename) as f:
        n_read = 0

        # Line 3 contains the number of atoms 
        while n_read < 2:
            f.readline()
            n_read += 1
        line = f.readline()
        n_atoms = int(line.split()[0])
        n_read += 1

        # Line 4 contains the number of bonds
        line = f.readline()
        n_bonds = int(line.split()[0])
        n_read += 1

        # Line 9 contains the number of atom types 
        while n_read < 8:
            f.readline()
            n_read += 1
        line = f.readline()
        n_atom_types = int(line.split()[0])
        n_read += 1

        # Line 10 contains the number of bond types
        line = f.readline()
        n_bond_types = int(line.split()[0])
        n_read += 1

        # Lines 15, 16, 17 contain the simulation domain bounds 
        while n_read < 14:
            f.readline()
            n_read += 1
        split = f.readline().split()
        xmin, xmax = float(split[0]), float(split[1])
        split = f.readline().split()
        ymin, ymax = float(split[0]), float(split[1])
        split = f.readline().split()
        zmin, zmax = float(split[0]), float(split[1])
        n_read += 3

        # The next set of lines contain the atom masses 
        f.readline()
        f.readline()    # "Masses"
        f.readline()
        n_read += 3
        atom_masses = {}
        for i in range(n_atom_types):
            split = f.readline().split()
            atom_masses[int(split[0])] = float(split[1])
            n_read += 1

        # The next set of lines contain the Lennard-Jones potential
        # coefficients
        f.readline()
        f.readline()    # "PairIJ Coeffs"
        f.readline()
        n_read += 3
        lj_coefs = {}
        for i in range(n_atom_types):
            for j in range(i, n_atom_types):
                split = f.readline().split()
                eps = float(split[2])
                sigma = float(split[3])
                cutoff = float(split[4])
                lj_coefs[(i + 1, j + 1)] = [eps, sigma, cutoff]
                n_read += 1

        # The next set of lines contain the bond coefficients
        f.readline()
        f.readline()    # "Bond Coeffs"
        f.readline()
        n_read += 3
        bond_coefs = {}
        for i in range(n_bond_types):
            split = f.readline().split()
            bond_coefs[i + 1] = [float(x) for x in split[2:]]
            n_read += 1

        # The next set of lines contain the atom types, molecule IDs, and
        # corresponding coordinates
        f.readline()
        f.readline()    # "Atoms"
        f.readline()
        n_read += 3
        atom_types = []
        molecule_ids = []
        coords = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            split = f.readline().split()
            molecule_ids.append(int(split[1]))
            atom_types.append(int(split[2]))
            coords[i, 0] = float(split[3])
            coords[i, 1] = float(split[4])
            coords[i, 2] = float(split[5])
            n_read += 1

        # The next set of lines contain the bonds 
        f.readline()
        f.readline()    # "Bonds"
        f.readline()
        n_read += 3
        bond_i = []
        bond_j = []
        bond_types = []
        bond_lengths = []
        for i in range(n_bonds):
            split = f.readline().split()
            bond_i.append(int(split[2]) - 1)
            bond_j.append(int(split[3]) - 1)
            bond_types.append(int(split[1]))
            coords_i = coords[bond_i, :].reshape(-1)
            coords_j = coords[bond_j, :].reshape(-1)
            dist_ij = dist_periodic(
                coords_i, coords_j, xmin, xmax, ymin, ymax, zmin, zmax
            )
            bond_lengths.append(dist_ij)
        bonds = csr_matrix(
            (bond_types, (bond_i, bond_j)), shape=(n_atoms, n_atoms)
        )
        bond_lengths = csr_matrix(
            (bond_lengths, (bond_i, bond_j)), shape=(n_atoms, n_atoms)
        )

    # Instantiate a new Simulation object
    sim = Simulation(
        n_atoms, atom_types, atom_masses, molecule_ids, xmin, xmax, ymin,
        ymax, zmin, zmax
    )
    sim.append_frame(0.0, coords, bonds, bond_lengths)
    
    return sim

#########################################################################
def parse_simulation(init_filename, lammpstrj_filename, bond_filename):
    """
    Parse a simulation, given the initial configuration file, the .lammpstrj
    file, and the .bond file.
    """
    # First initialize the simulation
    sim = parse_init(init_filename)

    # Each block in the .lammpstrj file has the format: 
    #
    # ITEM: TIMESTEP
    # [time]
    # ITEM: NUMBER OF ATOMS
    # [number of atoms]
    # ITEM: BOX BOUNDS pp pp pp
    # [xmin] [xmax]
    # [ymin] [ymax]
    # [zmin] [zmax]
    # ITEM: ATOMS id mol type xu yu zu
    # 1 [molecule id] [atom type] [x-pos] [y-pos] [z-pos] [x-vel] [y-vel] [z-vel]
    # 2 [molecule id] [atom type] [x-pos] [y-pos] [z-pos] [x-vel] [y-vel] [z-vel]
    # ...
    times = []
    coords = []
    with open(lammpstrj_filename) as f:
        curr_block = []
        curr_block.append(f.readline().strip())
        for line in f:
            # If a new block is encountered, process the old block
            if line == 'ITEM: TIMESTEP\n':
                # The timestep is in the second line 
                timestep = int(curr_block[1])
                # Skip over the number of atoms and the simulation domain
                # boundaries, then parse the atom coords from the remaining
                # lines 
                curr_coords = []
                for i in range(9, len(curr_block)):
                    split = curr_block[i].split()
                    curr_coords.append([
                        float(split[3]), float(split[4]), float(split[5])
                    ])
                times.append(timestep)
                coords.append(np.array(curr_coords))
                curr_block = []
            curr_block.append(line.strip())

    # Each block in the .bond file has the format: 
    #
    # ITEM: TIMESTEP
    # [time]
    # ITEM: NUMBER OF ENTRIES
    # [number of bonds]
    # ITEM: BOX BOUNDS pp pp pp
    # [xmin] [xmax]
    # [ymin] [ymax]
    # [zmin] [zmax]
    # ITEM: ENTRIES index c_1[1] c_1[2] c_1[3] c_2[1] c_2[2] c_2[3]
    # [bond id] [bond type] [atom i] [atom j] [bond length] [bond energy] [bond force]
    # [bond id] [bond type] [atom i] [atom j] [bond length] [bond energy] [bond force]
    # ...
    bond_types_all = []
    bond_lengths_all = []
    with open(bond_filename) as f:
        curr_block = []
        curr_block.append(f.readline().strip())
        for line in f:
            # If a new block is encountered, process the old block
            if line == 'ITEM: TIMESTEP\n':
                # The timestep is in the second line 
                timestep = int(curr_block[1])
                # Skip over the timestep, number of bonds and the simulation
                # domain boundaries, then parse the bonds from the remaining
                # lines
                bond_i = []
                bond_j = []
                bond_types = []
                bond_lengths = []
                for i in range(9, len(curr_block)):
                    split = curr_block[i].split()
                    bond_i.append(int(split[2]) - 1)
                    bond_j.append(int(split[3]) - 1)
                    bond_types.append(int(split[1]))
                    bond_lengths.append(float(split[4]))
                bond_types_all.append(
                    csr_matrix(
                        (bond_types, (bond_i, bond_j)),
                        shape=(sim.n_atoms, sim.n_atoms)
                    )
                )
                bond_lengths_all.append(
                    csr_matrix(
                        (bond_lengths, (bond_i, bond_j)),
                        shape=(sim.n_atoms, sim.n_atoms)
                    )
                )
                curr_block = []
            curr_block.append(line.strip())

    # Replace the initial configuration
    sim.coords[0] = coords[0]
    sim.bonds[0] = bond_types_all[0]
    sim.bond_lengths[0] = bond_lengths_all[0]

    # Append each subsequent configuration
    for i in range(1, len(times)):
        sim.append_frame(times[i], coords[i], bond_types_all[i], bond_lengths_all[i])

    return sim

