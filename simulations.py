"""
Classes for parsing and analyzing LAMMPS simulations. 

Authors:
    Kee-Myoung Nam

Last updated:
    8/13/2024
"""
import numpy as np
from lammps_utils import dist_periodic

#########################################################################
class Simulation:
    """
    A basic wrapper class for a LAMMPS simulation.
    """
    def __init__(self, n_atoms, atom_types, atom_masses, molecule_ids, 
                 xmin, xmax, ymin, ymax, zmin, zmax):
        """
        Initialize a Simulation object from the given attributes.  
        """
        # Number of atoms in the simulation
        self.n_atoms = n_atoms

        # Array of atom type IDs for each atom 
        self.atom_types = np.array(atom_types, dtype=np.int64)

        # Number of atom types in the simulation 
        self.n_atom_types = len(atom_types)

        # Dict of atom masses corresponding to each atom type 
        self.atom_masses = atom_masses

        # Array of molecule IDs for each atom 
        self.molecule_ids = np.array(molecule_ids, dtype=np.int64)

        # Number of molecules in the simulation 
        self.n_molecules = len(molecule_ids)

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
    def __init__(self, init_filename):
        """
        Initialize a Simulation object from the given initial configuration
        file. 
        """
        # Parse the initial configuration file ... 
        with open(init_filename) as f:
            n_read = 0

            # Line 3 contains the number of atoms 
            while n_read < 2:
                f.readline()
                n_read += 1
            line = f.readline()
            self.n_atoms = int(line.split()[0])
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
            self.n_atom_types = int(line.split()[0])
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
            self.atom_masses = {}
            for i in range(self.n_atom_types):
                split = f.readline().split()
                self.atom_masses[int(split[0])] = float(split[1])
                n_read += 1

            # The next set of lines contain the Lennard-Jones potential
            # coefficients
            f.readline()
            f.readline()    # "PairIJ Coeffs"
            f.readline()
            n_read += 3
            lj_coefs = {}
            for i in range(self.n_atom_types):
                for j in range(i, self.n_atom_types):
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
            self.atom_types = []
            self.molecule_ids = []
            coords = np.zeros((self.n_atoms, 3))
            for i in range(self.n_atoms):
                split = f.readline().split()
                self.molecule_ids.append(int(split[1]) - 1)
                self.atom_types.append(int(split[2]))
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
            bonds = {
                (bi, bj): btype for bi, bj, btype in zip(bond_i, bond_j, bond_types)
            }
            bond_lengths = {
                (bi, bj): blen for bi, bj, blen in zip(bond_i, bond_j, bond_lengths)
            }

        # Clean up class attributes as required
        self.atom_types = np.array(self.atom_types, dtype=np.int64)
        self.molecule_ids = np.array(self.molecule_ids, dtype=np.int64)
        self.n_molecules = len(self.molecule_ids)
        self.xlim = (xmin, xmax)
        self.ylim = (ymin, ymax)
        self.zlim = (zmin, zmax)
        self.n_frames = 0
        self.times = []
        self.coords = []          # List of 3-column arrays
        self.bonds = []           # List of dicts indicating bond types
        self.bond_lengths = []    # List of dicts indicating bond lengths

        # Append initial frame of coordinates and bonds 
        self.append_frame(0.0, coords, bonds, bond_lengths)

    #####################################################################
    def append_frame(self, time, coords, bonds, bond_lengths):
        """
        Append a frame to the Simulation instance.
        """
        self.times.append(time)
        self.coords.append(coords)
        self.bonds.append(bonds)
        self.bond_lengths.append(bond_lengths)
        self.n_frames += 1

    #####################################################################
    def get_atom_coords(self, time, atom_type):
        """
        Get an array of atom coordinates for the given atom type at the
        given timepoint.

        A ValueError is raised if the given atom type does not exist, or 
        data for the given timepoint does not exist.
        """
        # Assume that the atom types are labeled 1, 2, 3, ... 
        if atom_type <= 0 or atom_type > self.n_atom_types:
            raise ValueError(
                'Specified atom type does not exist: {}'.format(atom_type)
            )
       
        # Verify that there is data for the given timepoint 
        try:
            t = self.times.index(time)
        except ValueError:
            raise

        # Identify the atoms with the given type 
        matches_type = np.where(self.atom_types == atom_type)[0]
        return self.coords[t][matches_type, :]

    #####################################################################
    def get_molecule_coords(self, time, molecule_ids):
        """
        Get an array of atom coordinates for the given molecules at the 
        given timepoint. 

        A ValueError is raised if the given molecule does not exist, or 
        data for the given timepoint does not exist. 
        """
        # Verify that there is data for the given timepoint 
        try:
            t = self.times.index(time)
        except ValueError:
            raise

        # Assume that the molecules are labeled 1, 2, 3, ... 
        for mol_id in molecule_ids:
            if mol_id <= 0 or mol_id > self.n_molecules:
                raise ValueError(
                    'Specified molecule does not exist: {}'.format(mol_id)
                )

        # Create a set of molecule IDs for fast lookup, as well as a
        # dictionary that maps each molecule ID to its index 
        molecule_idset = set(molecule_ids)
        molecule_idx = {mol_id: i for i, mol_id in enumerate(molecule_ids)}
        coords = [[] for _ in molecule_ids]

        # For each atom ... 
        for i, mol_id in enumerate(self.molecule_ids):
            # Is the corresponding molecule within the given list? 
            if mol_id in molecule_idset:
                # Then fill in the atom coordinates
                j = molecule_idx[mol_id]
                coords[j].append(self.coords[t][i, :])

        # Note that each molecule may have a different number of atoms, 
        # and so this should be returned as a list of arrays
        for j in range(len(molecule_ids)):
            coords[j] = np.array(coords[j])
        return coords

    #####################################################################
    @classmethod
    def from_file(cls, init_filename, lammpstrj_filename, bond_filename):
        """
        Parse a simulation, given the initial configuration file, the
        .lammpstrj file, and the .bond file. 
        """
        # First initialize the simulation
        sim = cls(init_filename)
        n_atoms = sim.n_atoms

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
                    curr_coords = np.zeros((n_atoms, 3), dtype=np.float64)
                    for i in range(9, len(curr_block)):
                        split = curr_block[i].split()
                        atom_id = int(split[0]) - 1
                        curr_coords[atom_id, :] = [float(x) for x in split[3:6]]
                    # Store the parsed data and reset for the new block 
                    times.append(timestep)
                    coords.append(curr_coords)
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
                    bond_types_all.append({
                        (bi, bj): btype
                        for bi, bj, btype in zip(bond_i, bond_j, bond_types)
                    })
                    bond_lengths_all.append({
                        (bi, bj): blen
                        for bi, bj, blen in zip(bond_i, bond_j, bond_lengths)
                    })
                    curr_block = []
                curr_block.append(line.strip())

        # Replace the initial configuration
        sim.coords[0] = coords[0]
        sim.bonds[0] = bond_types_all[0]
        sim.bond_lengths[0] = bond_lengths_all[0]

        # Append each subsequent configuration
        for i in range(1, len(times)):
            sim.append_frame(
                times[i], coords[i], bond_types_all[i], bond_lengths_all[i]
            )

        return sim

#########################################################################
class PolymerCrosslinkSimulation(Simulation):
    """
    A simulation of a system of polymers and crosslinkers.

    This class assumes that there are three atom types: a polymer atom type
    (1) and two crosslinker atom types (2 and 3).

    This class also assumes that there are three bond types: intra-polymer 
    (1), intra-crosslinker (2), and polymer-crosslinker (3). 
    """
    def __init__(self, n_atoms, atom_types, atom_masses, molecule_ids, 
                 xmin, xmax, ymin, ymax, zmin, zmax):
        """
        Initialize a PolymerCrosslinkSimulation from the given attributes. 
        """
        super().__init__(
            n_atoms, atom_types, atom_masses, molecule_ids, xmin, xmax, ymin,
            ymax, zmin, zmax
        )

        # Define lists of polymer and crosslinker IDs
        #
        # The polymer atom type is assumed to be 1, the crosslinker atom types
        # to be 2 and 3
        polymer_ids = set()
        for i in range(self.n_atoms):
            if self.atom_types[i] == 1:
                polymer_ids.add(self.molecule_ids[i])
        self.polymer_ids = sorted(polymer_ids)
        crosslinker_ids = set()
        for i in range(self.n_atoms):
            if self.atom_types[i] != 1:
                crosslinker_ids.add(self.molecule_ids[i])
        self.crosslinker_ids = sorted(crosslinker_ids)

    #####################################################################
    def __init__(self, init_filename):
        """
        Initialize a PolymerCrosslinkSimulation object from the given initial
        configuration file. 
        """
        super().__init__(init_filename)

        # Define lists of polymer and crosslinker IDs
        #
        # The polymer atom type is assumed to be 1, the crosslinker atom types
        # to be 2 and 3
        polymer_ids = set()
        for i in range(self.n_atoms):
            if self.atom_types[i] == 1:
                polymer_ids.add(self.molecule_ids[i])
        self.polymer_ids = sorted(polymer_ids)
        crosslinker_ids = set()
        for i in range(self.n_atoms):
            if self.atom_types[i] != 1:
                crosslinker_ids.add(self.molecule_ids[i])
        self.crosslinker_ids = sorted(crosslinker_ids)

    #####################################################################
    def get_polymer_ids(self):
        """
        Get a list of molecule IDs for the polymers in the system.

        The polymer atom type should be 1.

        Returns
        -------
        List of molecule IDs for the polymers in the system. 
        """
        return self.polymer_ids

    #####################################################################
    def get_crosslinker_ids(self):
        """
        Get a list of molecule IDs for the crosslinkers in the system.

        The crosslinker atom types should be 2 and 3. 

        Returns
        -------
        List of molecule IDs for the crosslinkers in the system. 
        """
        return self.crosslinker_ids

    #####################################################################
    def get_intermolecular_bonds(self):
        """
        Return a list of dicts that summarizes the bonding between polymers 
        and crosslinkers in each frame of the simulation.  

        The polymers and crosslinkers are indexed in order of their IDs. 

        Returns 
        -------
        List of dicts that summarizes the bonding between polymers and 
        crosslinkers in each frame of the simulation. 
        """
        polymer_idx = {pid: i for i, pid in enumerate(self.polymer_ids)}
        crosslinker_idx = {cid: i for i, cid in enumerate(self.crosslinker_ids)}

        # List of polymer-crosslinker connectivity adjacency lists, each
        # stored as a dict 
        graphs_pc = [{} for _ in range(self.n_frames)]

        # For each frame in the simulation ... 
        for i in range(self.n_frames):
            # Identify the polymer-crosslinker bonds within the system in 
            # the i-th frame 
            for bi, bj in self.bonds[i]:
                if self.bonds[i][(bi, bj)] == 3:
                    # Get the molecules and atom types of atoms bi and bj
                    mol_i, mol_j = self.molecule_ids[bi], self.molecule_ids[bj]
                    type_i, type_j = self.atom_types[bi], self.atom_types[bj]
                    if type_i == 1:   # If bi is the polymer atom ... 
                        p = polymer_idx[mol_i]
                        q = crosslinker_idx[mol_j]
                    else:             # If bj is the polymer atom ...
                        p = polymer_idx[mol_j]
                        q = crosslinker_idx[mol_i]
                    graphs_pc[i][(p, q)] = 1
        
        # Return adjacency lists 
        return graphs_pc

