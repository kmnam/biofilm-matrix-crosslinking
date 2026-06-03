"""
Authors:
    Kee-Myoung Nam

Last updated:
    5/25/2026
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
import matplotlib.pyplot as plt
from mdtraj import load_prmtop, load_netcdf

########################################################################
def write_pdb(topology_filename, traj_filename, outfilename):
    """
    Write a PDB file for the trajectory specified in the given Amber topology
    and trajectory files.

    Parameters
    ----------
    topology_filename : str
        Amber topology filename. 
    traj_filename : str
        Trajectory filename. 
    outfilename : str
        Output PDB filename. 
    """
    topology = load_prmtop(topology_filename)
    traj = load_netcdf(traj_filename, top=topology)
    traj.save_pdb(outfilename)

########################################################################
def find_monomers(topology):
    """
    Partition the atoms in the topology file among the 10 VPS monomers.

    This function assumes there are 10 monomers in the segment.

    Parameters
    ----------
    topology : `mdtraj.Topology`
        Topology object. 

    Returns
    -------
    Atoms corresponding to each monomer.
    """
    # The atoms are grouped as follows: 
    #
    # ROH0-H01, ROH0-O1: terminal hydroxyl group 
    # 4LA[k]: galactose subunit
    # 4GA[k]: first glucose subunit
    # 4GB[k]: second glucose subunit 
    # UNL[k]/UNT[k]: gulose subunit 
    # ACX[k]: O-acetyl substitution on C3 hydroxyl of gulose  
    # GLY[k]: glycine modification
    #
    # There are therefore six subunits per monomer (except for the first), 
    # indexed 6*i + 1, ..., 6*i + 6
    monomers = [[] for _ in range(10)]
    subunit_types = set()
    for atom in topology.atoms:
        name = atom.residue.name + str(atom.residue.index)
        subunit_types.add(name)
        if name.startswith('ROH0'):
            monomers[0].append(atom)
        else:
            idx = int(name[3:])
            monomers[(idx - 1) // 6].append(atom)

    for monomer in monomers:
        subunits = []
        for atom in monomer:
            name = (atom.residue.name, atom.residue.index)
            subunits.append(name)

    return monomers

########################################################################
def find_glycosidic_linkages(topology):
    """
    Find the atoms in the glycosidic linkages along the 10-mer VPS chain. 

    Each linkage is characterized by two dihedral angles:

    phi: O5-C1-O4'-C4'
    psi: C1-O4'-C4'-C3'

    Therefore, we store the O5, C1, O4', C4', and C3' along the linkage.  

    This function assumes there are 10 monomers in the segment.

    Parameters
    ----------
    topology : `mdtraj.Topology`
        Topology object. 

    Returns
    -------
    Atoms corresponding to each glycosidic linkage. 
    """
    # The atoms are grouped as follows: 
    #
    # ROH0-H01, ROH0-O1: terminal hydroxyl group 
    # 4LA[k]: galactose subunit
    # 4GA[k]: first glucose subunit
    # 4GB[k]: second glucose subunit 
    # UNL[k]/UNT[k]: gulose subunit 
    # ACX[k]: O-acetyl substitution on C3 hydroxyl of gulose  
    # GLY[k]: glycine modification
    #
    # There are therefore six subunits per monomer (except for the first), 
    # indexed 6*i + 1, ..., 6*i + 6
    #
    # First run through the bonds in the molecule ...
    linkages = []
    for bond in topology.bonds:
        # Identify the functional groups that the two atoms belong to 
        name1 = bond.atom1.residue.name
        name2 = bond.atom2.residue.name

        # Are these functional groups both sugars along the chain?  
        in_backbone1 = all(not name1.startswith(prefix) for prefix in ['ROH', 'ACX', 'GLY'])
        in_backbone2 = all(not name2.startswith(prefix) for prefix in ['ROH', 'ACX', 'GLY'])

        # Any bond that bridges two sugars along the chain constitutes a 
        # glycosidic linkage 
        if name1 != name2 and in_backbone1 and in_backbone2:
            linkages.append([
                (name1, bond.atom1.residue.index),
                (name2, bond.atom2.residue.index)
            ])

    # Now run through the glycosidic linkages and collect the contributing
    # atoms ...
    #
    # The phi angle is given by O5-C1-O4'-C4', and the psi angle is given
    # by C1-O4'-C4'-C3', walking in the Gul -> Glc -> Glc -> Gal direction
    #
    # Going backwards, we start with the very first sugar (Gal) and collect
    # its C3, C4, O4 atoms 
    name, idx = linkages[0][0]
    curr_atoms = [
        atom for atom in topology.atoms if atom.residue.name == name and 
        atom.residue.index == idx
    ]
    
    # Look for the O4, C4, and C3 atoms
    sugar1_O4 = next(atom for atom in curr_atoms if atom.name == 'O4')
    sugar1_C4 = next(atom for atom in curr_atoms if atom.name == 'C4')
    sugar1_C3 = next(atom for atom in curr_atoms if atom.name == 'C3')

    # Add them to the atoms constituting the first linkage 
    linkage_atoms = [[] for _ in range(len(linkages))]
    linkage_atoms[0].append(sugar1_O4)
    linkage_atoms[0].append(sugar1_C4)
    linkage_atoms[0].append(sugar1_C3)

    # Then run through the remaining sugars ... 
    for i in range(len(linkages) - 1):
        name, idx = linkages[i][1]
        curr_atoms = [
            atom for atom in topology.atoms if atom.residue.name == name and 
            atom.residue.index == idx
        ]
        
        # Look for the O5 and C1 atoms in the (i+1)-th sugar 
        sugar2_O5 = next(atom for atom in curr_atoms if atom.name == 'O5')
        sugar2_C1 = next(atom for atom in curr_atoms if atom.name == 'C1')

        # Add them to the atoms constituting the i-th linkage 
        linkage_atoms[i].append(sugar2_O5)
        linkage_atoms[i].append(sugar2_C1)

        # Move onto the O4, C4, and C3 atoms in the (i+1)-th sugar
        name, idx = linkages[i+1][0]
        curr_atoms = [
            atom for atom in topology.atoms if atom.residue.name == name and 
            atom.residue.index == idx
        ]
        sugar1_O4 = next(atom for atom in curr_atoms if atom.name == 'O4')
        sugar1_C4 = next(atom for atom in curr_atoms if atom.name == 'C4')
        sugar1_C3 = next(atom for atom in curr_atoms if atom.name == 'C3')

        # Add them to the atoms constituting the (i+1)-th linkage 
        linkage_atoms[i+1].append(sugar1_O4)
        linkage_atoms[i+1].append(sugar1_C4)
        linkage_atoms[i+1].append(sugar1_C3)

    # Fill in the last two atoms in the final linkage
    name, idx = linkages[-1][1]
    curr_atoms = [
        atom for atom in topology.atoms if atom.residue.name == name and 
        atom.residue.index == idx
    ]
    sugar2_O5 = next(atom for atom in curr_atoms if atom.name == 'O5')
    sugar2_C1 = next(atom for atom in curr_atoms if atom.name == 'C1')
    linkage_atoms[-1].append(sugar2_O5)
    linkage_atoms[-1].append(sugar2_C1)

    return linkage_atoms

########################################################################
def compute_glycosidic_dihedrals(linkage_atoms, traj):
    """
    Compute the phi and psi dihedral angles along each glycosidic linkage 
    along the 10-mer VPS chain, for each frame in the given trajectory.  
    
    This function assumes there are 10 monomers in the segment.

    Parameters
    ----------
    linkage_atoms : list of lists
        Atoms constituting each glycosidic linkage, as obtained from 
        `find_glycosidic_linkages()`.
    traj : `mdtraj.Trajectory`
        Trajectory object.

    Returns
    -------
    Array of \phi and \psi dihedral angles along each glycosidic linkage,
    for each frame in the trajectory. 
    """
    phi = np.zeros((traj.n_frames, len(linkage_atoms)), dtype=np.float64)
    psi = np.zeros((traj.n_frames, len(linkage_atoms)), dtype=np.float64)

    # Run through the frames in the trajectory ... 
    for i, frame in enumerate(traj):
        atoms = list(frame.topology.atoms)
        coords = frame._xyz[0, :, :]

        # For each glycosidic linkage along the chain ... 
        for j in range(len(linkage_atoms)):
            # Find the coordinates of the five atoms contributing to each
            # linkage
            atom_O4 = linkage_atoms[j][0]
            atom_C4 = linkage_atoms[j][1]
            atom_C3 = linkage_atoms[j][2]
            atom_O5 = linkage_atoms[j][3]
            atom_C1 = linkage_atoms[j][4]
            idx_O4 = next(i for i in range(coords.shape[0]) if atoms[i] == atom_O4)
            idx_C4 = next(i for i in range(coords.shape[0]) if atoms[i] == atom_C4)
            idx_C3 = next(i for i in range(coords.shape[0]) if atoms[i] == atom_C3)
            idx_O5 = next(i for i in range(coords.shape[0]) if atoms[i] == atom_O5)
            idx_C1 = next(i for i in range(coords.shape[0]) if atoms[i] == atom_C1)
            r_O4 = coords[idx_O4, :]
            r_C4 = coords[idx_C4, :]
            r_C3 = coords[idx_C3, :]
            r_O5 = coords[idx_O5, :]
            r_C1 = coords[idx_C1, :]

            # Calculate the corresponding dihedrals
            #
            # The bonds are O5-C1, C1-O4', O4'-C4', C4'-C3'
            u1 = r_C1 - r_O5
            u1 /= np.linalg.norm(u1)
            u2 = r_O4 - r_C1
            u2 /= np.linalg.norm(u2)
            u3 = r_C4 - r_O4
            u3 /= np.linalg.norm(u3)
            u4 = r_C3 - r_C4
            u4 /= np.linalg.norm(u4)
            phi_ij = np.arctan2(
                np.linalg.norm(u2) * np.dot(u1, np.cross(u2, u3)), 
                np.dot(np.cross(u1, u2), np.cross(u2, u3))
            )
            psi_ij = np.arctan2(
                np.linalg.norm(u3) * np.dot(u2, np.cross(u3, u4)),
                np.dot(np.cross(u2, u3), np.cross(u3, u4))
            )

            # Map the values into [0, 2*\pi)
            phi[i, j] = (phi_ij if phi_ij >= 0 else (phi_ij + 2 * np.pi))
            psi[i, j] = (psi_ij if psi_ij >= 0 else (psi_ij + 2 * np.pi))

    return phi, psi

########################################################################
def get_monomer_mass(topology):
    """
    Get the mass of a VPS monomer from the topology file, in atomic mass
    units.

    Parameters
    ----------
    topology : `mdtraj.Topology`
        Topology object. 

    Returns
    -------
    VPS monomer mass.
    """
    monomers = find_monomers(topology)
    mass = 0
    for atom in monomers[1]:     # Use an internal monomer
        element = str(atom.element)
        if element == 'hydrogen':
            mass += 1
        elif element == 'carbon': 
            mass += 12
        elif element == 'nitrogen':
            mass += 14
        elif element == 'oxygen':
            mass += 16
        else:
            raise RuntimeError('Encountered unrecognized element: {}'.format(element))

    return mass

########################################################################
def find_monomer_coords(topology_filename, traj_filename, truncate_ends=False):
    """
    Coarse-grain the polymer in each trajectory into one bead per monomer,
    each located at the monomer's center of mass at each timepoint, and 
    return the coordinates of each coarse-grained monomer. 

    Parameters
    ----------
    topology_filename : str
        Amber topology filename. 
    traj_filename : str
        Trajectory filename.
    truncate_ends : bool
        If True, disregard the terminal monomers. 

    Returns
    -------
    3-D array of monomer coordinates ([i, j, k] corresponds to k-th dimension
    (k = 0, 1, 2) of j-th monomer in i-th frame), together with the timepoints
    corresponding to each frame. 
    """
    # Parse the topology and trajectory files 
    topology = load_prmtop(topology_filename)
    traj = load_netcdf(traj_filename, top=topology)

    # Identify the atoms belonging to each monomer 
    monomers = find_monomers(topology)

    # Get the center of mass of each monomer at each timepoint ... 
    monomer_coords = np.zeros((traj.n_frames, len(monomers), 3), dtype=np.float64)
    times = np.zeros(traj.n_frames, dtype=np.float64)
    for i, frame in enumerate(traj):
        # Get the atoms and their coordinates in each frame 
        atoms = list(frame.topology.atoms)
        coords = frame._xyz[0, :, :]
        times[i] = frame.time[0]

        # For each atom, identify the monomer that it belongs to
        for j in range(coords.shape[0]):
            atom = atoms[j]
            element = str(atom.element)
            if element == 'hydrogen':
                mass = 1
            elif element == 'carbon': 
                mass = 12
            elif element == 'nitrogen':
                mass = 14
            elif element == 'oxygen':
                mass = 16
            else:
                raise RuntimeError('Encountered unrecognized element: {}'.format(element))
            for k, monomer in enumerate(monomers):
                if atom in monomers[k]:
                    monomer_coords[i, k, :] += mass * coords[j, :]
                    break

    # Normalize each monomer coordinate by its total mass
    for k in range(len(monomers)):
        monomer_mass = 0
        for atom in monomers[k]:
            element = str(atom.element)
            if element == 'hydrogen':
                mass = 1
            elif element == 'carbon': 
                mass = 12
            elif element == 'nitrogen':
                mass = 14
            elif element == 'oxygen':
                mass = 16
            else:
                raise RuntimeError('Encountered unrecognized element: {}'.format(element))
            monomer_mass += mass
        monomer_coords[:, k, :] /= monomer_mass

    # Truncate the two ends if desired 
    if truncate_ends:
        return monomer_coords[:, 1:-1, :], times
    else:
        return monomer_coords, times

########################################################################
def get_bond_stats(monomer_coords):
    """
    Calculate empirical distributions of coarse-grained bond lengths, bond
    angles, and dihedral angles.

    Parameters
    ----------
    monomer_coords : `numpy.ndarray`
        Input array of monomer coordinates, as obtained from 
        `find_monomer_coords()`.

    Returns
    -------
    2-D arrays of coarse-grained bond lengths, bond angles, and dihedrals 
    ([i, j] corresponds to j-th observable in i-th frame, as defined below).
    """
    lengths = np.zeros((monomer_coords.shape[0], monomer_coords.shape[1] - 1))
    angles = np.zeros((monomer_coords.shape[0], monomer_coords.shape[1] - 2))
    dihedrals = np.zeros((monomer_coords.shape[0], monomer_coords.shape[1] - 3))
    
    # For each timepoint ... 
    for i in range(monomer_coords.shape[0]):
        # Calculate bond lengths ... 
        for j in range(monomer_coords.shape[1] - 1):
            # This is the length of the bond between j and j + 1
            lengths[i, j] = np.linalg.norm(
                monomer_coords[i, j + 1, :] - monomer_coords[i, j, :]
            )
        # ... bond angles ... 
        for j in range(1, monomer_coords.shape[1] - 1):
            # This is the angle at atom j 
            v1 = monomer_coords[i, j - 1, :] - monomer_coords[i, j, :]
            v2 = monomer_coords[i, j + 1, :] - monomer_coords[i, j, :]
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            dot = np.dot(v1, v2)
            if dot < -1:
                angles[i, j - 1] = np.pi
            elif dot > 1:
                angles[i, j - 1] = 0
            else:
                angles[i, j - 1] = np.arccos(np.dot(v1, v2))
        # ... and dihedral angles 
        for j in range(1, monomer_coords.shape[1] - 2):
            # This is the dihedral along the segment formed by atoms j - 1,
            # j, j + 1, j + 2
            v1 = monomer_coords[i, j, :] - monomer_coords[i, j - 1, :]
            v2 = monomer_coords[i, j + 1, :] - monomer_coords[i, j, :]
            v3 = monomer_coords[i, j + 2, :] - monomer_coords[i, j + 1, :]
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            v3 /= np.linalg.norm(v3)
            dihedrals[i, j - 1] = np.arctan2(
                np.linalg.norm(v2) * np.dot(v1, np.cross(v2, v3)), 
                np.dot(np.cross(v1, v2), np.cross(v2, v3))
            )

    return lengths, angles, dihedrals

