"""
Authors:
    Kee-Myoung Nam

Last updated:
    10/22/2025
"""

import sys
import os
sys.path.append(os.path.abspath('../VPS-MD-data/trajectories_500ns'))
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from vonmises_mixture import VonMisesMixture
from polymers import Polymer

#####################################################################
class PolymerTrajectory:
    def __init__(self):
        """
        Trivial constructor. 
        """
        self.ntimes = 0
        self.length = 0
        self.coords = np.zeros((self.ntimes, self.length, 3), dtype=np.float64)
        self.times = np.zeros(self.ntimes, dtype=np.float64)
        self.bounds = np.array([-1, 1, -1, 1, -1, 1], dtype=np.float64)
        self.attributes = {}
    
    #################################################################
    def parse_lammpstrj(self, filename, dt, max_timestep=None):
        """
        Parse the given LAMMPS trajectory file.

        Parameters
        ----------
        filename : str
            Input .lammpstrj filename.
        dt : float
            Time increment in each timestep (in ns). 
        max_timestep : int
            Maximum (integer) timestep at which to stop parsing. 
        """
        timesteps = []
        data = []
        bounds = []

        # First read the first few lines of the file to identify the number of 
        # atoms, the box bounds, and the attributes provided per atom per timestep 
        with open(filename) as f:
            for i in range(3):    # Skip the first three lines 
                f.readline()
            n_atoms = int(f.readline().strip())   # The fourth line gives the number of atoms
            f.readline()          # Skip the next line
            line = f.readline()   # Parse bounds along x-axis
            coords = [float(x) for x in line.strip().split(' ')]
            bounds += coords
            line = f.readline()   # Parse bounds along y-axis
            coords = [float(x) for x in line.strip().split(' ')]
            bounds += coords
            line = f.readline()   # Parse bounds along z-axis
            coords = [float(x) for x in line.strip().split(' ')]
            bounds += coords
            line = f.readline().strip()     # The next line gives the data columns
            attributes = line.split(' ')[3:]

        # Assume that 'xu', 'yu', and 'zu' are among the attributes
        xi = attributes.index('xu')
        yi = attributes.index('yu')
        zi = attributes.index('zu')

        # Then read the entire file ...
        i = -1
        timestep = 0
        curr_data = np.zeros((n_atoms, len(attributes)), dtype=np.float64)
        with open(filename) as f:
            reached_eof = False
            line = f.readline()

            # While we have not reached the end of the file ... 
            while not reached_eof:
                line = line.strip()
                
                # If the line specifies the start of a new block ... 
                if line == 'ITEM: TIMESTEP':
                    # If we are in the very first block, parse the next timestep
                    # and move on 
                    if i == -1:
                        line = f.readline().strip()
                        timestep = int(line)
                        i += 1
                    # Otherwise ... 
                    else:
                        # Collect all data for the current timestep 
                        timesteps.append(timestep)
                        data.append({})
                        # Collect all atom coordinates
                        data[i]['coords'] = curr_data[:, [xi, yi, zi]]
                        # Collect all other coordinates
                        for j, attribute in enumerate(attributes):
                            if attribute != 'xu' and attribute != 'yu' and attribute != 'zu':
                                data[i][attribute] = curr_data[:, j]
                        # Read the next line, which gives the next timestep
                        line = f.readline().strip()
                        timestep = int(line)
                        i += 1
                        # Clear data array for the next timestep 
                        curr_data = np.zeros((n_atoms, len(attributes)), dtype=np.float64) 
                # If the line specifies other metadata ...
                elif line == 'ITEM: NUMBER OF ATOMS':
                    # Skip over this line and the next line 
                    f.readline()
                elif line.startswith('ITEM: BOX BOUNDS'):
                    # Skip over this line and the next three lines 
                    f.readline()
                    f.readline()
                    f.readline()
                # If the line specifies the atom coordinates ... 
                else:
                    if line.startswith('ITEM:'):    # Skip over the header 
                        pass
                    else:
                        coords = line.split(' ')
                        idx = int(coords[0]) - 1
                        for j in range(len(attributes)):
                            curr_data[idx, j] = float(coords[j + 1])
                
                # Read the next line 
                line = f.readline()
                reached_eof = (len(line) == 0)
                
                # Break if we have reached the maximum timestep 
                if max_timestep is not None and timestep > max_timestep:
                    break

        # Populate object attributes 
        self.coords = np.array(
            [data[i]['coords'] for i in range(len(data))], dtype=np.float64
        )
        self.ntimes = self.coords.shape[0]
        self.length = self.coords.shape[1]
        self.times = dt * np.array(timesteps)
        self.bounds = np.array(bounds)
        self.attributes = {}
        for key in data[0]:
            if key != 'coords':
                self.attributes[key] = np.array(
                    [data[i][key] for i in range(len(data))], dtype=np.float64
                )

    #################################################################
    def mean_square_end_to_end_dist(self, tmin=None, tmax=None):
        """
        Get the mean square end-to-end distance. 

        Returns
        -------
        Mean square end-to-end distance, calculated over the given time 
        window. 
        """
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)

        sqdist = 0.0
        n = 0
        for i, t in enumerate(self.times):
            if t >= tmin and t <= tmax:
                polymer = Polymer(coords=self.coords[i, :, :], atom_types=1)
                sqdist += np.dot(polymer.end_to_end(), polymer.end_to_end())
                n += 1

        return sqdist / n

    #################################################################
    def kuhn_length(self, tmin=None, tmax=None):
        """
        Get the Kuhn length of the polymer. 

        Returns 
        -------
        Kuhn length, calculated over the given time window. 
        """
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)

        # Define the maximal end-to-end distance in terms of the mean bond 
        # length
        mean_length = np.mean([
            Polymer(coords=self.coords[i, :, :], atom_types=1).bond_lengths()
            for i in range(self.ntimes)
        ])
        Rmax = (self.length - 1) * mean_length

        return self.mean_square_end_to_end_dist(tmin=tmin, tmax=tmax) / Rmax

    #################################################################
    def radii_of_gyration(self, tmin=None, tmax=None):
        """
        Get the radius of gyration of the polymer over time. 

        Returns
        -------
        Radius of gyration as a function of time over the given time window. 
        """
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)

        radii = []
        for i, t in enumerate(self.times):
            if t >= tmin and t <= tmax:
                polymer = Polymer(coords=self.coords[i, :, :], atom_types=1)
                radii.append(polymer.radius_of_gyration())

        return radii

    #################################################################
    def bond_stats(self, tmin=None, tmax=None):
        """
        Calculate empirical distributions of bond lengths, bond angles, and
        dihedral angles over time.
        """
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)
        lengths = []
        angles = []
        dihedrals = []

        # For each polymer configuration ... 
        for i, t in enumerate(self.times):
            if t >= tmin and t <= tmax:
                # Get the bond lengths, angles, and dihedrals
                polymer = Polymer(coords=self.coords[i, :, :], atom_types=1)
                lengths.append(polymer.bond_lengths())
                angles.append(polymer.bond_angles())
                dihedrals.append(polymer.dihedrals())

        return np.array(lengths), np.array(angles), np.array(dihedrals)

    #################################################################
    def bond_lengths(self, tmin=None, tmax=None):
        """
        Calculate empirical distributions of bond lengths over time.
        """
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)
        lengths = []

        # For each polymer configuration ... 
        for i, t in enumerate(self.times):
            if t >= tmin and t <= tmax:
                # Get the bond lengths, angles, and dihedrals
                polymer = Polymer(coords=self.coords[i, :, :], atom_types=1)
                lengths.append(polymer.bond_lengths())

        return np.array(lengths)

    #################################################################
    def bond_angles(self, tmin=None, tmax=None):
        """
        Calculate empirical distributions of bond angles over time. 
        """
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)
        angles = []

        # For each polymer configuration ... 
        for i, t in enumerate(self.times):
            if t >= tmin and t <= tmax:
                # Get the bond lengths, angles, and dihedrals
                polymer = Polymer(coords=self.coords[i, :, :], atom_types=1)
                angles.append(polymer.bond_angles())

        return np.array(angles)

    #################################################################
    def dihedrals(self, tmin=None, tmax=None):
        """
        Calculate empirical distributions of dihedral angles over time. 
        """
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)
        dihedrals = []

        # For each polymer configuration ... 
        for i, t in enumerate(self.times):
            if t >= tmin and t <= tmax:
                # Get the bond lengths, angles, and dihedrals
                polymer = Polymer(coords=self.coords[i, :, :], atom_types=1)
                dihedrals.append(polymer.dihedrals())

        return np.array(dihedrals)

    #################################################################
    def squared_displacements(self, tmin=None, tmax=None):
        """
        Calculate the squared displacements of the monomers. 
        """
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)
        idxmin = np.argmin(np.abs(self.times - tmin))
        tmin = self.times[idxmin]
        idxmax = np.argmin(np.abs(self.times - tmax))
        tmax = self.times[idxmax]

        # For each timepoint after the first in the desired window ...
        sqdists = np.zeros((idxmax - idxmin, self.length), dtype=np.float64)
        for i, j in enumerate(range(idxmin + 1, idxmax + 1)):
            # Calculate the squared displacement of each monomer, relative to its
            # initial position 
            for k in range(self.length):
                sqdists[i, k] = np.linalg.norm(
                    self.coords[j, k, :] - self.coords[idxmin, k, :]
                ) ** 2

        return sqdists
    
    #################################################################
    def fit_diffusion_coef(self, tmin=None, tmax=None):
        """
        Fit a diffusion coefficient to the mean squared displacement of the 
        monomers. 
        """
        sqdists = self.squared_displacements(tmin=tmin, tmax=tmax)
        mean_sqdists = np.mean(sqdists, axis=1)

        # Fit a line to the mean squared displacement as a function of time
        params, _ = curve_fit(lambda x, m: m * x, self.times[1:], mean_sqdists)
        slope = params[0]

        return slope / 6

    #################################################################
    def fit_bond_length_mixture(self, n_components=2, tmin=None, tmax=None):
        """
        """
        # Get the bond lengths over the given time window 
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)
        lengths = []

        # For each polymer configuration ... 
        for i, t in enumerate(self.times):
            if t >= tmin and t <= tmax:
                # Gather the bond lengths 
                polymer = Polymer(coords=self.coords[i, :, :], atom_types=1)
                lengths.append(polymer.bond_lengths())
        lengths = np.array(lengths)

        # Fit the mixture model 
        length_mixture = GaussianMixture(
            n_components=n_components, covariance_type='full'
        )
        length_mixture.fit(lengths.reshape(-1, 1))

        return length_mixture

    #################################################################
    def fit_bond_angle_mixture(self, rng, n_components=4, n_sample=None,
                               tmin=None, tmax=None,
                               init_weights=[0.45, 0.05, 0.05, 0.45],
                               init_means=[-160 / 180 * np.pi, -np.pi / 2, np.pi / 2, 160 / 180 * np.pi],
                               init_kappa=[5, 1, 1, 5]):
        """
        """
        # Get the bond angles over the given time window 
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)
        angles = []

        # For each polymer configuration ... 
        for i, t in enumerate(self.times):
            if t >= tmin and t <= tmax:
                # Gather the bond angles
                polymer = Polymer(coords=self.coords[i, :, :], atom_types=1)
                angles.append(polymer.bond_angles())
        angles = np.array(angles)

        # Make half the angles negative 
        coefs = rng.choice([-1, 1], size=angles.shape)
        angles_ = (angles * coefs).reshape(-1)

        # Fit a von Mises mixture model
        if n_sample is None:
            n_sample = angles_.shape[0]
        angle_mixture = VonMisesMixture(n_components=n_components)
        angle_mixture.fit(
            angles_[rng.choice(len(angles_), n_sample, replace=False)], rng,
            init_weights=init_weights, init_means=init_means,
            init_kappa=init_kappa, verbose=True
        )

        return angle_mixture

    #################################################################
    def fit_dihedral_mixture(self, rng, n_components=4, n_sample=None,
                             tmin=None, tmax=None,
                             init_weights=[0.45, 0.05, 0.05, 0.45],
                             init_means=[-160 / 180 * np.pi, -np.pi / 2, np.pi / 2, 160 / 180 * np.pi],
                             init_kappa=[5, 1, 1, 5]):
        """
        """
        # Get the dihedral angles over the given time window 
        if tmin is None:
            tmin = 0.0
        if tmax is None:
            tmax = np.max(self.times)
        dihedrals = []

        # For each polymer configuration ... 
        for i, t in enumerate(self.times):
            if t >= tmin and t <= tmax:
                # Get the bond lengths, angles, and dihedrals
                polymer = Polymer(coords=self.coords[i, :, :], atom_types=1)
                dihedrals.append(polymer.dihedrals())
        dihedrals = np.array(dihedrals)

        # Make half the angles and dihedrals negative 
        coefs = rng.choice([-1, 1], size=dihedrals.shape)
        dihedrals_ = (dihedrals * coefs).reshape(-1)

        # Fit a von Mises mixture model
        if n_sample is None:
            n_sample = dihedrals_.shape[0]
        dihedral_mixture = VonMisesMixture(n_components=n_components)
        dihedral_mixture.fit(
            dihedrals_[rng.choice(len(dihedrals_), n_sample, replace=False)],
            rng, init_weights=init_weights, init_means=init_means,
            init_kappa=init_kappa, verbose=True
        )

        return dihedral_mixture

#####################################################################
if __name__ == '__main__':
    prefix = sys.argv[1]
    sim = PolymerTrajectory()
    sim.parse_lammpstrj('{}.lammpstrj'.format(prefix), 1e-6, max_timestep=None)
    rng = np.random.default_rng(1234567890)

    # Fit mixture models for each distribution
    length_mixture = sim.fit_bond_length_mixture(n_components=2)
    angle_mixture = sim.fit_bond_angle_mixture(
        rng, n_components=4, n_sample=2000, init_weights=[0.45, 0.05, 0.05, 0.45],
        init_means=[-160 / 180 * np.pi, -np.pi / 2, np.pi / 2, 160 / 180 * np.pi],
        init_kappa=[5, 1, 1, 5]
    )
    dihedral_mixture = sim.fit_dihedral_mixture(
        rng, n_components=4, n_sample=2000, init_weights=[0.45, 0.05, 0.05, 0.45],
        init_means=[-160 / 180 * np.pi, -np.pi / 2, np.pi / 2, 160 / 180 * np.pi],
        init_kappa=[5, 1, 1, 5]
    )
    length_weights = length_mixture.weights_
    length_means = length_mixture.means_[:, 0]
    length_stds = np.sqrt(length_mixture.covariances_[:, 0, 0])
    print(length_means, length_stds, length_weights)
    print(angle_mixture.means, angle_mixture.kappa, angle_mixture.weights)
    print(dihedral_mixture.means, dihedral_mixture.kappa, dihedral_mixture.weights)

    # Get the empirical bond length, bond angle, and dihedral angle distributions
    lengths, angles, dihedrals = sim.bond_stats()

    # Make some of the angles negative
    coefs = rng.choice([-1, 1], size=angles.shape)
    angles_ = (angles * coefs).reshape(-1)
    coefs = rng.choice([-1, 1], size=dihedrals.shape)
    dihedrals_ = (dihedrals * coefs).reshape(-1)

    # Get the corresponding probability density for the bond lengths
    x = np.linspace(np.min(lengths), np.max(lengths), 100)
    length_pdf = (
        length_weights[0] * norm.pdf(x, loc=length_means[0], scale=length_stds[0]) +
        length_weights[1] * norm.pdf(x, loc=length_means[1], scale=length_stds[1])
    )

    # Plot distributions and mixture densities 
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    axes[0].hist(lengths.reshape(-1), bins=20, density=True)
    axes[0].plot(x, length_pdf)
    axes[1].hist(angles_, bins=np.linspace(-np.pi, np.pi, 21), density=True)
    angle_mixture.plot(axes[1])
    axes[2].hist(dihedrals_, bins=np.linspace(-np.pi, np.pi, 21), density=True)
    dihedral_mixture.plot(axes[2])
    axes[0].set_xlabel('Bond length (nm)')
    axes[0].set_ylabel('Density')
    axes[1].set_xlabel('Bond angle (rad)')
    axes[1].set_ylabel('Density')
    axes[2].set_xlabel('Dihedral angle (rad)')
    axes[2].set_ylabel('Density')
    plt.tight_layout()
    plt.savefig('{}_bond_stats.pdf'.format(prefix))

    # Get squared displacements and diffusion coefficient  
    sqdists = sim.squared_displacements()
    diff_coef = sim.fit_diffusion_coef()
    
    # Plot the squared displacements and the Einstein relation fit (in units
    # of ps, not ns)
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    for j in range(sqdists.shape[1]):
        ax.scatter(
            1000 * sim.times[1:], sqdists[:, j], color=(0.8, 0.8, 0.8),
            rasterized=True
        )
    ax.plot(1000 * sim.times, [6 * diff_coef * t for t in sim.times])
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Squared displacement (nm)')
    plt.savefig('{}_displacements.pdf'.format(prefix))

    # Calculate the effective viscosity, which is kT divided by the 
    # diffusion coefficient
    #
    # Convert the diffusion coefficient to nm^2/ps
    diff_coef /= 1000
    kT = 4.141947e-6              # 1 kT = 4.141947 ag*nm^2/ns^2
    viscosity = kT / diff_coef    # As defined in fix viscous in LAMMPS

    # Calculate the damping factor, which is the monomer mass divided
    # by the viscosity
    #
    # Convert to attograms from atomic mass units
    monomer_mass = 800
    damp_coef = monomer_mass * 1.66053906892e-6 / viscosity
    print('Diffusion coefficient: {:.10f} nm^2 / ps'.format(diff_coef))
    print('Effective viscosity: {:.10f} ag / ps'.format(viscosity))
    print('Damping coefficient: {:.10f} ps'.format(damp_coef)) 

    # Identify the distribution of dihedral angles adjacent to the smaller
    # bond angles 
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
    angle_order = []
    for i in range(angle_mixture.n_components):
        angle_order.append((i, angle_mixture.means[i])) 
    angle_order.sort(key=lambda x: np.abs(x[1]))
    print(angle_mixture.means, np.abs(angle_mixture.means), angle_order)
    nt, na = angles.shape
    adj_dihedrals_small_angles = []
    adj_dihedrals_large_angles = []
    for i in range(nt):
        angle_states = angle_mixture.predict(angles[i, :])
        for j in range(1, na - 1):
            # If the angle is small ...
            if next(k for k, (m, p) in enumerate(angle_order) if angle_states[j] == m) < 2:
                # The j-th angle is the angle formed by atoms j, j+1, j+2
                #
                # Therefore, the adjacent dihedrals are those formed by
                # j-1, j, j+1, j+2 and by j, j+1, j+2, j+3
                adj_dihedrals_small_angles.append([
                    dihedrals[i, j - 1], dihedrals[i, j]
                ])
            else:
                adj_dihedrals_large_angles.append([
                    dihedrals[i, j - 1], dihedrals[i, j]
                ])
    adj_dihedrals_small_angles = np.array(adj_dihedrals_small_angles)
    adj_dihedrals_large_angles = np.array(adj_dihedrals_large_angles)
    sns.kdeplot(
        x=adj_dihedrals_small_angles[:, 0],
        y=adj_dihedrals_small_angles[:, 1],
        fill=True,
        cbar=True,
        ax=axes[0]
    )
    sns.kdeplot(
        x=adj_dihedrals_large_angles[:, 0],
        y=adj_dihedrals_large_angles[:, 1],
        fill=True,
        cbar=True,
        ax=axes[1]
    )
    axes[0].set_title('Small angles')
    axes[1].set_title('Large angles')
    for j in range(2):
        axes[j].set_xlabel(r'$\phi$ (rad)')
        axes[j].set_ylabel(r'$\psi$ (rad)')
    plt.tight_layout()
    plt.savefig('{}_dihedrals.pdf'.format(prefix))

