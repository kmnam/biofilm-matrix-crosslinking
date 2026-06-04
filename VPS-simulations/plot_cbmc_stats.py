"""
Analyze dimensions (radius of gyration, end-to-end distance, persistence
length) of sampled polymer configurations.

Authors:
    Kee-Myoung Nam

Last updated:
    4/13/2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import pyvista as pv
import seaborn as sns
from utils import parse_configurations

########################################################################
def parse_lammpstrj(filename):
    """
    Parse the polymer configurations in the given .lammpstrj file. 
    """
    coords = []

    # Start parsing the file ...
    with open(filename) as f:
        # Read the first line (ITEM: TIMESTEP)
        f.readline()
        
        # Parse past the number of atoms and the box bounds ...
        f.readline()           # Timestep 
        f.readline()           # ITEM: NUMBER OF ATOMS
        line = f.readline()    # Number of atoms
        length = int(line.strip())
        f.readline()           # ITEM: BOX BOUNDS pp pp pp
        f.readline()           # x-bounds
        f.readline()           # y-bounds
        f.readline()           # z-bounds
        f.readline()           # ITEM: ATOMS id mol ...

        # Parse the rest of the file ... 
        curr_coords = []
        for line in f:
            # If we encounter a new timestep ... 
            if line.startswith('ITEM: TIMESTEP'):
                # Keep track of the last set of coordinates 
                coords.append(curr_coords)
                curr_coords = []

                # Parse past the number of atoms and the box bounds ...
                f.readline()           # Timestep 
                f.readline()           # ITEM: NUMBER OF ATOMS
                line = f.readline()    # Number of atoms
                length = int(line.strip())
                f.readline()           # ITEM: BOX BOUNDS pp pp pp
                f.readline()           # x-bounds
                f.readline()           # y-bounds
                f.readline()           # z-bounds
                f.readline()           # ITEM: ATOMS id mol ...
            else:    # Otherwise, parse the coordinates in the line 
                data = line.strip().split()
                curr_coords.append([float(data[3]), float(data[4]), float(data[5])])

    # Return the entire set of coordinates as an array
    return np.array(coords)

########################################################################
def get_energy(coords, eps=1.0, sigma=1.8652849741, k_fene=9.0, R0=2.7979274611):
    """
    Calculate the energy of each polymer configuration in the given 
    coordinate array.

    These polymers are assumed to be random coils with FENE bonds and 
    non-bonded interactions. 
    """
    kT = 4.141947
    eps *= kT
    k_fene *= kT

    # For each non-bonded pair of atoms, calculate the Lennard-Jones energy
    #
    # Assume that the coordinates are unwrapped
    n1, n2, _ = coords.shape
    energies = np.zeros(n1, dtype=np.float64)
    dmin = (2 ** (1. / 6.)) * sigma
    for i in range(n1):
        for j in range(n2):
            for k in range(j + 2, n2):
                dist = np.linalg.norm(coords[i, j, :] - coords[i, k, :])
                if dist < dmin:
                    r6 = (sigma / dist) ** 6
                    energies[i] += 4 * eps * (r6 * r6 - r6) + eps

    # For each pair of bonded atoms, calculate the FENE energy 
    for i in range(n1):
        for j in range(n2 - 1):
            dist = np.linalg.norm(coords[i, j + 1, :] - coords[i, j, :])
            if dist < dmin:
                r6 = (sigma / dist) ** 6
                energies[i] += 4 * eps * (r6 * r6 - r6) + eps
            energies[i] -= 0.5 * k_fene * R0 * R0 * np.log(1 - (dist / R0) ** 2)

    return energies

########################################################################
def get_radius_of_gyration(coords):
    """
    Get the radius of gyration of each polymer in the given coordinate 
    array. 
    """
    center = np.mean(coords, axis=1)
    delta = coords - center.reshape((-1, 1, 3))
    return np.sqrt(
        (np.linalg.norm(delta, axis=2) ** 2).sum(axis=1) / delta.shape[1]
    )

########################################################################
def get_end_to_end_dist(coords):
    """
    Get the end-to-end distance of each polymer in the given coordinate 
    array. 
    """
    r1 = coords[:, 0, :]
    rN = coords[:, -1, :]
    return np.linalg.norm(rN - r1, axis=1)

########################################################################
def get_persistence_length(k, autocorr, mean_bond_length, helical=False):
    """
    Given an array of tangent vector autocorrelations for different increments,
    calculate the persistence length. 
    """
    if not helical:
        fit = curve_fit(
            lambda x, p: np.exp(-x * mean_bond_length / p), k, autocorr
        )
        return fit[0][0] 
    else:
        fit = curve_fit(
            lambda x, p, q, r: np.exp(-x * mean_bond_length / p) * np.cos(q * x + r),
            k, autocorr
        )
        return fit[0]

########################################################################
# Parse 2100-mer configurations for each kink fraction and plot:
# - their mean dimensions (radius of gyration and persistence length)
# - their energy distributions
# - their radius of gyration distributions 
# -------------------------------------------------------------------- #
fractions = [100, 95, 90, 85, 80, 75, 70]
fig1, axes1 = plt.subplots(nrows=len(fractions), ncols=2, figsize=(6, 6))
fig2, axes2 = plt.subplots(nrows=4, ncols=2, figsize=(6, 3.5))

coords_all = {}
energies = {}
energies_bins = {}
radii = {}
radii_bins = {}
rms_radii = {}
rms_end_to_end_dists = {}
persistence_lengths = {}

# Parse each set of 2100-mer configurations for each kink fraction 
for i, fraction in enumerate(fractions):
    coords_i = []
    energies_i = []
    radii_i = []
    end_to_end_dists_i = []
    persistence_lengths_i = []
    for j in range(10):
        # Parse configurations, their energies, and their radii of gyration 
        config_filename = 'data/cbmc/cbmc_gaussian_{}_2100mer_dihedral_{}.txt'.format(fraction, j)
        print('... parsing: {}'.format(config_filename))
        coords_ij, energies_ij, radii_ij = parse_configurations(config_filename)
        coords_i.append(coords_ij)
        energies_i.append(energies_ij)
        radii_i.append(radii_ij)

        # Calculate their end-to-end distances
        end_to_end_dists_i.append(get_end_to_end_dist(coords_ij))

        # Parse tangent vector autocorrelation function for the ensemble and
        # calculate the persistence length
        autocorr_filename = 'data/cbmc/cbmc_gaussian_{}_2100mer_dihedral_{}_autocorr.txt'.format(fraction, j)
        autocorr_data = np.loadtxt(autocorr_filename, delimiter='\t')
        persistence_length = get_persistence_length(
            autocorr_data[:, 0], autocorr_data[:, 1], 1.8, helical=False
        )
        persistence_lengths_i.append(persistence_length)
        print('- radius of gyration:', np.sqrt(np.mean(radii_i[-1] ** 2)))
        print('- end-to-end:', np.sqrt(np.mean(end_to_end_dists_i[-1] ** 2)))
        print('- persistence length:', persistence_lengths_i[-1])

    # The coordinates are stored in an array of size (n1, n2, n3, 3), where
    # n1 is the number of runs (10), n2 is the number of configurations per
    # run, and n3 is the chain length
    coords_i = np.array(coords_i)

    # The energies, radii of gyration, and end-to-end distances are stored
    # in arrays of size (n1, n2) 
    energies_i = np.array(energies_i) / 1000    # Convert to units of 10^3 * kT
    energies[fraction] = energies_i
    radii_i = np.array(radii_i)
    radii[fraction] = radii_i
    end_to_end_dists_i = np.array(end_to_end_dists_i)

    # The persistence lengths are stored in an array of size (n1,)
    persistence_lengths_i = np.array(persistence_lengths_i)

    # Plot distributions of energy and radius of gyration
    #
    # Exclude outliers from the energy distribution, if any exist  
    energy_q1 = np.quantile(energies_i, 0.25)
    energy_q3 = np.quantile(energies_i, 0.75)
    iqr = energy_q3 - energy_q1
    min_energy = energy_q1 - 10 * iqr
    max_energy = energy_q3 + 10 * iqr
    is_outlier = ((energies_i < min_energy) | (energies_i > max_energy))
    print('- found {} / {} energetic outliers'.format(is_outlier.sum(), energies_i.size))

    # Get a set of bins for the pooled distribution
    energies_bins[fraction] = np.histogram_bin_edges(
        energies_i[~is_outlier], bins=20
    )

    # Plot the energy distribution for each run separately 
    #
    # Plot the distribution for all fractions on the first set of axes 
    for j in range(10):
        is_outlier_j = is_outlier[j, :]
        axes1[i, 0].hist(
            energies_i[j, :][~is_outlier_j], bins=energies_bins[fraction],
            density=True, histtype='step'
        )
    # Also plot the distribution for all even-indexed fractions (0, 2, ...)
    # on the second set of axes 
    if i % 2 == 0:
        for j in range(10):
            axes2[i // 2, 0].hist(
                energies_i[j, :][~is_outlier_j], bins=energies_bins[fraction], 
                density=True, histtype='step'
            )

    # Do the same for the radii of gyration ... 
    #
    # First get a set of bins for the pooled distribution 
    radii_bins[fraction] = np.histogram_bin_edges(radii_i, bins=20)
    
    # Then plot the Rg distribution for each run separately
    #
    # Plot the distribution for all fractions on the first set of axes 
    for j in range(10):
        axes1[i, 1].hist(
            radii_i[j, :].reshape(-1), bins=radii_bins[fraction], density=True,
            histtype='step'
        )
    # Also plot the distribution for all even-indexed fractions (0, 2, ...)
    # on the second set of axes
    if i % 2 == 0:
        for j in range(10):
            axes2[i // 2, 1].hist(
                radii_i[j, :].reshape(-1), bins=radii_bins[fraction],
                density=True, histtype='step'
            )

    # Annotate each plot at the top right with the kink fraction
    for j in range(2):
        axes1[i, j].annotate(
            '{}%'.format(100 - fraction), (0.98, 0.92), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top'
        )
    if i % 2 == 0:
        for j in range(2):
            axes2[i // 2, j].annotate(
                '{}%'.format(100 - fraction), (0.98, 0.92),
                xycoords='axes fraction', horizontalalignment='right',
                verticalalignment='top'
            )

    # Get the square root of the mean squared radius of gyration and end-to-end
    # distance for each run
    rms_radii[fraction] = np.sqrt(np.mean(radii_i ** 2, axis=1))
    rms_end_to_end_dists[fraction] = np.sqrt(np.mean(end_to_end_dists_i ** 2, axis=1))

    # Keep track of the persistence lengths for each run 
    persistence_lengths[fraction] = persistence_lengths_i

    # Gather the coordinates of the configurations 
    _, n2, n3, _ = coords_i.shape
    coords_all[fraction] = coords_i.reshape((10 * n2, n3, 3)) 

# Configure axes labels 
for i in range(len(fractions)):
    axes1[i, 0].set_ylabel('Density')
axes1[-1, 0].set_xlabel(r'Energy ($10^3 k_{\mathrm{B}} T$)')
axes1[-1, 1].set_xlabel(r'$R_{\mathrm{g}}$ (nm)')
fig1.tight_layout()
fig1.savefig('cbmc_gaussian_dihedral_dists_separate.pdf')
for i in range(4):
    axes2[i, 0].set_ylabel('Density')
axes2[-1, 0].set_xlabel(r'Energy ($10^3 k_{\mathrm{B}} T$)')
axes2[-1, 1].set_xlabel(r'$R_{\mathrm{g}}$ (nm)')
fig2.tight_layout()
fig2.savefig('cbmc_gaussian_dihedral_dists_separate_subset.pdf')

# Generate the same plots, but with the pooled ensemble ... 
fig1, axes1 = plt.subplots(nrows=len(fractions), ncols=2, figsize=(6, 6))
fig2, axes2 = plt.subplots(nrows=4, ncols=2, figsize=(6, 3.5))
for i, fraction in enumerate(fractions):
    # Plot distributions of energy and radius of gyration
    #
    # Exclude outliers from the energy distribution, if any exist  
    energy_q1 = np.quantile(energies[fraction], 0.25)
    energy_q3 = np.quantile(energies[fraction], 0.75)
    iqr = energy_q3 - energy_q1
    min_energy = energy_q1 - 10 * iqr
    max_energy = energy_q3 + 10 * iqr
    is_outlier = ((energies[fraction] < min_energy) | (energies[fraction] > max_energy))
    axes1[i, 0].hist(
        energies[fraction][~is_outlier].reshape(-1), bins=energies_bins[fraction],
        density=True
    )
    axes1[i, 1].hist(
        radii[fraction].reshape(-1), bins=radii_bins[fraction], density=True
    )
    if i % 2 == 0:
        axes2[i // 2, 0].hist(
            energies[fraction][~is_outlier].reshape(-1),
            bins=energies_bins[fraction], density=True
        )
        axes2[i // 2, 1].hist(
            radii[fraction].reshape(-1), bins=radii_bins[fraction],
            density=True
        )

    # Plot the RMS value for each distribution as a vertical line
    mean_energy = np.mean(energies[fraction][~is_outlier])
    mean_radius = np.mean(radii[fraction])
    ymin, ymax = axes1[i, 0].get_ylim()
    axes1[i, 0].plot([mean_energy, mean_energy], [ymin, ymax], linestyle='--')
    axes1[i, 0].set_ylim([ymin, ymax])
    if i % 2 == 0:
        ymin, ymax = axes2[i // 2, 0].get_ylim()
        axes2[i // 2, 0].plot([mean_energy, mean_energy], [ymin, ymax], linestyle='--')
        axes2[i // 2, 0].set_ylim([ymin, ymax])
    ymin, ymax = axes1[i, 1].get_ylim()
    axes1[i, 1].plot([mean_radius, mean_radius], [ymin, ymax], linestyle='--')
    axes1[i, 1].set_ylim([ymin, ymax])
    if i % 2 == 0:
        ymin, ymax = axes2[i // 2, 1].get_ylim()
        axes2[i // 2, 1].plot([mean_radius, mean_radius], [ymin, ymax], linestyle='--')
        axes2[i // 2, 1].set_ylim([ymin, ymax])

    # Annotate each plot at the top right with the kink fraction
    for j in range(2):
        axes1[i, j].annotate(
            '{}%'.format(100 - fraction), (0.98, 0.92), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top'
        )
        if i % 2 == 0:
            axes2[i // 2, j].annotate(
                '{}%'.format(100 - fraction), (0.98, 0.92),
                xycoords='axes fraction', horizontalalignment='right',
                verticalalignment='top'
            )

# Configure axes labels 
for i in range(len(fractions)):
    axes1[i, 0].set_ylabel('Density')
axes1[-1, 0].set_xlabel(r'Energy ($10^3 k_{\mathrm{B}} T$)')
axes1[-1, 1].set_xlabel(r'$R_{\mathrm{g}}$ (nm)')
fig1.tight_layout()
fig1.savefig('cbmc_gaussian_dihedral_dists.pdf')
for i in range(4):
    axes2[i, 0].set_ylabel('Density')
axes2[-1, 0].set_xlabel(r'Energy ($10^3 k_{\mathrm{B}} T$)')
axes2[-1, 1].set_xlabel(r'$R_{\mathrm{g}}$ (nm)')
fig2.tight_layout()
fig2.savefig('cbmc_gaussian_dihedral_dists_subset.pdf')

# Plot RMS radii of gyration and persistence lengths
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 4))
df = pd.DataFrame({
    'fraction': [100 - fraction for fraction in fractions for _ in range(10)],
    'radius': np.array([rms_radii[fraction] for fraction in fractions]).reshape(-1),
    'persist': np.array(
        [persistence_lengths[fraction] for fraction in fractions]
    ).reshape(-1)
})
sns.boxplot(data=df, x='fraction', y='radius', showfliers=False, ax=axes[0])
sns.boxplot(data=df, x='fraction', y='persist', showfliers=False, ax=axes[1])
axes[0].set_xlabel('')
axes[1].set_xlabel('Kink fraction (%)', size=14)
axes[0].set_ylabel(r'$R_{\mathrm{g}}$ (nm)', size=14)
axes[1].set_ylabel(r'$\ell_{\mathrm{p}}$ (nm)', size=14)
axes[0].tick_params(labelsize=14)
axes[1].tick_params(labelsize=14)
plt.tight_layout()
plt.savefig('cbmc_gaussian_dihedral_dims.pdf')

# -------------------------------------------------------------------- #
# Parse 2100-mer configurations with other dihedral angle stiffnesses, 
# and do the same calculations 
# -------------------------------------------------------------------- #
# Parse each set of 2100-mer configurations for each kink fraction 
rms_radii_nodihedral = {}
rms_radii_dihedral_K1 = {}
rms_radii_dihedral_K2 = {}
persistence_lengths_nodihedral = {}
persistence_lengths_dihedral_K1 = {}
persistence_lengths_dihedral_K2 = {}
for i, fraction in enumerate(fractions):
    radii_nodihedral_i = []
    radii_dihedral_K1_i = []
    radii_dihedral_K2_i = []
    persistence_lengths_nodihedral_i = []
    persistence_lengths_dihedral_K1_i = []
    persistence_lengths_dihedral_K2_i = []
    for j in range(10):
        # Start with configurations with no dihedral potential 
        #
        # Parse configurations and their radii of gyration 
        config_filename = 'data/cbmc/cbmc_gaussian_{}_2100mer_{}.txt'.format(fraction, j)
        print('... parsing: {}'.format(config_filename))
        _, _, radii_ij = parse_configurations(config_filename)
        radii_nodihedral_i.append(radii_ij)

        # Parse tangent vector autocorrelation function for the ensemble and
        # calculate the persistence length
        autocorr_filename = 'data/cbmc/cbmc_gaussian_{}_2100mer_{}_autocorr.txt'.format(fraction, j)
        autocorr_data = np.loadtxt(autocorr_filename, delimiter='\t')
        persistence_length = get_persistence_length(
            autocorr_data[:, 0], autocorr_data[:, 1], 1.8, helical=False
        )
        persistence_lengths_nodihedral_i.append(persistence_length)
        print('- radius of gyration:', np.sqrt(np.mean(radii_nodihedral_i[-1] ** 2)))
        print('- persistence length:', persistence_lengths_nodihedral_i[-1])

        # Then parse configurations with K = 1 dihedral potential
        #
        # Parse configurations and their radii of gyration 
        config_filename = 'data/cbmc/cbmc_gaussian_{}_2100mer_dihedral_K1_{}.txt'.format(fraction, j)
        print('... parsing: {}'.format(config_filename))
        _, _, radii_ij = parse_configurations(config_filename)
        radii_dihedral_K1_i.append(radii_ij)

        # Parse tangent vector autocorrelation function for the ensemble and
        # calculate the persistence length
        autocorr_filename = 'data/cbmc/cbmc_gaussian_{}_2100mer_dihedral_K1_{}_autocorr.txt'.format(fraction, j)
        autocorr_data = np.loadtxt(autocorr_filename, delimiter='\t')
        persistence_length = get_persistence_length(
            autocorr_data[:, 0], autocorr_data[:, 1], 1.8, helical=False
        )
        persistence_lengths_dihedral_K1_i.append(persistence_length)
        print('- radius of gyration:', np.sqrt(np.mean(radii_dihedral_K1_i[-1] ** 2)))
        print('- persistence length:', persistence_lengths_dihedral_K1_i[-1])

        # Then parse configurations with K = 2 dihedral potential
        #
        # Parse configurations and their radii of gyration 
        config_filename = 'data/cbmc/cbmc_gaussian_{}_2100mer_dihedral_K2_{}.txt'.format(fraction, j)
        print('... parsing: {}'.format(config_filename))
        _, _, radii_ij = parse_configurations(config_filename)
        radii_dihedral_K2_i.append(radii_ij)

        # Parse tangent vector autocorrelation function for the ensemble and
        # calculate the persistence length
        autocorr_filename = 'data/cbmc/cbmc_gaussian_{}_2100mer_dihedral_K2_{}_autocorr.txt'.format(fraction, j)
        autocorr_data = np.loadtxt(autocorr_filename, delimiter='\t')
        persistence_length = get_persistence_length(
            autocorr_data[:, 0], autocorr_data[:, 1], 1.8, helical=False
        )
        persistence_lengths_dihedral_K2_i.append(persistence_length)
        print('- radius of gyration:', np.sqrt(np.mean(radii_dihedral_K2_i[-1] ** 2)))
        print('- persistence length:', persistence_lengths_dihedral_K2_i[-1])

    # Get the square root of the mean squared radius of gyration for each run
    rms_radii_nodihedral[fraction] = np.sqrt(np.mean(np.array(radii_nodihedral_i) ** 2, axis=1))
    rms_radii_dihedral_K1[fraction] = np.sqrt(np.mean(np.array(radii_dihedral_K1_i) ** 2, axis=1))
    rms_radii_dihedral_K2[fraction] = np.sqrt(np.mean(np.array(radii_dihedral_K2_i) ** 2, axis=1))

    # Keep track of the persistence lengths for each run 
    persistence_lengths_nodihedral[fraction] = persistence_lengths_nodihedral_i
    persistence_lengths_dihedral_K1[fraction] = persistence_lengths_dihedral_K1_i
    persistence_lengths_dihedral_K2[fraction] = persistence_lengths_dihedral_K2_i

# Plot RMS radii of gyration and persistence lengths
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 4))
df = pd.DataFrame({
    'fraction': [100 - fraction for fraction in fractions for _ in range(10)],
    'radius': np.array(
        [rms_radii_nodihedral[fraction] for fraction in fractions]
    ).reshape(-1),
    'persist': np.array(
        [persistence_lengths_nodihedral[fraction] for fraction in fractions]
    ).reshape(-1),
    'K': [0 for _ in fractions for _ in range(10)],
})
df = pd.concat((
    df,
    pd.DataFrame({
        'fraction': [100 - fraction for fraction in fractions for _ in range(10)],
        'radius': np.array([rms_radii[fraction] for fraction in fractions]).reshape(-1),
        'persist': np.array(
            [persistence_lengths[fraction] for fraction in fractions]
        ).reshape(-1),
        'K': [0.5 for _ in fractions for _ in range(10)]
    })
))
df = pd.concat((
    df,
    pd.DataFrame({
        'fraction': [100 - fraction for fraction in fractions for _ in range(10)],
        'radius': np.array(
            [rms_radii_dihedral_K1[fraction] for fraction in fractions]
        ).reshape(-1),
        'persist': np.array(
            [persistence_lengths_dihedral_K1[fraction] for fraction in fractions]
        ).reshape(-1),
        'K': [1 for _ in fractions for _ in range(10)]
    })
))
df = pd.concat((
    df,
    pd.DataFrame({
        'fraction': [100 - fraction for fraction in fractions for _ in range(10)],
        'radius': np.array(
            [rms_radii_dihedral_K2[fraction] for fraction in fractions]
        ).reshape(-1),
        'persist': np.array(
            [persistence_lengths_dihedral_K2[fraction] for fraction in fractions]
        ).reshape(-1),
        'K': [2 for _ in fractions for _ in range(10)]
    })
))
for i, K in enumerate([0, 0.5, 1, 2]):
    sns.boxplot(
        data=df.loc[df['K'] == K], x='fraction', y='radius', color='white',
        linecolor=sns.color_palette()[i], linewidth=1.5, showfliers=False,
        ax=axes[0]
    )
    sns.boxplot(
        data=df.loc[df['K'] == K], x='fraction', y='persist', color='white',
        linecolor=sns.color_palette()[i], linewidth=1.5, showfliers=False,
        ax=axes[1]
    )
axes[0].set_xlabel('')
axes[1].set_xlabel('Kink fraction (%)', size=14)
axes[0].set_ylabel(r'$R_{\mathrm{g}}$ (nm)', size=14)
axes[1].set_ylabel(r'$\ell_{\mathrm{p}}$ (nm)', size=14)
axes[0].tick_params(labelsize=14)
axes[1].tick_params(labelsize=14)
axes[0].legend(
    handles=[
        Line2D(
            [0], [0], color=sns.color_palette()[0], linewidth=1.5,
            label=r'$\kappa = 0$'
        ),
        Line2D(
            [0], [0], color=sns.color_palette()[1], linewidth=1.5,
            label=r'$\kappa = 0.5$'
        ),
        Line2D(
            [0], [0], color=sns.color_palette()[2], linewidth=1.5,
            label=r'$\kappa = 1$'
        ),
        Line2D(
            [0], [0], color=sns.color_palette()[3], linewidth=1.5,
            label=r'$\kappa = 2$'
        )
    ]
)
plt.tight_layout()
plt.savefig('cbmc_gaussian_variable_dihedral_dims.pdf')

# Plot autocorrelation functions 
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4, 4))
suffixes = [
    '0_autocorr',
    'dihedral_0_autocorr',
    'dihedral_K1_0_autocorr',
    'dihedral_K2_0_autocorr'
]
for i, fraction in enumerate([100, 90, 70]):
    for j, suffix in enumerate(suffixes):
        autocorr_data = np.loadtxt(
            'data/cbmc/cbmc_gaussian_{}_2100mer_{}.txt'.format(fraction, suffix),
            delimiter='\t'
        )
        axes[i].plot(autocorr_data[:60, 0], autocorr_data[:60, 1])
axes[-1].set_xlabel(r'$s$', size=14)
for i in range(3):
    axes[i].set_ylabel(r'$C(s)$', size=14)
    axes[i].tick_params(labelsize=14)
axes[0].annotate(
    'KF = 0%', (0.99, 0.92), xycoords='axes fraction', horizontalalignment='right',
    verticalalignment='top', size=14
)
axes[1].annotate(
    'KF = 10%', (0.99, 0.92), xycoords='axes fraction', horizontalalignment='right',
    verticalalignment='top', size=14
)
axes[2].annotate(
    'KF = 30%', (0.99, 0.92), xycoords='axes fraction', horizontalalignment='right',
    verticalalignment='top', size=14
)
plt.tight_layout()
plt.savefig('cbmc_gaussian_variable_dihedral_autocorr.pdf')

# -------------------------------------------------------------------- #
# Parse 2100-mer random coil configurations and plot: 
# - their mean dimensions (radius of gyration and persistence length)
# - their energy distributions
# - their radius of gyration distributions
# -------------------------------------------------------------------- #
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
coords_random = []
energies_random = []
radii_random = []
for j in range(10):
    # Parse configurations, their energies, and their radii of gyration 
    config_filename = 'data/cbmc/cbmc_coil_2100mer_{}.txt'.format(j)
    print('... parsing: {}'.format(config_filename))
    coords_j, energies_j, radii_j = parse_configurations(config_filename)
    coords_random.append(coords_j)
    energies_random.append(energies_j)
    radii_random.append(radii_j)

# The coordinates are stored in an array of size (n1, n2, n3, 3), where
# n1 is the number of runs (10), n2 is the number of configurations per
# run, and n3 is the chain length
coords_random = np.array(coords_random)

# The energies and radii of gyration are stored in arrays of size (n1, n2)
energies_random = np.array(energies_random) / 1000    # Convert to units of 10^3 * kT
radii_random = np.array(radii_random)

# Plot distributions of energy and radius of gyration
#
# Exclude outliers from the energy distribution, if any exist  
energy_q1 = np.quantile(energies_random, 0.25)
energy_q3 = np.quantile(energies_random, 0.75)
iqr = energy_q3 - energy_q1
min_energy = energy_q1 - 10 * iqr
max_energy = energy_q3 + 10 * iqr
is_outlier = ((energies_random < min_energy) | (energies_random > max_energy))
print('- found {} / {} energetic outliers'.format(is_outlier.sum(), energies_random.size))

# Get a set of bins for the pooled distribution
energies_bins_random = np.histogram_bin_edges(
    energies_random[~is_outlier], bins=20
)

# Plot the distribution for each run separately  
for j in range(10):
    is_outlier_j = is_outlier[j, :]
    axes[0].hist(
        energies_random[j, :][~is_outlier_j], bins=energies_bins_random,
        density=True, histtype='step'
    )

# Do the same for the radii of gyration ... 
#
# First get a set of bins for the pooled distribution 
radii_bins_random = np.histogram_bin_edges(radii_random, bins=20)

# Then plot the distribution for each run separately 
for j in range(10):
    axes[1].hist(
        radii_random[j, :].reshape(-1), bins=radii_bins_random, density=True,
        histtype='step'
    )

# Annotate each plot at the top right with the kink fraction
for j in range(2):
    axes[j].annotate(
        'Random coil', (0.98, 0.92), xycoords='axes fraction',
        horizontalalignment='right', verticalalignment='top'
    )

# Configure axes labels 
axes[0].set_ylabel('Density')
axes[0].set_xlabel(r'Energy ($10^3 k_{\mathrm{B}} T$)')
axes[1].set_xlabel(r'$R_{\mathrm{g}}$ (nm)')
plt.tight_layout()
plt.savefig('cbmc_coil_dists_separate.pdf')

# -------------------------------------------------------------------- #
# Make the same plots for the 3000-mers ... 
#
# Parse each set of 3000-mer configurations for each kink fraction 
# -------------------------------------------------------------------- #
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(6, 3))
for i, fraction in enumerate([100, 95, 90]):
    coords_i = []
    energies_i = []
    radii_i = []
    end_to_end_dists_i = []
    persistence_lengths_i = []
    for j in range(10):
        # Parse configurations, their energies, and their radii of gyration 
        config_filename = 'data/cbmc/cbmc_gaussian_{}_3000mer_dihedral_{}.txt'.format(fraction, j)
        print('... parsing: {}'.format(config_filename))
        coords_ij, energies_ij, radii_ij = parse_configurations(config_filename)
        coords_i.append(coords_ij)
        energies_i.append(energies_ij)
        radii_i.append(radii_ij)

        # Calculate their end-to-end distances
        end_to_end_dists_i.append(get_end_to_end_dist(coords_ij))
        print('- end-to-end:', np.sqrt(np.mean(end_to_end_dists_i[-1] ** 2)))

    # The coordinates are stored in an array of size (n1, n2, n3, 3), where
    # n1 is the number of runs (10), n2 is the number of configurations per
    # run, and n3 is the chain length
    coords_i = np.array(coords_i)

    # The energies, radii of gyration, and end-to-end distances are stored
    # in arrays of size (n1, n2) 
    energies_i = np.array(energies_i) / 1000    # Convert to units of 10^3 * kT
    radii_i = np.array(radii_i)
    end_to_end_dists_i = np.array(end_to_end_dists_i)

    # Plot distributions of energy and radius of gyration
    #
    # Exclude outliers from the energy distribution, if any exist  
    energy_q1 = np.quantile(energies_i, 0.25)
    energy_q3 = np.quantile(energies_i, 0.75)
    iqr = energy_q3 - energy_q1
    min_energy = energy_q1 - 10 * iqr
    max_energy = energy_q3 + 10 * iqr
    is_outlier = ((energies_i < min_energy) | (energies_i > max_energy))
    print('- found {} / {} energetic outliers'.format(is_outlier.sum(), energies_i.size))

    # Get a set of bins for the pooled distribution
    energies_bins[fraction] = np.histogram_bin_edges(
        energies_i[~is_outlier], bins=20
    )

    # Plot the distribution for each run separately  
    for j in range(10):
        is_outlier_j = is_outlier[j, :]
        axes[i, 0].hist(
            energies_i[j, :][~is_outlier_j], bins=energies_bins[fraction],
            density=True, histtype='step'
        )

    # Do the same for the radii of gyration ... 
    #
    # First get a set of bins for the pooled distribution 
    radii_bins[fraction] = np.histogram_bin_edges(radii_i, bins=20)
    
    # Then plot the distribution for each run separately 
    for j in range(10):
        axes[i, 1].hist(
            radii_i[j, :].reshape(-1), bins=radii_bins[fraction], density=True,
            histtype='step'
        )

    # Annotate each plot at the top right with the kink fraction
    for j in range(2):
        axes[i, j].annotate(
            '{}%'.format(100 - fraction), (0.98, 0.92), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top'
        )

# Configure axes labels 
for i in range(3):
    axes[i, 0].set_ylabel('Density')
axes[-1, 0].set_xlabel(r'Energy ($10^3 k_{\mathrm{B}} T$)')
axes[-1, 1].set_xlabel(r'$R_{\mathrm{g}}$ (nm)')
plt.tight_layout()
plt.savefig('cbmc_gaussian_3000mer_dihedral_dists_separate.pdf')

# -------------------------------------------------------------------- #
# (Re-)parse the 300-, 500-, 1000-, 2100-, 3000-mer configurations and plot: 
# - their end-to-end distances vs. their radii of gyration
# - their lengths vs. their radii of gyration 
# -------------------------------------------------------------------- #
fractions = [100, 95, 90]
lengths = [300, 500, 1000, 2100, 3000]
sq_radii_all_lengths = {}
sq_end_to_end_dists_all_lengths = {}
for i, fraction in enumerate(fractions):
    for j, length in enumerate(lengths):
        energies = []
        radii = []
        end_to_end_dists = []
        n_runs = (50 if length >= 1000 else 10)
        for k in range(n_runs):
            filename = 'data/cbmc/cbmc_gaussian_{}_{}mer_dihedral_{}.txt'.format(
                fraction, length, k
            )
            print('... parsing: {}'.format(filename))
            coords_k, energies_k, radii_k = parse_configurations(filename)
            energies.append(energies_k)
            radii.append(radii_k)
            end_to_end_dists.append(get_end_to_end_dist(coords_k))
        energies = np.array(energies) / 1000
        radii = np.array(radii)
        end_to_end_dists = np.array(end_to_end_dists)

        # Get the RMS radius of gyration and end-to-end distance over each run
        sq_radii_all_lengths[(fraction, length)] = radii.reshape(-1) ** 2
        sq_end_to_end_dists_all_lengths[(fraction, length)] = end_to_end_dists.reshape(-1) ** 2

# Plot RMS radii of gyration and end-to-end distances for different lengths
# and kink fractions 
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
fraction_colors = sns.color_palette('deep')[:len(fractions)]
length_markers = ['X', 'v', '^', '*', 'P', 'o', 'd']
for i, (fraction, color) in enumerate(zip(fractions, fraction_colors)):
    for length, markerstyle in zip(lengths, length_markers):
        rms_Rg = np.sqrt(np.mean(sq_radii_all_lengths[(fraction, length)]))
        rms_Re = np.sqrt(np.mean(sq_end_to_end_dists_all_lengths[(fraction, length)]))
        print(
            '{}% {}-mers Rg:'.format(fraction, length), rms_Rg, ', Re:', rms_Re
        )
        axes[0].scatter(
            [rms_Rg], [rms_Re],
            color=color,
            marker=markerstyle,
            zorder=i+1
        )
        axes[1].scatter(
            [length], [rms_Rg],
            color=color,
            marker='.',
            zorder=i
        )

# Parse random coil configurations and calculate radii of gyration and end-to-end
# distances
sq_radii_random = {}
sq_end_to_end_dists_random = {}
for i, length in enumerate(lengths):
    radii_i = []
    end_to_end_dists_i = []
    for j in range(10):
        # Parse configurations and their radii of gyration 
        config_filename = 'data/cbmc/cbmc_coil_{}mer_{}.txt'.format(length, j)
        print('... parsing: {}'.format(config_filename))
        coords_ij, _, radii_ij = parse_configurations(config_filename)
        radii_i.append(radii_ij)

        # Calculate their end-to-end distances
        end_to_end_dists_i.append(get_end_to_end_dist(coords_ij))

    # The radii of gyration and end-to-end distances are stored in arrays of
    # size (n1, n2) 
    radii_i = np.array(radii_i)
    end_to_end_dists_i = np.array(end_to_end_dists_i)

    # Get the RMS radius of gyration and end-to-end distance for each run
    sq_radii_random[length] = radii_i.reshape(-1) ** 2
    sq_end_to_end_dists_random[length] = end_to_end_dists_i.reshape(-1) ** 2

# Plot the dimensions of the random coils 
for length, markerstyle in zip(lengths, length_markers):
    rms_Rg = np.sqrt(np.mean(sq_radii_random[length]))
    rms_Re = np.sqrt(np.mean(sq_end_to_end_dists_random[length]))
    print('coil {}-mers Rg:'.format(length), rms_Rg, ', Re:', rms_Re)
    axes[0].scatter(
        [rms_Rg], [rms_Re],
        color=sns.color_palette('deep')[len(fractions)],
        marker=markerstyle,
        zorder=i+1
    )
    axes[1].scatter(
        [length], [rms_Rg],
        color=sns.color_palette('deep')[len(fractions)],
        marker='.',
        zorder=i
    )

# Fit radius of gyration vs. end-to-end distance relations for a random coil
x = [np.sqrt(np.mean(sq_radii_random[length])) for length in lengths]
y = [np.sqrt(np.mean(sq_end_to_end_dists_random[length])) for length in lengths]
random_Rg_Re_fit = curve_fit(lambda x_, m: m * x_, x, y)
print(random_Rg_Re_fit)

# Fit radius of gyration vs. end-to-end distance relations for each kink
# fraction 
Rg_Re_fits = {}
for fraction in fractions:
    x = [
        np.sqrt(np.mean(sq_radii_all_lengths[(fraction, length)]))
        for length in lengths
    ]
    y = [
        np.sqrt(np.mean(sq_end_to_end_dists_all_lengths[(fraction, length)]))
        for length in lengths
    ]
    Rg_Re_fits[fraction] = curve_fit(lambda x_, m: m * x_, x, y)
    print(fraction, Rg_Re_fits[fraction])
xmin, xmax = axes[0].get_xlim()
x = np.linspace(xmin, xmax, 100)
for i, fraction in enumerate(fractions):
    m = Rg_Re_fits[fraction][0][0]
    axes[0].plot(
        x, m * x, linestyle='--', color=fraction_colors[i],
        zorder=0
    )
m = random_Rg_Re_fit[0][0]
axes[0].plot(
    x, m * x, linestyle='--', color=sns.color_palette('deep')[len(fraction_colors)],
    zorder=0
)
axes[0].set_xlim([xmin, xmax])

# Fit radius of gyration vs. end-to-end distance relations for a random coil
x = lengths
y = [np.sqrt(np.mean(sq_radii_random[length])) for length in lengths]
random_N_Rg_fit = curve_fit(lambda x_, m, b: m * x_ + b, np.log10(x), np.log10(y))
print(random_N_Rg_fit)

# Fit the polymer length vs. radius of gyration relations
N_Rg_fits = {}
for fraction in fractions:
    x = lengths
    y = [
        np.sqrt(np.mean(sq_radii_all_lengths[(fraction, length)]))
        for length in lengths
    ]
    N_Rg_fits[fraction] = curve_fit(
        lambda x_, m, b: m * x_ + b, np.log10(x), np.log10(y)
    )
    print(fraction, N_Rg_fits[fraction])
xmin = 50
_, xmax = axes[1].get_xlim()
x = np.linspace(xmin, xmax, 100)
for i, fraction in enumerate(fractions):
    m = N_Rg_fits[fraction][0][0]
    b = N_Rg_fits[fraction][0][1]
    axes[1].plot(
        x, (10 ** b) * (x ** m), linestyle='--', color=fraction_colors[i],
        zorder=0
    )
m = random_N_Rg_fit[0][0]
b = random_N_Rg_fit[0][1]
axes[1].plot(
    x, (10 ** b) * (x ** m), linestyle='--',
    color=sns.color_palette('deep')[len(fraction_colors)],
    zorder=0
)
axes[1].set_xlim([xmin, xmax])

# Add legends
first_legend = axes[0].legend(
    handles=[
        Line2D(
            [0], [0], color=fraction_colors[0], linewidth=4,
            label=r'$\mathrm{KF} = 0\%$'
        ),
        Line2D(
            [0], [0], color=fraction_colors[1], linewidth=4,
            label=r'$\mathrm{KF} = 5\%$'
        ),
        Line2D(
            [0], [0], color=fraction_colors[2], linewidth=4,
            label=r'$\mathrm{KF} = 10\%$'
        ),
        Line2D(
            [0], [0], color=sns.color_palette('deep')[3], linewidth=4,
            label='Random coil'
        )
    ],
    fontsize=7,
    loc='upper left'
)
axes[0].add_artist(first_legend)
axes[0].legend(
    handles=[
        Line2D(
            [0], [0], marker=length_markers[0], color='white',
            markerfacecolor='grey', markersize=8, label=r'$N = 100$'
        ),
        Line2D(
            [0], [0], marker=length_markers[1], color='white',
            markerfacecolor='grey', markersize=8, label=r'$N = 200$'
        ),
        Line2D(
            [0], [0], marker=length_markers[2], color='white', 
            markerfacecolor='grey', markersize=8, label=r'$N = 300$'
        ),
        Line2D(
            [0], [0], marker=length_markers[3], color='white',
            markerfacecolor='grey', markersize=8, label=r'$N = 500$'
        ),
        Line2D(
            [0], [0], marker=length_markers[4], color='white',
            markerfacecolor='grey', markersize=8, label=r'$N = 1000$'
        ),
        Line2D(
            [0], [0], marker=length_markers[5], color='white',
            markerfacecolor='grey', markersize=8, label=r'$N = 2100$'
        ),
        Line2D(
            [0], [0], marker=length_markers[6], color='white',
            markerfacecolor='grey', markersize=8, label=r'$N = 3000$'
        )
    ],
    fontsize=7,
    loc='lower right'
)
axes[1].legend(
    handles=[
        Line2D(
            [0], [0], color=fraction_colors[0], linewidth=4,
            label=r'$\mathrm{KF} = 0\%$'
        ),
        Line2D(
            [0], [0], color=fraction_colors[1], linewidth=4,
            label=r'$\mathrm{KF} = 5\%$'
        ),
        Line2D(
            [0], [0], color=fraction_colors[2], linewidth=4,
            label=r'$\mathrm{KF} = 10\%$'
        ),
        Line2D(
            [0], [0], color=sns.color_palette('deep')[3], linewidth=4,
            label='Random coil'
        )
    ],
    fontsize=7
)

# Configure axes
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[0].set_xlabel(r'$R_{\mathrm{g}}$ (nm)')
axes[0].set_ylabel(r'$R_{\mathrm{e}}$ (nm)')
axes[1].set_xlabel('$N$')
axes[1].set_ylabel(r'$R_{\mathrm{g}}$ (nm)')
plt.tight_layout()
plt.savefig('cbmc_Rg_vs_Re.pdf')

# -------------------------------------------------------------------- #
# Parse the 100- and 200-mer random coil configurations and plot: 
# - their energy distributions 
# - their radius of gyration distributions 
# -------------------------------------------------------------------- #
# Do the same exercise with the 100- and 200-mer random coil runs
lengths = [100, 200]
fig, axes = plt.subplots(nrows=2 * len(lengths), ncols=1, figsize=(3, 6))
energies_random_short = {}
radii_random_short = {}
for i, length in enumerate(lengths):
    energies_i = []
    radii_i = []
    for j in range(10):
        # Parse configurations, their energies, and their radii of gyration 
        filename = 'data/cbmc/cbmc_coil_{}mer_{}.txt'.format(length, j)
        print('... parsing: {}'.format(filename))
        _, energies_ij, radii_ij = parse_configurations(filename)
        energies_i.append(energies_ij)
        radii_i.append(radii_ij)

    # The energies and radii of gyration are stored in arrays of size (n1, n2),
    # where n1 is the number of runs (10) and n2 is the number of configurations
    # per run 
    energies_i = np.array(energies_i) / 1000    # Convert to units of 10^3 * kT
    radii_i = np.array(radii_i)
    energies_random_short[length] = energies_i
    radii_random_short[length] = radii_i
    
    # Plot distributions of energy and radius of gyration
    energy_q1 = np.quantile(energies_i, 0.25)
    energy_q3 = np.quantile(energies_i, 0.75)
    iqr = energy_q3 - energy_q1
    min_energy = energy_q1 - 10 * iqr
    max_energy = energy_q3 + 10 * iqr
    is_outlier = ((energies_i < min_energy) | (energies_i > max_energy))
    print('- found {} / {} energetic outliers'.format(is_outlier.sum(), energies_i.size))
    axes[i].hist(energies_i[~is_outlier].reshape(-1), bins=20, density=True)

    # Plot the mean value for each distribution as a vertical line
    mean_energy = np.mean(energies_i[~is_outlier])
    mean_radius = np.mean(radii_i)
    ymin, ymax = axes[i].get_ylim()
    axes[i].plot([mean_energy, mean_energy], [ymin, ymax], linestyle='--')
    axes[i].set_ylim([ymin, ymax])
    print('coil', length, mean_energy, mean_radius)

    # Annotate each plot at the top right with the polymer length
    for k in [0, 2]:
        axes[i + k].annotate(
            '{}-mer'.format(length), (0.98, 0.92),
            xycoords='axes fraction', horizontalalignment='right',
            verticalalignment='top'
        )

# Configure axes labels 
for i in range(2 * len(lengths)):
    axes[i].set_ylabel('Density')
axes[0].set_xlabel(r'Energy ($10^3 k_{\mathrm{B}} T$)')
axes[1].set_xlabel(r'Energy ($10^3 k_{\mathrm{B}} T$)')
axes[2].set_xlabel(r'$R_{\mathrm{g}}$ (nm)')
axes[3].set_xlabel(r'$R_{\mathrm{g}}$ (nm)')
plt.tight_layout()
plt.savefig('cbmc_dists_coil.pdf')

# -------------------------------------------------------------------- #
# Parse the 100- and 200-mer random coil LAMMPS trajectories and plot
# their energy distributions with the CBMC energy distributions 
# -------------------------------------------------------------------- #
fig, axes = plt.subplots(nrows=len(lengths), ncols=1, figsize=(3, 2))

# Plot the energy distributions obtained from CBMC
bins = {}
_, bins[100], _ = axes[0].hist(
    energies_random_short[100].reshape(-1), bins=20, density=True, histtype='step'
)
_, bins[200], _ = axes[1].hist(
    energies_random_short[200].reshape(-1), bins=20, density=True, histtype='step'
)

for i, length in enumerate(lengths):
    energies_i = []
    for j in range(20):
        # Parse configurations 
        filename = 'data/lammps/lammps_coil_{}mer_{}.lammpstrj'.format(length, j)
        print('... parsing: {}'.format(filename))
        coords_ij = parse_lammpstrj(filename)

        # Get the energy of each configuration
        energies_ij = get_energy(coords_ij)
        energies_i.append(energies_ij)

    # The energies are stored in arrays of size (n1, n2), where n1 is the
    # number of runs (10) and n2 is the number of configurations per run 
    energies_i = np.array(energies_i) / 1000    # Convert to units of 10^3 * kT
    
    # Plot distributions of energy and radius of gyration
    energy_q1 = np.quantile(energies_i, 0.25)
    energy_q3 = np.quantile(energies_i, 0.75)
    iqr = energy_q3 - energy_q1
    min_energy = energy_q1 - 10 * iqr
    max_energy = energy_q3 + 10 * iqr
    is_outlier = ((energies_i < min_energy) | (energies_i > max_energy))
    print('- found {} / {} energetic outliers'.format(is_outlier.sum(), energies_i.size))
    print(
        '- found {} / {} energies outside CBMC bin range'.format(
            np.sum((energies_i < bins[length][0]) | (energies_i > bins[length][-1])),
            energies_i.size
        )
    )
    axes[i].hist(
        energies_i[~is_outlier].reshape(-1), bins=bins[length], density=True,
        histtype='step'
    )

    # Plot the mean value for each distribution as a vertical line
    print('coil', length, mean_energy)

    # Annotate each plot at the top right with the polymer length
    axes[i].annotate(
        '{}-mer'.format(length), (0.98, 0.92),
        xycoords='axes fraction', horizontalalignment='right',
        verticalalignment='top'
    )

# Configure axes labels 
for i in range(len(lengths)):
    axes[i].set_ylabel('Density')
axes[1].set_xlabel(r'Energy ($10^3 k_{\mathrm{B}} T$)')
plt.tight_layout()
plt.savefig('cbmc_vs_lammps_dists_coil.pdf')

