"""
Fit von Mises mixture models to coarse-grained bond angles and dihedrals 
from sampled polymer configurations. 

Authors:
    Kee-Myoung Nam

Last updated:
    5/23/2026
"""

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
import matplotlib.pyplot as plt
from utils import parse_configurations
from vonmises_mixture import VonMisesMixture

#########################################################################
fractions = [100, 90, 70]
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6, 2.5))
rng = np.random.default_rng(1234567890)

# For each set of MC runs ...
for i, fraction in enumerate(fractions):
    # Parse configurations from each run and collect bond statistics
    lengths = []
    angles = []
    dihedrals = []
    for j in range(10):
        filename = 'data/cbmc/cbmc_gaussian_{}_2100mer_dihedral_{}.txt'.format(fraction, j)
        print('- Parsing: {}'.format(filename))
        configs, _, _ = parse_configurations(filename)
        lengths_j = []
        angles_j = []
        dihedrals_j = []

        # For each configuration in the run ... 
        for k in range(configs.shape[0]):
            # Get the bond lengths along the configuration 
            lengths_k = np.linalg.norm(configs[k, 1:, :] - configs[k, :-1, :], axis=1)
            for length in lengths_k:
                lengths_j.append(length)

            # Get the bond angles along the configuration
            for m in range(configs.shape[1] - 2):
                u1 = configs[k, m, :] - configs[k, m + 1, :]
                u2 = configs[k, m + 2, :] - configs[k, m + 1, :]
                u1 /= np.linalg.norm(u1)
                u2 /= np.linalg.norm(u2)
                dot = np.dot(u1, u2)
                if dot > 1:
                    angles_j.append(0.0)
                elif dot < -1:
                    angles_j.append(np.pi)
                else:
                    angles_j.append(np.arccos(dot))

            # Get the dihedral angles along the configuration 
            for m in range(configs.shape[1] - 3):
                u1 = configs[k, m + 1, :] - configs[k, m, :]
                u2 = configs[k, m + 2, :] - configs[k, m + 1, :]
                u3 = configs[k, m + 3, :] - configs[k, m + 2, :]
                u1 /= np.linalg.norm(u1)
                u2 /= np.linalg.norm(u2)
                u3 /= np.linalg.norm(u3)
                phi = np.arctan2(
                    np.linalg.norm(u2) * np.dot(u1, np.cross(u2, u3)),
                    np.dot(np.cross(u1, u2), np.cross(u2, u3))
                )
                dihedrals_j.append(phi)
        
        # Collect all statistics from the j-th run
        lengths += lengths_j
        angles += angles_j
        dihedrals += dihedrals_j

    # Fit a von Mises mixture model to the bond angle distribution
    #
    # First unravel the bond angle distribution by negating half the angles
    nsample = 2000
    if i > 0:
        signs = rng.choice([-1, 1], size=len(angles), replace=True)
        angles_ = signs * np.array(angles)
        angle_mixture = VonMisesMixture(n_components=4)
        angle_mixture.fit(
            angles_[rng.choice(len(angles_), nsample, replace=False)], rng,
            init_weights=[0.45, 0.05, 0.05, 0.45],
            init_means=[-160 / 180 * np.pi, -np.pi / 2, np.pi / 2, 160 / 180 * np.pi],
            init_kappa=[5, 1, 1, 5],
            verbose=True
        )
    else:
        signs = rng.choice([-1, 1], size=len(angles), replace=True)
        angles_ = signs * np.array(angles)
        angle_mixture = VonMisesMixture(n_components=2)
        angle_mixture.fit(
            angles_[rng.choice(len(angles_), nsample, replace=False)], rng,
            init_weights=[0.5, 0.5],
            init_means=[-160 / 180 * np.pi, 160 / 180 * np.pi],
            init_kappa=[5, 5],
            verbose=True
        )
    print('- Mixture model statistics for fraction = {}'.format(fraction))
    print(
        '  - Bond angles:', angle_mixture.means, angle_mixture.kappa,
        angle_mixture.weights
    )

    # Define a folded bond angle mixture model 
    angle_means = angle_mixture.means
    angle_kappa = angle_mixture.kappa
    angle_weights = angle_mixture.weights
    if i > 0:
        component_idx = np.argsort(angle_means)
        folded_means = np.array([
            (np.abs(angle_means[component_idx[0]]) + np.abs(angle_means[component_idx[3]])) / 2,
            (np.abs(angle_means[component_idx[1]]) + np.abs(angle_means[component_idx[2]])) / 2
        ])
        folded_kappa = np.array([
            (angle_kappa[component_idx[0]] + angle_kappa[component_idx[3]]) / 2,
            (angle_kappa[component_idx[1]] + angle_kappa[component_idx[2]]) / 2
        ])
        folded_weights = np.array([
            angle_weights[component_idx[0]] + angle_weights[component_idx[3]],
            angle_weights[component_idx[1]] + angle_weights[component_idx[2]]
        ])
        angle_mixture_folded = VonMisesMixture(
            n_components=2, means=folded_means, kappa=folded_kappa,
            weights=folded_weights
        )
    else:
        angle_mixture_folded = VonMisesMixture(
            n_components=1,
            means=np.array([np.mean(np.abs(angle_means))]), 
            kappa=np.array([np.mean(angle_kappa)]),
            weights=np.array([1.0])
        )
    print(
        '  - Folded bond angles:', angle_mixture.means, angle_mixture.kappa,
        angle_mixture.weights
    )
    
    # Fit a von Mises distribution to the dihedral distribution
    dihedrals = np.array(dihedrals)
    dihedral_mixture = VonMisesMixture(n_components=1)
    dihedral_mixture.fit(
        dihedrals[rng.choice(dihedrals.size, nsample, replace=False)], rng, 
        init_weights=[1.0],
        init_means=[np.pi],
        init_kappa=[1],
        verbose=True
    )
    print(
        '  - Dihedrals:', dihedral_mixture.means, dihedral_mixture.kappa,
        dihedral_mixture.weights
    )

    # Plot the bond statistics 
    axes[i, 0].hist(lengths, bins=20, density=True)
    axes[i, 1].hist(angles, bins=np.linspace(0, np.pi, 21), density=True)
    axes[i, 2].hist(dihedrals, bins=np.linspace(-np.pi, np.pi, 21), density=True)

    # Plot the von Mises fits
    angle_mixture_folded.plot(axes[i, 1], folded=True)
    dihedral_mixture.plot(axes[i, 2])

# Configure axes labels 
axes[-1, 0].set_xlabel('Bond length (nm)')
axes[-1, 1].set_xlabel('Bond angle')
axes[-1, 2].set_xlabel('Dihedral angle')
for j in range(3):
    axes[j, 0].set_ylabel('Density')
    axes[j, 1].set_xticks([0, np.pi / 2, np.pi])
    axes[j, 1].set_xticklabels(['0', '90', '180'])
    axes[j, 2].set_xticks([-np.pi, 0, np.pi])
    axes[j, 2].set_xticklabels(['-180', '0', '180'])
length_xmin = min([axes[i, 0].get_xlim()[0] for i in range(3)])
length_xmax = max([axes[i, 0].get_xlim()[1] for i in range(3)])
for j in range(3):
    axes[j, 0].set_xlim([length_xmin, length_xmax])

plt.tight_layout()
plt.savefig('cbmc_gaussian_bond_stats.pdf')
