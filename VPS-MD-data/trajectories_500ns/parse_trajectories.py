"""
A script for fitting mixture models to coarse-grained bond lengths, bond 
angles, and dihedrals from the all-atom MD trajectories. 

Authors:
    Kee-Myoung Nam

Last updated:
    5/25/2026
"""

import sys
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from mdtraj import load_prmtop, load_netcdf
from vonmises_mixture import VonMisesMixture
from utils import (
    find_monomer_coords,
    get_monomer_mass,
    get_bond_stats
)

########################################################################
if __name__ == '__main__':
    rng = np.random.default_rng(1234567890)

    # ---------------------------------------------------------------------- #
    # First fit separate mixture models for each of the three MD trajectories 
    # ---------------------------------------------------------------------- #
    dists_all = []
    angles_all = []
    dihedrals_all = []
    for run in [1, 2, 3]:
        monomer_coords, times = find_monomer_coords(
            'VPS/run{}/CPLX.parm7'.format(run),
            'VPS/run{}/produ_strip.nc'.format(run)
        )
        monomer_mass = get_monomer_mass(
            load_prmtop('VPS/run{}/CPLX.parm7'.format(run))
        )

        # Get bond statistics 
        dists, angles, dihedrals = get_bond_stats(monomer_coords)
        dists_all.append(dists)
        angles_all.append(angles)
        dihedrals_all.append(dihedrals)

        # Make half the angles and dihedrals negative 
        coefs1 = rng.choice([-1, 1], size=angles.shape)
        angles_ = (angles * coefs1).reshape(-1)
        dihedrals_ = dihedrals.reshape(-1)

        # Fit mixture models for each distribution
        dist_mixture = GaussianMixture(n_components=2, covariance_type='full')
        dist_mixture.fit(dists.reshape(-1, 1))
        nsample = 2000
        angle_mixture = VonMisesMixture(n_components=4)
        angle_mixture.fit(
            angles_[rng.choice(len(angles_), nsample, replace=False)], rng,
            init_weights=[0.45, 0.05, 0.05, 0.45],
            init_means=[-160 / 180 * np.pi, -np.pi / 2, np.pi / 2, 160 / 180 * np.pi],
            init_kappa=[5, 1, 1, 5],
            verbose=True
        )
        dihedral_mixture = VonMisesMixture(n_components=1)
        dihedral_mixture.fit(
            dihedrals_[rng.choice(len(dihedrals_), nsample, replace=False)], rng,
            init_weights=[1.0],
            init_means=[np.pi],
            init_kappa=[1],
            verbose=True
        )
        dist_weights = dist_mixture.weights_
        dist_means = dist_mixture.means_[:, 0]
        dist_stds = np.sqrt(dist_mixture.covariances_[:, 0, 0])
        print('- Mixture model statistics from run {}:'.format(run))
        print('  - Bond lengths:', dist_means, dist_stds, dist_weights)
        print(
            '  - Bond angles:', angle_mixture.means, angle_mixture.kappa,
            angle_mixture.weights
        )
        print(
            '  - Dihedrals:', dihedral_mixture.means, dihedral_mixture.kappa,
            dihedral_mixture.weights
        )

        # Get the corresponding probability density for the bond lengths 
        x1 = np.linspace(np.min(dists), np.max(dists), 100)
        dist_pdf = (
            dist_weights[0] * norm.pdf(x1, loc=dist_means[0], scale=dist_stds[0]) +
            dist_weights[1] * norm.pdf(x1, loc=dist_means[1], scale=dist_stds[1])
        )

        # Plot distributions and mixture densities 
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4, 6))
        axes[0].hist(dists.reshape(-1), bins=20, density=True)
        axes[0].plot(x1, dist_pdf)
        axes[1].hist(
            angles_.reshape(-1), bins=np.linspace(-np.pi, np.pi, 21), density=True
        )
        angle_mixture.plot(axes[1])
        axes[2].hist(
            dihedrals_.reshape(-1), bins=np.linspace(-np.pi, np.pi, 21),
            density=True
        )
        dihedral_mixture.plot(axes[2])
        axes[0].set_xlabel('Bond length (nm)')
        axes[0].set_ylabel('Density')
        axes[1].set_xlabel('Bond angle (rad)')
        axes[1].set_ylabel('Density')
        axes[2].set_xlabel('Dihedral angle (rad)')
        axes[2].set_ylabel('Density')
        plt.tight_layout()
        plt.savefig('VPS/run{}/bond_stats.pdf'.format(run))

    # ---------------------------------------------------------------------- #
    # Then fit new mixture models for data pooled from all three trajectories
    # ---------------------------------------------------------------------- #
    # Pool together the bond lengths, angles, and dihedrals from the three runs
    dists_all = np.array(dists_all).reshape(-1)
    angles_all = np.array(angles_all).reshape(-1)
    dihedrals_all = np.array(dihedrals_all).reshape(-1)

    # Make half the angles and dihedrals negative 
    coefs1 = rng.choice([-1, 1], size=angles_all.shape)
    angles_all_ = angles_all * coefs1
    dihedrals_all_ = dihedrals_all

    # Fit mixture models for each distribution
    dist_mixture = GaussianMixture(n_components=2, covariance_type='full')
    dist_mixture.fit(dists_all.reshape(-1, 1))
    nsample = 5000
    angle_mixture = VonMisesMixture(n_components=4)
    angle_mixture.fit(
        angles_all_[rng.choice(len(angles_all_), nsample, replace=False)], rng,
        init_weights=[0.45, 0.05, 0.05, 0.45],
        init_means=[-160 / 180 * np.pi, -np.pi / 2, np.pi / 2, 160 / 180 * np.pi],
        init_kappa=[5, 1, 1, 5],
        verbose=True
    )
    dihedral_mixture = VonMisesMixture(n_components=1)
    dihedral_mixture.fit(
        dihedrals_all_[rng.choice(len(dihedrals_all_), nsample, replace=False)],
        rng,
        init_weights=[1.0],
        init_means=[np.pi],
        init_kappa=[1],
        verbose=True
    )
    dist_weights = dist_mixture.weights_
    dist_means = dist_mixture.means_[:, 0]
    dist_stds = np.sqrt(dist_mixture.covariances_[:, 0, 0])
    print('- Mixture model statistics from pooled data:')
    print('  - Bond lengths:', dist_means, dist_stds, dist_weights)
    print(
        '  - Bond angles:', angle_mixture.means, angle_mixture.kappa,
        angle_mixture.weights
    )
    print(
        '  - Dihedrals:', dihedral_mixture.means, dihedral_mixture.kappa,
        dihedral_mixture.weights
    )

    # Get the corresponding probability density for the bond lengths 
    x1 = np.linspace(np.min(dists_all), np.max(dists_all), 100)
    dist_pdf = (
        dist_weights[0] * norm.pdf(x1, loc=dist_means[0], scale=dist_stds[0]) +
        dist_weights[1] * norm.pdf(x1, loc=dist_means[1], scale=dist_stds[1])
    )

    # Define folded angle mixture
    angle_means = angle_mixture.means
    angle_kappa = angle_mixture.kappa
    angle_weights = angle_mixture.weights
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

    # Plot distributions and mixture densities 
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(4, 4))
    axes[0].tick_params(labelsize=12)
    axes[0].hist(dists_all, bins=20, density=True)
    axes[0].plot(x1, dist_pdf)
    axes[1].tick_params(labelsize=12)
    axes[1].hist(
        (180 / np.pi) * angles_all, bins=np.linspace(0, 180, 21),
        density=True
    )
    angle_mixture_folded.plot(axes[1], folded=True, deg=True)
    axes[1].set_xticks([0, 45, 90, 135, 180])
    axes[2].tick_params(labelsize=12)
    axes[2].hist(
        (180 / np.pi) * dihedrals_all_, bins=np.linspace(-180, 180, 21),
        density=True
    )
    dihedral_mixture.plot(axes[2], deg=True)
    axes[2].set_xticks([-180, -90, 0, 90, 180])
    axes[0].set_xlabel('Bond length (nm)', size=14)
    axes[0].set_ylabel('Density', size=14)
    axes[1].set_xlabel('Bond angle', size=14)
    axes[1].set_ylabel('Density', size=14)
    axes[2].set_xlabel('Dihedral angle', size=14)
    axes[2].set_ylabel('Density', size=14)
    plt.tight_layout()
    plt.savefig('bond_stats_combined.pdf')

