"""
Authors:
    Kee-Myoung Nam

Last updated:
    5/25/2026
"""

import sys
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from mdtraj import load_prmtop, load_netcdf
from utils import (
    find_glycosidic_linkages, compute_glycosidic_dihedrals,
    find_monomer_coords, get_bond_stats
)

########################################################################
def plot_glycosidic_dihedrals(linkage_atoms, linkage_type, phi, psi, ax,
                              add_title=False, titlesize=None, labelsize=None,
                              ticklabelsize=None):
    """
    Plot contour maps for the \phi and \psi dihedral angles along the
    specified glycosidic linkage type, given the required atomic coordinates. 

    Parameters
    ----------
    linkage_atoms : list of lists
        Atoms constituting each glycosidic linkage. 
    linkage_type : str
        Glycosidic linkage type. 
    phi : `numpy.ndarray`
        \phi dihedral angles along all glycosidic linkages.
    psi : `numpy.ndarray`
        \psi dihedral angles along all glycosidic linkages.
    ax : `matplotlib.pyplot.Axes`
        Input axes object. 
    add_title : bool
        If True, add a title to the plot.
    titlesize : int
        Title font size. 
    labelsize : int
        Axes label font size. 
    ticklabelsize : int
        Axes tick label font size.

    Returns
    -------
    Updated axes. 
    """
    # Collect all the dihedrals of the desired linkage type
    if linkage_type == 'Gal-GlcA':
        idx = [
            i for i, atoms in enumerate(linkage_atoms)
            if atoms[0].residue.name.startswith('4LA')
            and atoms[3].residue.name.startswith('4GA')
        ]
    elif linkage_type == 'GlcA-GlcB':
        idx = [
            i for i, atoms in enumerate(linkage_atoms)
            if atoms[0].residue.name.startswith('4GA')
            and atoms[3].residue.name.startswith('4GB')
        ]
    elif linkage_type == 'GlcB-Gul':
        idx = [
            i for i, atoms in enumerate(linkage_atoms)
            if atoms[0].residue.name.startswith('4GB')
            and atoms[3].residue.name.startswith('UN')
        ]
    elif linkage_type == 'Gul-Gal':
        idx = [
            i for i, atoms in enumerate(linkage_atoms)
            if atoms[0].residue.name.startswith('UN')
            and atoms[3].residue.name.startswith('4LA')
        ]
    else:
        raise RuntimeError('Undefined glycosidic linkage type')

    # Calculate a Gaussian KDE for the data
    x = phi[:, idx].reshape(-1)
    y = psi[:, idx].reshape(-1)
    kde = gaussian_kde(np.vstack((x, y)))

    # Choose a set of levels to plot for the KDE 
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    xi = np.linspace(xmin, xmax, 200)
    yi = np.linspace(ymin, ymax, 200)
    X, Y = np.meshgrid(xi, yi)
    Z = kde(np.vstack((X.ravel(), Y.ravel()))).reshape(X.shape)
    nlevels = 20
    levels = np.linspace(0, Z.max(), 2 * nlevels + 2)[1::2]
    print(len(levels))

    # Plot the contour map 
    sns.set_theme(style='white')
    cmap = plt.get_cmap('Blues')
    cmap = LinearSegmentedColormap.from_list('', cmap(np.linspace(0.2, 0.9, 256)))
    contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, antialiased=True)
    sns.despine(ax=ax)
    ax.tick_params(bottom=True, left=True, length=3, width=1)

    # Count the number of points within each contour level
    points = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1))) 
    for level, seglist in zip(contour.levels, contour.allsegs):
        print('- Points inside level:', level)
        inside = np.zeros(points.shape[0], dtype=bool)
        for i, seg in enumerate(seglist):
            print('=> Inside path {}:'.format(i), Path(seg).contains_points(points).mean())
            inside |= Path(seg).contains_points(points)
        print('=> All paths:', inside.mean())

    # Make the colorbar continuous 
    #cbar = ax.collections[0].colorbar
    cbar = plt.colorbar(contour, ax=ax, ticks=levels[::2], spacing='proportional')
    vmin, vmax = cbar.vmin, cbar.vmax
    formatter = ScalarFormatter(useMathText=True)
    if vmin > 1e-2:
        formatter.set_powerlimits((-2, -2)) 
    elif vmin > 1e-3:
        formatter.set_powerlimits((-3, -3))
    elif vmin > 1e-4:
        formatter.set_powerlimits((-4, -4))
    else:
        formatter.set_powerlimits((-5, -5))
    cbar.formatter = formatter
    cbar.update_ticks()
    cbar.set_label('Density', size=labelsize)

    # Add plot title
    if add_title:
        if linkage_type == 'Gal-GlcA':
            title = r'D-Glcα-(1 → 4)-D-Galα'
        elif linkage_type == 'GlcA-GlcB':
            title = r'D-Glcβ-(1 → 4)-D-Glcα'
        elif linkage_type == 'GlcB-Gul':
            title = r'L-GulNAc(3Ac,6Gly)α-(1 → 4)-D-Glcβ'
        else:
            title = r'D-Galα-(1 → 4)-L-GulNAc(3Ac,6Gly)α'
        ax.set_title(title, size=titlesize)

    # Configure axes limits
    if ticklabelsize is not None:
        ax.tick_params(labelsize=ticklabelsize)
    ax.set_xlim([0, 360])
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim([0, 360])
    ax.set_yticks([0, 90, 180, 270, 360])

    # Configure axes labels 
    ax.set_xlabel(r'$\phi$', size=labelsize)
    ax.set_ylabel(r'$\psi$', size=labelsize)

    return ax

########################################################################
def plot_glycosidic_dihedrals_vs_cg_bond_angles(linkage_atoms, linkage_type,
                                                dihedrals, cg_bond_angles, ax,
                                                labelsize=None,
                                                ticklabelsize=None):
    """
    Plot a scatter plot comparing the \psi dihedral angles along the specified
    glycosidic linkage type with the coarse-grained bond angles along the VPS
    segment, given the required atomic coordinates and coarse-grained bond
    angles. 

    Parameters
    ----------
    linkage_atoms : list of lists 
        Atoms constituting each glycosidic linkage. 
    linkage_type : str
        Glycosidic linkage type. 
    dihedrals : `numpy.ndarray`
        \psi dihedral angles along all glycosidic linkages.
    cg_bond_angles : `numpy.ndarray`
        Coarse-grained bond angles along the VPS segment. 
    ax : `matplotlib.pyplot.Axes`
        Input axes object. 
    labelsize : int
        Axes label font size. 
    ticklabelsize : int
        Axes tick label font size.

    Returns
    -------
    Updated axes. 
    """
    # Collect all the dihedrals of the desired linkage type
    if linkage_type == 'Gal-GlcA':
        idx = [
            i for i, atoms in enumerate(linkage_atoms)
            if atoms[0].residue.name.startswith('4LA')
            and atoms[3].residue.name.startswith('4GA')
        ]
    elif linkage_type == 'GlcA-GlcB':
        idx = [
            i for i, atoms in enumerate(linkage_atoms)
            if atoms[0].residue.name.startswith('4GA')
            and atoms[3].residue.name.startswith('4GB')
        ]
    elif linkage_type == 'GlcB-Gul':
        idx = [
            i for i, atoms in enumerate(linkage_atoms)
            if atoms[0].residue.name.startswith('4GB')
            and atoms[3].residue.name.startswith('UN')
        ]
    elif linkage_type == 'Gul-Gal':
        idx = [
            i for i, atoms in enumerate(linkage_atoms)
            if atoms[0].residue.name.startswith('UN')
            and atoms[3].residue.name.startswith('4LA')
        ]
    else:
        raise RuntimeError('Undefined glycosidic linkage type')

    # Extract the desired dihedral angles (should be psi)
    dihedrals = dihedrals[:, idx]

    # Remove the dihedral angles within the first and last monomer 
    if linkage_type in ['Gal-GlcA', 'GlcA-GlcB', 'GlcB-Gul']:
        dihedrals = dihedrals[:, 1:-1]
    else:
        dihedrals = dihedrals[:, 1:]

    # Plot the dihedrals and the coarse-grained bond angles 
    ax.scatter(
        dihedrals.reshape(-1), cg_bond_angles.reshape(-1), s=1, rasterized=True
    )

    # Configure axes limits and labels
    if ticklabelsize is not None:
        ax.tick_params(labelsize=ticklabelsize)
    ax.set_xlim([0, 360])
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_ylim([45, 180])
    ax.set_yticks([45, 90, 135, 180])
    if linkage_type == 'Gal-GlcA':
        label = r'D-Glcα-(1 → 4)-D-Galα $\psi$'
    elif linkage_type == 'GlcA-GlcB':
        label = r'D-Glcβ-(1 → 4)-D-Glcα $\psi$'
    elif linkage_type == 'GlcB-Gul':
        label = r'L-GulNAc(3Ac,6Gly)α-(1 → 4)-D-Glcβ $\psi$'
    else:
        label = r'D-Galα-(1 → 4)-L-GulNAc(3Ac,6Gly)α $\psi$'
    ax.set_xlabel(label, size=labelsize)
    ax.set_ylabel('CG bond angle', size=labelsize)

    return ax

########################################################################
if __name__ == '__main__':
    phi = np.zeros((0, 39), dtype=np.float64)
    psi = np.zeros((0, 39), dtype=np.float64)
    cg_bond_angles = np.zeros((0, 8), dtype=np.float64)

    # For each trajectory ...
    for i in [1, 2, 3]:
        # Parse the topology and trajectory files
        traj_filename = 'VPS/run{}/produ_strip.nc'.format(i)
        topology_filename = 'VPS/run{}/CPLX.parm7'.format(i)
        topology = load_prmtop(topology_filename)
        traj = load_netcdf(traj_filename, top=topology)

        # Identify the atoms belonging to each glycosidic linkage 
        linkage_atoms = find_glycosidic_linkages(topology)

        # Compute the dihedral angles along each glycosidic linkage
        phi_i, psi_i = compute_glycosidic_dihedrals(linkage_atoms, traj)
        phi = np.vstack((phi, phi_i))
        psi = np.vstack((psi, psi_i))

        # Identify the monomers
        monomer_coords_i, _ = find_monomer_coords(topology_filename, traj_filename)
        _, cg_bond_angles_i, _ = get_bond_stats(monomer_coords_i)
        cg_bond_angles = np.vstack((cg_bond_angles, cg_bond_angles_i))

    # Convert all angles to degrees 
    phi *= (180 / np.pi)
    psi *= (180 / np.pi)
    cg_bond_angles *= (180 / np.pi)

    # Plot the psi dihedral angles of the GlcA-GlcB linkage for the eight
    # internal monomers vs. the coarse-grained bond angles
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    ax = plot_glycosidic_dihedrals_vs_cg_bond_angles(
        linkage_atoms, 'GlcA-GlcB', psi, cg_bond_angles, ax, labelsize=14,
        ticklabelsize=14
    )
    plt.tight_layout()
    plt.savefig('GlcA-GlcB_psi_vs_cg_bond_angle.pdf', dpi=300)

    # Plot the dihedral angle histogram for the GlcA-GlcB linkage
    fig = plt.figure(figsize=(4.5, 4))
    ax = plt.gca()
    ax = plot_glycosidic_dihedrals(
        linkage_atoms, 'GlcA-GlcB', phi, psi, ax, add_title=True, titlesize=16,
        labelsize=14, ticklabelsize=14
    )
    plt.tight_layout()
    plt.savefig('GlcA-GlcB_phi_psi.pdf')

    # Plot the dihedral angle histograms
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 8))
    plot_glycosidic_dihedrals(
        linkage_atoms, 'Gal-GlcA', phi, psi, axes[0, 0], add_title=True,
        titlesize=14, labelsize=14, ticklabelsize=14
    )
    plot_glycosidic_dihedrals(
        linkage_atoms, 'GlcA-GlcB', phi, psi, axes[0, 1], add_title=True,
        titlesize=14, labelsize=14, ticklabelsize=14
    )
    plot_glycosidic_dihedrals(
        linkage_atoms, 'GlcB-Gul', phi, psi, axes[1, 0], add_title=True,
        titlesize=10, labelsize=14, ticklabelsize=14
    )
    plot_glycosidic_dihedrals(
        linkage_atoms, 'Gul-Gal', phi, psi, axes[1, 1], add_title=True,
        titlesize=10, labelsize=14, ticklabelsize=14
    )
    for i in range(2):
        for j in range(2):
            axes[i, j].set_xlabel(r'$\phi$')
            axes[i, j].set_ylabel(r'$\psi$')
    plt.tight_layout()
    plt.savefig('glycosidic_linkages_phi_psi.pdf')

