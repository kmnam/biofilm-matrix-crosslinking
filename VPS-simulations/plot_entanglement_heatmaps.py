"""
Plot entanglement statistics across different concentrations, kink fractions,
and dihedral stiffnesses.

Authors:
    Kee-Myoung Nam

Last updated:
    5/28/2026
"""
import os
import re
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
import pandas as pd
pd.set_option('display.max_colwidth', None)
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

######################################################################
if __name__ == '__main__':
    # Compile the various MC simulations 
    prefixes = [
        'gaussian_100_vol4',
        'gaussian_95_vol4',
        'gaussian_90_vol4',
        'gaussian_85_vol4',
        'gaussian_80_vol4',
        'coil_vol4',
        'gaussian_100_2fold_vol2',
        'gaussian_95_2fold_vol2',
        'gaussian_90_2fold_vol2',
        'gaussian_85_2fold_vol2',
        'gaussian_80_2fold_vol2',
        'coil_2fold_vol2',
        'gaussian_100_3fold',
        'gaussian_95_3fold',
        'gaussian_90_3fold',
        'gaussian_85_3fold',
        'gaussian_80_3fold',
        'coil_3fold',
        'gaussian_100_5fold',
        'gaussian_95_5fold',
        'gaussian_90_5fold',
        'gaussian_85_5fold',
        'gaussian_80_5fold',
        'coil_5fold'
    ]
    for Kd in [1, 2, 10]:
        prefixes += [
            'gaussian_100_vol4_dihedral_K{}'.format(Kd),
            'gaussian_95_vol4_dihedral_K{}'.format(Kd),
            'gaussian_90_vol4_dihedral_K{}'.format(Kd),
            'gaussian_85_vol4_dihedral_K{}'.format(Kd),
            'gaussian_80_vol4_dihedral_K{}'.format(Kd),
            'gaussian_100_2fold_vol2_dihedral_K{}'.format(Kd),
            'gaussian_95_2fold_vol2_dihedral_K{}'.format(Kd),
            'gaussian_90_2fold_vol2_dihedral_K{}'.format(Kd),
            'gaussian_85_2fold_vol2_dihedral_K{}'.format(Kd),
            'gaussian_80_2fold_vol2_dihedral_K{}'.format(Kd),
            'gaussian_100_3fold_dihedral_K{}'.format(Kd),
            'gaussian_95_3fold_dihedral_K{}'.format(Kd),
            'gaussian_90_3fold_dihedral_K{}'.format(Kd),
            'gaussian_85_3fold_dihedral_K{}'.format(Kd),
            'gaussian_80_3fold_dihedral_K{}'.format(Kd),
            'gaussian_100_5fold_dihedral_K{}'.format(Kd),
            'gaussian_95_5fold_dihedral_K{}'.format(Kd),
            'gaussian_90_5fold_dihedral_K{}'.format(Kd),
            'gaussian_85_5fold_dihedral_K{}'.format(Kd),
            'gaussian_80_5fold_dihedral_K{}'.format(Kd)
        ]

    # For each group of simulations, parse/calculate:
    # 1) Number of entanglements (Z)
    # 2) Primitive path length (Lpp)
    # 3) Normalized primitive path length (Lpp/Re)
    # 4) S-kink estimator (NeSK)
    chunks = []
    for prefix in prefixes:
        for i in range(10):
            data_ = pd.read_csv(
                'data/cbmc/cbmc_melt_{}_{}_Z1+summary.dat'.format(prefix, i),
                sep=' ', header=None, index_col=None, usecols=[3, 4, 5, 10],
                na_values='************'
            )
            data_.columns = ['Re', 'Lpp', 'Z', 'NeSK']
            data_['Lpp_norm'] = data_['Lpp'] / data_['Re']
            data_['prefix'] = prefix
            if 'fold' not in prefix:
                data_['fold'] = 1.0
            else:
                data_['fold'] = float(re.search(r'([0-9])fold', prefix).group(1))
            chunks.append(data_[['prefix', 'Z', 'Lpp', 'Lpp_norm', 'NeSK', 'fold']])
    data = pd.concat(chunks, ignore_index=True)

    # Plot the variables, for each concentration, kink fraction, and dihedral
    # stiffness, as heatmaps 
    #
    # Each heatmap is 4x5, spanning the dihedral stiffnesses and kink
    # fractions, plus a special cell for random coils 
    heatmap_ratio = 4. / 6.
    nrows = 4
    ncols = 4
    fig = plt.figure(
        figsize=(10, 10 * heatmap_ratio * nrows / ncols), constrained_layout=True
    )
    gs = fig.add_gridspec(
        nrows=nrows, ncols=ncols + 1, width_ratios=([6] * ncols + [0.35]),
        height_ratios=[4, 4, 4, 4], hspace=0.02, wspace=0.05
    )
    axes = np.array([
        [fig.add_subplot(gs[i, j]) for j in range(ncols)] for i in range(nrows)
    ])

    # Add colorbar axes 
    cbar_axes = [fig.add_subplot(gs[i, ncols]) for i in range(nrows)]

    # Compile heatmaps 
    heatmaps = [np.zeros((4, 6)) for _ in range(nrows * ncols)]
    for i, conc_prefix in enumerate(['vol4', '2fold_vol2', '3fold', '5fold']):
        for j, fraction in enumerate([100, 95, 90, 85, 80]):
            for k, stiffness in enumerate([0.5, 1, 2, 10]):
                prefix = 'gaussian_{}_{}'.format(fraction, conc_prefix)
                if stiffness >= 1:
                    prefix += '_dihedral_K{}'.format(stiffness)
                subdata = data.loc[data['prefix'] == prefix]
                heatmaps[i][k, j] = subdata['Z'].mean()
                heatmaps[ncols + i][k, j] = subdata['Lpp'].mean()
                heatmaps[2 * ncols + i][k, j] = subdata['Lpp_norm'].mean()
                heatmaps[3 * ncols + i][k, j] = subdata['NeSK'].mean()

        # Parse data for the random coil simulations 
        subdata = data.loc[data['prefix'] == 'coil_{}'.format(conc_prefix)]
        heatmaps[i][0, 5] = subdata['Z'].mean()
        heatmaps[ncols + i][0, 5] = subdata['Lpp'].mean()
        heatmaps[2 * ncols + i][0, 5] = subdata['Lpp_norm'].mean()
        heatmaps[3 * ncols + i][0, 5] = subdata['NeSK'].mean()

        # Fill in the rest of the heatmap with NaN's
        heatmaps[i][1:, 5] = np.nan
        heatmaps[ncols + i][1:, 5] = np.nan
        heatmaps[2 * ncols + i][1:, 5] = np.nan
        heatmaps[3 * ncols + i][1:, 5] = np.nan

    # Plot the heatmaps 
    for i in range(nrows):
        for j in range(ncols):
            idx = ncols * i + j
            mask = np.isnan(heatmaps[idx])
            vmin = min(np.nanmin(heatmaps[ncols * i + k]) for k in range(3))
            vmax = max(np.nanmax(heatmaps[ncols * i + k]) for k in range(3))
            annotations = [[None for _ in range(6)] for _ in range(4)]
            for k in range(4):
                for m in range(6):
                    if heatmaps[idx][k, m] >= 1:
                        annotations[k][m] = '{:.3g}'.format(heatmaps[idx][k, m])
                    else:
                        annotations[k][m] = '{:.2g}'.format(heatmaps[idx][k, m])
            sns.heatmap(
                heatmaps[idx], ax=axes[i, j], mask=mask, annot=annotations,
                fmt='', cmap='Blues', vmin=vmin, vmax=vmax, cbar=(j == 2),
                cbar_ax=(cbar_axes[i] if j == 2 else None),
            )

    # Add a rectangle around the random coil cells
    outline_color = sns.color_palette()[3]
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].add_patch(
                Rectangle(
                    (5, 0), 1, 1, fill=False, edgecolor=outline_color,
                    linewidth=2, clip_on=False
                )
            )

    # Configure axes labels 
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].set_xticks([x + 0.5 for x in range(5)])
            axes[i, j].set_xticklabels(['0', '5', '10', '15', '20'])
            axes[i, j].set_yticks([x + 0.5 for x in range(4)])
            axes[i, j].set_yticklabels(['0.5', '1', '2', '10'])
            axes[i, j].invert_yaxis()
            axes[i, j].set_xlabel('Kink fraction (%)', size=10)
            axes[i, j].set_ylabel(r'$\kappa$', size=10)
            axes[i, j].xaxis.set_label_coords(5. / 12., -0.24)

    # Add axes titles
    for j in range(ncols):
        if j == 0:
            axes[0, j].set_title(r'$c = c_0$')
        elif j == 1 or j == 2:
            axes[0, j].set_title(r'$c = {}c_0$'.format(j + 1))
        else:
            axes[0, j].set_title(r'$c = 5c_0$')

    cbar_axes[0].set_ylabel(r'$\langle Z \rangle$', size=12)
    cbar_axes[1].set_ylabel(r'$\langle L_{\mathrm{pp}} \rangle$ (nm)', size=12)
    cbar_axes[2].set_ylabel(r'$\langle L_{\mathrm{pp}} \rangle / R_{\mathrm{e}}$', size=12)
    cbar_axes[3].set_ylabel(r'$\langle N_{\mathrm{e}}^{\mathrm{SK}} \rangle$', size=12)
    plt.savefig('cbmc_Z_Lpp_heatmap.pdf')

    #####################################################################
    # Calculate and plot the M-coil and M-kink estimators ...  
    prefixes = [
        'gaussian_100_vol4',
        'gaussian_95_vol4',
        'gaussian_90_vol4',
        'gaussian_85_vol4',
        'gaussian_80_vol4',
        'gaussian_100_2fold_vol2',
        'gaussian_95_2fold_vol2',
        'gaussian_90_2fold_vol2',
        'gaussian_85_2fold_vol2',
        'gaussian_80_2fold_vol2',
        'gaussian_100_3fold',
        'gaussian_95_3fold',
        'gaussian_90_3fold',
        'gaussian_85_3fold',
        'gaussian_80_3fold', 
        'gaussian_100_5fold',
        'gaussian_95_5fold',
        'gaussian_90_5fold',
        'gaussian_85_5fold',
        'gaussian_80_5fold'
    ]
    lengths = [200, 150, 100, 50]
    chunks = []
    for prefix in prefixes:
        for length in lengths:
            if length == 200:
                full_prefix = 'data/cbmc/cbmc_melt_{}'.format(prefix)
            else:
                full_prefix = 'data/cbmc/cbmc_melt_{}_len{}'.format(prefix, length)
            # Parse the Z1+ data from each run ...
            for i in range(10):
                filename = '{}_{}_Z1+summary.dat'.format(full_prefix, i)
                if os.path.isfile(filename):
                    data_ = pd.read_csv(
                        filename, sep=' ', header=None, index_col=None,
                        usecols=[3, 4, 5], na_values='************'
                    )
                    data_.columns = ['Re', 'Lpp', 'Z']
                    if 'fold' not in prefix:
                        data_['fold'] = 1.0
                    else:
                        data_['fold'] = float(re.search(r'([0-9])fold', prefix).group(1))
                    data_['kink'] = float(re.search(r'gaussian_([0-9]+)', prefix).group(1))
                    data_['N'] = length
                    data_['contour'] = (length - 1) * 1.8
                    data_['run'] = i
                    data_['filename'] = filename
                    chunks.append(
                        data_[[
                            'Re', 'Lpp', 'Z', 'fold', 'kink', 'N', 'contour',
                            'run', 'filename'
                        ]]
                    )
    data = pd.concat(chunks, ignore_index=True)

    # Calculate the right-hand quantity in the M-coil estimator equation  
    data['rhs'] = (data['Lpp'] ** 2) / (data['contour'] * 1.8)

    # Fit a linear regression for this right-hand quantity for each fold and
    # kink fraction
    Ne_Mcoil = {}
    combinations = []
    rng = np.random.default_rng(1234567890)
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(8, 8))
    for i, fold in enumerate([1, 2, 3, 5]):
        for j, kink in enumerate([100, 95, 90, 85, 80]):
            subdata = data.loc[((data['kink'] == kink) & (data['fold'] == fold))]
            if subdata.shape[0] > 0:
                combinations.append((fold, kink))

                # Calculate the right-hand derivative in the M-coil estimator 
                x = subdata['N'].to_numpy(dtype=np.float64)
                y = subdata['rhs'].to_numpy(dtype=np.float64)
                rhs_fit = linregress(x, y)
                slope = rhs_fit.slope
                axes[i, j].scatter(x, y, rasterized=True)

                # Fit the left-hand quantity to (c / (N - 1)) + d
                z = ((subdata['Re'] / subdata['contour']) ** 2).to_numpy(dtype=np.float64)
                lhs_fit = curve_fit(lambda x_, c, d: c / (x - 1) + d, x, z)
                c, d = lhs_fit[0]

                # Solve for the value of N for which the left-hand side matches
                # the slope
                Ne = 1 + c / (slope - d)
                Ne_Mcoil[(fold, kink)] = [Ne]
                print(
                    'M-coil fit for {}, {}: slope = {}, R^2 = {}, Ne = {}'.format(
                        fold, kink, slope, rhs_fit.rvalue, Ne
                    )
                )
    
    for i in range(4):
        axes[i, 0].set_ylabel('M-coil RHS')
    for j in range(5):
        axes[-1, j].set_xlabel(r'$N$')
    fig.tight_layout()
    fig.savefig('cbmc_Mcoil_fits.pdf', dpi=300)

    # Fit a linear regression for N vs. Z 
    Ne_Mkink = {}
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(8, 8))
    for fold, kink in combinations:
        subdata = data.loc[((data['kink'] == kink) & (data['fold'] == fold))]
        x = subdata['Z'].to_numpy(dtype=np.float64)
        y = subdata['N'].to_numpy(dtype=np.float64)
        NZ_fit = linregress(x, y)
        slope = NZ_fit.slope
        Ne_Mkink[(fold, kink)] = [slope]
        i = [1, 2, 3, 5].index(fold)
        j = [100, 95, 90, 85, 80].index(kink)
        axes[i, j].scatter(x, y, rasterized=True)
        print(
            'M-kink fit for {}, {}: slope = Ne = {}, R^2 = {}'.format(
                fold, kink, slope, NZ_fit.rvalue
            )
        )
        
    for i in range(4):
        axes[i, 0].set_ylabel(r'$N$')
    for j in range(5):
        axes[-1, j].set_xlabel(r'$Z$')
    fig.tight_layout()
    fig.savefig('cbmc_Mkink_fits.pdf', dpi=300)

    # Plot M-coil and M-coil on the same plot
    fig = plt.figure(figsize=(7, 1.5))
    ax = plt.gca()
    ax.errorbar(
        np.arange(len(combinations)), 
        [Ne_Mcoil[combination][0] for combination in combinations], 
        yerr=np.zeros((2, len(combinations))),
        fmt='_',
        linestyle='none',
        capsize=5,
        elinewidth=1,
        ecolor='black',
        capthick=1,
        markersize=10,
        markerfacecolor=sns.color_palette()[0]
    )
    ax.errorbar(
        np.arange(len(combinations)), 
        [Ne_Mkink[combination][0] for combination in combinations], 
        yerr=np.zeros((2, len(combinations))),
        fmt='_',
        linestyle='none',
        capsize=5,
        elinewidth=1,
        ecolor='black',
        capthick=1,
        markersize=10,
        markerfacecolor=sns.color_palette()[1]
    )

    # Configure axes labels for the last plot
    groups = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15)), list(range(15, 20))]
    ax.set_xlabel('')
    ax.set_xticks(list(range(len(combinations))))
    ax.set_xticklabels(
        [
            '0%', '5%', '10%', '15%', '20%',
            '0%', '5%', '10%', '15%', '20%',
            '0%', '5%', '10%', '15%', '20%',
            '0%', '5%', '10%', '15%', '20%'
        ],
        size=9
    )
    sec = ax.secondary_xaxis(location=0)
    sec.tick_params(axis='x', length=0, pad=1.8)
    sec.set_xticks([(group[0] + group[-1]) / 2 for group in groups])
    sec.set_xticklabels([
        '\n\n' + r'$c = c_0$',
        '\n\n' + r'$c = 2c_0$',
        '\n\n' + r'$c = 3c_0$',
        '\n\n' + r'$c = 5c_0$'
    ])
    for group in groups:
        imin, imax = group[0], group[-1]
        ax.plot(
            [imin - 0.4, imax + 0.4], [-0.35, -0.35], color='grey',
            transform=ax.get_xaxis_transform(), lw=1.2, clip_on=False
        )
    
    # Fix x-axis limits 
    ax.set_xlim([-0.8, len(combinations) - 1 + 0.8])

    # Set y-axis label
    ax.text(
        -0.08, 0.48, r'$N_{\mathrm{e}}^{\mathrm{MC}}$',
        color=sns.color_palette()[0],
        rotation=90,
        verticalalignment='top', horizontalalignment='center',
        transform=ax.transAxes
    )
    ax.text(
        -0.08, 0.49, ',',
        rotation=90,
        verticalalignment='center', horizontalalignment='center',
        transform=ax.transAxes
    )
    ax.text(
        -0.08, 0.56, r'$N_{\mathrm{e}}^{\mathrm{MK}}$',
        color=sns.color_palette()[1],
        rotation=90,
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes
    )
    plt.tight_layout()
    plt.savefig('cbmc_Ne_updated.pdf')

