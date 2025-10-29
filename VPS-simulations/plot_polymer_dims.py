"""
Authors:
    Kee-Myoung Nam

Last updated:
    10/28/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from parse_trajectories import PolymerTrajectory

#########################################################################
if __name__ == '__main__':
    fractions = [100, 95, 90, 85, 80, 75, 70]
    n = 10
    n_runs = [0 for _ in range(len(fractions))]

    # Parse each trajectory file ... 
    radii_of_gyration = [[] for _ in range(len(fractions))]
    kuhn_lengths = [[] for _ in range(len(fractions))]
    for i, fraction in enumerate(fractions):
        for j in range(n):
            filename = 'data/static_{}_{}/static_{}_{}.lammpstrj'.format(
                fraction, j, fraction, j
            )
            sim = PolymerTrajectory()
            sim.parse_lammpstrj(filename, 1e-6, max_timestep=None)
            n_runs[i] += 1

            # Get the mean radius of gyration over time
            radii = sim.radii_of_gyration()
            radii_of_gyration[i].append(np.mean(radii))

            # Get the time-averaged Kuhn length
            kuhn_lengths[i].append(sim.kuhn_length())

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    xvalues = [j for i in range(n_runs) for j in range(len(fractions))]
    xticks = [str(fraction) for fraction in fractions]
    y1 = [
        radii_of_gyration[i][j] for i in range(len(fractions))
        for j in range(n_runs[i])
    ]
    y2 = [
        kuhn_lengths[i][j] for i in range(len(fractions))
        for j in range(n_runs[i])
    ]
    axes[0].scatter(xvalues, y1, zorder=0)
    axes[1].scatter(xvalues, y2, zorder=0)
    axes[0].errorbar(
        list(range(len(fractions))),
        [np.mean(radii_of_gyration[i]) for i in range(len(fractions))],
        yerr=[np.std(radii_of_gyrations[i]) for i in range(len(fractions))],
        fmt='_', marker='_', markersize=10, capsize=3, color='black', zorder=1
    )
    axes[1].errorbar(
        list(range(len(fractions))),
        [np.mean(kuhn_lengths[i]) for i in range(len(fractions))],
        yerr=[np.std(kuhn_lengths[i] for i in range(len(fractions))],
        fmt='_', marker='_', markersize=10, capsize=3, color='black', zorder=1
    )
    axes[0].set_xlabel(r'Percentage of $160^\circ$ bond angles')
    axes[0].set_ylabel('Radius of gyration (nm)')
    axes[0].set_xticks(list(range(len(fractions))))
    axes[0].set_xticklabels(xticks)
    axes[1].set_xlabel(r'Percentage of $160^\circ$ bond angles')
    axes[1].set_ylabel('Kuhn length (nm)')
    axes[1].set_xticks(list(range(len(fractions))))
    axes[1].set_xticklabels(xticks)
    plt.tight_layout()
    plt.savefig('polymer_dims.pdf')
    
