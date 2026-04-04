"""
A simple script for running LAMMPS for a single-polymer system, according 
to the preferences enumerated in the given .json file. 

Authors:
    Kee-Myoung Nam

Last updated:
    4/3/2026
"""

import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt

#########################################################################
if __name__ == '__main__':
    # Parse the given .json file
    json_filename = sys.argv[1]
    with open(json_filename) as f:
        params = json.load(f)

    # Parse input arguments
    outprefix = sys.argv[2]
    init_filename = outprefix + '_init.data'
    pdf_filename = outprefix + '_init.pdf'
    rng_seed = int(sys.argv[3])

    # Generate an initial polymer configuration
    subprocess.call(['./bin/generateMelt', json_filename, init_filename, str(rng_seed)])

    # Prepare for the first LAMMPS run with the soft potential ... 
    # 
    # Write the LAMMPS input/output file paths to a file to be parsed by LAMMPS
    rng = np.random.default_rng(rng_seed)
    input_filename = outprefix + '_soft_paths.txt'
    with open(input_filename, 'w') as f:
        # Input file containing initial configurations 
        f.write('{}\n'.format(init_filename))
        # Output files 
        f.write('{}_soft.lammpstrj\n'.format(outprefix))
        f.write('{}_resolved.data\n'.format(outprefix))
        # Timestep and other simulation parameters 
        f.write('{:.10e}\n'.format(params['dt']))
        f.write('{:.10e}\n'.format(params['damp']))
        nsteps = int(params['t_final_soft'] / params['dt'])
        nsteps_per_dump = int(params['dt_per_dump'] / params['dt'])
        f.write('{}\n'.format(nsteps))
        f.write('{}\n'.format(nsteps_per_dump))
        # Random seeds 
        f.write('{}\n'.format(rng.integers(0, 1000)))
        f.write('{}\n'.format(rng.integers(0, 1000)))
        f.write('{}\n'.format(rng.integers(0, 1000)))

    # Run the appropriate LAMMPS script 
    if params['angle_mode'] == 1:    # Gaussian angle potential
        subprocess.run([
            'mpirun', '-np', sys.argv[4],
            'lmp', '-i', 'bimodal_angles_gaussian_soft.lammps', '-v', 'VARS',
            input_filename
        ])
    elif params['angle_mode'] == 0 and params['cosine_K'] > 0:    # Cosine angle potential
        subprocess.run([
            'mpirun', '-np', sys.argv[4],
            'lmp', '-i', 'bimodal_angles_static.lammps', '-v', 'VARS',
            input_filename
        ])
    else:
        subprocess.run([
            'mpirun', '-np', sys.argv[4],
            'lmp', '-i', 'random_coil_soft.lammps', '-v', 'VARS',
            input_filename
        ])

    # Generate a new LAMMPS input file from the output .lammpstrj file
    subprocess.run([
        './bin/lammpstrjToLammpsData', json_filename,
        '{}_soft.lammpstrj'.format(outprefix),
        '{}_resolved_reformatted.data'.format(outprefix)
    ])

    # Prepare for the full LAMMPS run ... 
    # 
    # Write the LAMMPS input/output file paths to a file to be parsed by LAMMPS
    rng = np.random.default_rng(rng_seed)
    input_filename = outprefix + '_paths.txt'
    with open(input_filename, 'w') as f:
        # Input file containing initial configurations 
        f.write('{}_resolved_reformatted.data\n'.format(outprefix))
        # Output files 
        f.write('{}.lammpstrj\n'.format(outprefix))
        f.write('{}.bond\n'.format(outprefix))
        f.write('{}.angle\n'.format(outprefix))
        # Timestep and other simulation parameters 
        f.write('{:.10e}\n'.format(params['dt']))
        f.write('{:.10e}\n'.format(params['damp']))
        nsteps = int(params['t_final'] / params['dt'])
        nsteps_per_dump = int(params['dt_per_dump'] / params['dt'])
        f.write('{}\n'.format(nsteps))
        f.write('{}\n'.format(nsteps_per_dump))
        # Random seeds 
        f.write('{}\n'.format(rng.integers(0, 1000)))
        f.write('{}\n'.format(rng.integers(0, 1000)))
        f.write('{}\n'.format(rng.integers(0, 1000)))

    # Run the appropriate LAMMPS script 
    if params['angle_mode'] == 1:    # Gaussian angle potential
        subprocess.run([
            'mpirun', '-np', sys.argv[4],
            'lmp', '-i', 'bimodal_angles_gaussian.lammps', '-v', 'VARS',
            input_filename
        ])
    elif params['angle_mode'] == 0 and params['cosine_K'] > 0:    # Cosine angle potential
        subprocess.run([
            'mpirun', '-np', sys.argv[4],
            'lmp', '-i', 'bimodal_angles_static.lammps', '-v', 'VARS',
            input_filename
        ])
    else:
        subprocess.run([
            'mpirun', '-np', sys.argv[4],
            'lmp', '-i', 'random_coil.lammps', '-v', 'VARS',
            input_filename
        ])

