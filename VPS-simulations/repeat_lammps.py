"""
A simple script for running LAMMPS for a single-polymer system, according 
to the preferences enumerated in the given .json file. 

Authors:
    Kee-Myoung Nam

Last updated:
    4/5/2026
"""

import sys
import json
import subprocess
import numpy as np

#########################################################################
if __name__ == '__main__':
    # Parse the given .json file
    json_filename = sys.argv[1]
    with open(json_filename) as f:
        params = json.load(f)

    # Generate a new initial configuration file from the last frame of the 
    # given .lammpstrj file
    lammpstrj_filename = sys.argv[2]
    outprefix = sys.argv[3]
    init_filename = outprefix + '_init.data'
    subprocess.call([
        './bin/lammpstrjToLammpsData', json_filename, lammpstrj_filename,
        init_filename
    ])

    # Write the LAMMPS input/output file paths to a file to be parsed by LAMMPS
    rng_seed = int(sys.argv[4])
    rng = np.random.default_rng(rng_seed)
    input_filename = outprefix + '_paths.txt'
    with open(input_filename, 'w') as f:
        # Input file containing initial configurations 
        f.write('{}\n'.format(init_filename))
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
            'mpirun', '-np', sys.argv[5],
            'lmp', '-i', 'bimodal_angles_gaussian.lammps', '-v', 'VARS',
            input_filename
        ])
    elif params['angle_mode'] == 0 and params['cosine_K'] > 0:    # Cosine angle potential
        subprocess.run([
            'mpirun', '-np', sys.argv[5],
            'lmp', '-i', 'bimodal_angles_static.lammps', '-v', 'VARS',
            input_filename
        ])
    else:
        subprocess.run([
            'mpirun', '-np', sys.argv[5],
            'lmp', '-i', 'random_coil.lammps', '-v', 'VARS',
            input_filename
        ])

