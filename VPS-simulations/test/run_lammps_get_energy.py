import sys
import subprocess

script_filename = 'get_energy_{}.lammps'.format(sys.argv[1])
config_filename = 'configs/test_10mer_{}.txt'.format(sys.argv[1])
log_filename = 'configs/test_10mer_{}_lammps.log'.format(sys.argv[1])
output_filename = 'configs/test_10mer_{}_energy.txt'.format(sys.argv[1])
with open(log_filename, 'w') as f:
    subprocess.run(
        [
            'mpirun', '-np', '1', 'lmp', '-i', script_filename, '-v', 'VARS',
            config_filename
        ],
        stdout=f
    )

with open(log_filename) as f1, open(output_filename, 'w') as f2:
    for line in f1:
        if line.strip().startswith('Step'):
            break
    data = f1.readline().strip().split()
    f2.write(' '.join(data[-4:]) + '\n')
