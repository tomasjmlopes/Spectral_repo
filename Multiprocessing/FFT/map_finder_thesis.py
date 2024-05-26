from mpi4py import MPI
import h5py
import sys
from fft_funcs_v2 import *
import time
import numpy as np
import warnings

warnings.filterwarnings('ignore')
start = time.time()

########################################
rank = MPI.COMM_WORLD.Get_rank()
nprocesses = MPI.COMM_WORLD.Get_size()
########################################

########################################
actual_size_big = int(sys.argv[1])
actual_size_small = int(sys.argv[2])
file_read = sys.argv[3]
folder_write = sys.argv[4]
########################################

########################################
f = h5py.File(file_read, 'r')
sample = list(f.keys())[0].split('Sample_ID: ')[-1]
total_shots = len(f['Sample_ID: '+sample].keys())
wavelengths = np.array(f['System properties']['wavelengths']).flatten()
total_wavelengths = int(len(wavelengths))
shots_per_sample = 1
########################################

########################################
wave_i = int(rank*total_wavelengths/nprocesses)
wave_f = int((rank + 1)*total_wavelengths/nprocesses)
spectrums = np.zeros((len(f['Sample_ID: '+ sample]), int(total_wavelengths/nprocesses)))
positions = np.zeros((len(f['Sample_ID: '+ sample]), 2))
########################################

for spot in range(0, len(f['Sample_ID: '+ sample])):
    spectrums[spot, :] = np.array([f['Sample_ID: ' + sample]['Spot_%i' %spot]['Shot_0']['Pro']]).flatten()[wave_i:wave_f]
    positions[spot, :] = np.array([f['Sample_ID: '+sample]['Spot_%i' %spot]['position']])
    
nx, ny = len(np.unique(positions[:, 1])), len(np.unique(positions[:, 0]))
dx, dy = np.unique(positions[:, 1])[1] - np.unique(positions[:, 1])[0], np.unique(positions[:, 0])[1] - np.unique(positions[:, 0])[0]
freqs_x = 2*np.pi*np.fft.fftfreq(nx, dx)
freqs_y = 2*np.pi*np.fft.fftfreq(ny, dy)

index_sorted = np.lexsort((positions[:, 0], positions[:, 1]))
spectrums = spectrums[index_sorted, :].reshape(nx, ny, -1)

indexes_big, indexes_small = generate_mask(spectrums, actual_size_big, actual_size_small, freqs_x, freqs_y, dx)

map_scores = np.array([fft_feature(spectrums[:, :, wv], indexes_big, indexes_small) for wv in range(0, int(total_wavelengths/nprocesses))])

f1 = h5py.File(folder_write + 'pro_sample_' + str(wave_i) + '.hdf5', 'w')
grp1 = f1.create_group('System properties:' + str(wave_i))
grp1.create_dataset('wavelengths', data = np.array(wavelengths[wave_i:wave_f], dtype = np.single))
grp1.create_dataset('fft_score', data = np.array(map_scores, dtype = np.single))
# grp1 = f1.create_group('Sample_ID: '+sample)
# grp2 = grp1.create_group('Spot_' + str(spot))
# grp2['position'] = np.array(position, dtype=np.single)
# grp3 = grp2.create_group('Shot_0')
# grp3['raw_spectrum'] = np.array(signals, dtype=np.single)
# grp3['Pro'] = np.array(pro_signals, dtype=np.single)
# grp3['Pro_Norm'] = np.array(pro_signals_norm, dtype=np.single)
# grp3['Baseline'] = np.array(baselines, dtype=np.single)
# if spot == 0:
#     f.copy('System properties', f1)
f1.close()



# f1 = h5py.File(folder_write + str(sample)+ 'fft_scores' + str(wave_i) +'.hdf5', 'w')
# grp1 = f1.create_group('Sample_ID: ' + sample)
# grp2 = grp1.create_group('Scores')
# grp2['fft'] = np.array(map_scores, dtype=np.float)
# f1.close()
    
f.close()
    
end = time.time()
print('Time: ' + str(end-start))