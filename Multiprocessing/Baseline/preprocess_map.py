from mpi4py import MPI
import h5py
import sys
from funcs import *
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


start = time.time()

#################################

# import pickle
# file = open('lines', 'rb')
# data_l = pickle.load(file)
# data_lines1 = data_l[0]
# ion_energies1 = data_l[1]
# file.close()

##################################

rank = MPI.COMM_WORLD.Get_rank()
nprocesses = MPI.COMM_WORLD.Get_size()



file_read = sys.argv[1]
folder_write = sys.argv[2]

f = h5py.File(file_read, 'r')
sample = list(f.keys())[0].split('Sample_ID: ')[-1]
total_shots = len(f['Sample_ID: '+sample].keys())

# wavelengths = np.array(hf['System properties']['wavelengths'])
    
# total_shots = int(sys.argv[3])

# print(file_read, folder_write)

# f = h5py.File(file_read, 'r')
# wl = np.array(f['wavelengths'])

shots_per_sample = 1



for I in range(0,int(total_shots/nprocesses)+1):
    spot = int(int(rank*int(total_shots/nprocesses+1)+I))
    
    if spot==total_shots:
        print(spot)
        break

  
    signals =  np.array(f['Sample_ID: '+sample]['Spot_'+str(spot)]['Shot_0']['raw_spectrum'])
    position = np.array(f['Sample_ID: ' + sample]['Spot_' + str(spot)]['position'])
    n_spectrometers = signals.shape[0]
    
    new_a = []
    ns_pre = []
    ns_pro = []
    new_b = []
    baselines = []
    for spec in range(0,n_spectrometers):
        a = signals[spec]
        #print(f,spec,j)
        #nspec_pre.append(np.sum(a))
        #new_a.append(a - baseline_als(a,100000,0.001))
        
        baseline = baseline_als(a,100000,0.001)
        baselines.append(baseline)
        
        spec_baseline_removed = a - baseline
        new_a.append(spec_baseline_removed)
        
        nspec_pre = np.sum(a)
        nspec_pro = np.sum(new_a[-1])
        #maxspec.append(np.max(new_a[-1]))
        ns_pre.append(np.array(nspec_pre))
        ns_pro.append(np.array(nspec_pro))
        
        new_b.append(spec_baseline_removed / nspec_pro)
        
        #maxs.append(np.array(maxspec))
        #new_signals.append(np.array(new_a))
    
    pro_signals = np.array(new_a)
    pro_signals_norm = np.array(new_b)
    baselines = np.array(baselines)

    f1 = h5py.File(folder_write+'pro_sample_'+str(sample)+'_spot_'+str(spot)+'.hdf5', 'w')
    grp1 = f1.create_group('Sample_ID: '+sample)
    grp2 = grp1.create_group('Spot_' + str(spot))
    grp2['position'] = np.array(position, dtype=np.single)
    grp3 = grp2.create_group('Shot_0')
    grp3['raw_spectrum'] = np.array(signals,dtype=np.single)
    grp3['Pro'] = np.array(pro_signals,dtype=np.single)
    grp3['Pro_Norm'] = np.array(pro_signals_norm,dtype=np.single)
    grp3['Baseline'] = np.array(baselines,dtype=np.single)
    if spot == 0:
        f.copy('System properties', f1)
    f1.close()
  
    

f.close()

end = time.time() 

print('Time: ' + str(end-start))