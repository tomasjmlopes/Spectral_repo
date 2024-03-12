# -*- coding: utf-8 -*-
"""
experiment.py
    Implements a class experiment to store a given experimental dataset in a folder 
    (i.e. grouping multiple shots for the same sample in similar conditions)
"""
    
__author__ = "Nuno Azevedo Silva"
__mantainer__ = "Nuno Azevedo Silva"
__email__ = "nunoazevedosilva@gmail.com"

import numpy as np
from scipy import *
from scipy.interpolate import *
from matplotlib.pyplot import *
from Data_Core.fundamental_constants import *
from Data_Core.scrap_databases import *
from Data_Core.line_shapes import *
from Data_Core.partition_function import *
from Data_Core.element import *
from Data_Core.digital_twin import *
from Data_Core.boltzmann_functions import *
from Data_Core.read_files_functions import *
from Data_Core.signal import *

from sklearn.linear_model import HuberRegressor
from scipy.stats import *

class experiment:
    """
    An object to store a multiple signals from a LIBS experiment
    
    
    
    Attributes:
    
        signal: list of signal_spec objects
            Contains the multiple signals for each shot
        mean_signal: signal_spec object
            Contains the mean signal 
            
    Methods:
    
    """
    
    def __init__(self, folder,ignore=None,normalize=False,SNV=False,Specialtype=None):
        
        if Specialtype==None:

            signals_raw = read_folder(folder,ignore=ignore)
            list_of_signals = []
            spectrums = []
            positions_raw = get_position_signal(folder,ignore=ignore)
        

                
            if normalize:
                for i in range(0,len(signals_raw)):
                    current_signal = signal_spec(folder,"",start_empty = True)
                    #print(folder)
                    #print(current_signal.folder)
                    current_signal.from_data(signals_raw[i],folder,positions_raw[i][0],positions_raw[i][1])
                    current_signal.normalize()
                    list_of_signals.append(current_signal)
                    #print(current_signal)
                    spectrums.append(current_signal.spectrum)
                    
            elif SNV!=False:
                for i in range(0,len(signals_raw)):
                    current_signal = signal_spec(folder,"",start_empty = True)
                    #print(folder)
                    #print(current_signal.folder)
                    current_signal.from_data(signals_raw[i],folder,positions_raw[i][0],positions_raw[i][1])
                    current_signal.SNV(SNV)
                    list_of_signals.append(current_signal)
                    #print(current_signal)
                    spectrums.append(current_signal.spectrum)
            
                    
            else:
                for i in range(0,len(signals_raw)):
                    current_signal = signal_spec(folder,"",start_empty = True)
                    #print(folder)
                    #print(current_signal.folder)
                    #print(positions_raw[i])
                    current_signal.from_data(signals_raw[i],folder,positions_raw[i][0],positions_raw[i][1])
                    list_of_signals.append(current_signal)
                    #print(current_signal)
                    spectrums.append(current_signal.spectrum)
                
        else:
            i=0
            signals_raw = read_folder_sciaps(folder,ignore=ignore)
            list_of_signals = []
            spectrums = []
            current_signal = signal_spec(folder,"",Specialtype=Specialtype)
            list_of_signals.append(current_signal)
            spectrums.append(current_signal.spectrum)
            
        self.list_of_signals = list_of_signals
        mean_spectrum = np.mean(np.array(spectrums),axis=0)
        current_signal = signal_spec(folder,"",start_empty = True)
        current_signal.from_data(signals_raw[i],folder)
        current_signal.spectrum = mean_spectrum
        current_signal.shot_number = len(signals_raw)
        self.mean_signal = current_signal
        
        
    def concatenate(self, list_of_experiments):
        spectrums=[self.mean_signal.spectrum]
        
        for i in range(0,len(list_of_experiments)):
            self.list_of_signals = self.list_of_signals+list_of_experiments[i].list_of_signals
            spectrums.append(list_of_experiments[i].mean_signal.spectrum)
        
        #print(self.list_of_signals)

        mean_spectrum = np.mean(np.array(spectrums),axis=0)
        mean_signal = self.mean_signal
        mean_signal.spectrum = mean_spectrum
        self.mean_signal = mean_signal
      
    def update_mean_signal(self):
        spectrums = np.array([current_signal.spectrum 
                              for current_signal in self.list_of_signals])
        mean_spectrum = np.mean(np.array(spectrums),axis=0)
        mean_signal = self.mean_signal
        mean_signal.spectrum = mean_spectrum
        self.mean_signal = mean_signal
    
        
    def apply_msc_correction(self,n_iter=1):
        # Get the reference spectrum from the mean
        
        for it in range(0,n_iter):
            reference_signal = self.mean_signal.spectrum
            
            # mean centre correction for each signal
            for sig in range(len(self.list_of_signals)):
                for sp in range(len(self.list_of_signals[sig].spectrum)):
                    self.list_of_signals[sig].spectrum[sp] -= self.list_of_signals[sig].spectrum[sp].mean()
            
            #for each spectrum apply the correction
            
            for sig in range(len(self.list_of_signals)):
                for sp in range(len(self.list_of_signals[sig].spectrum)):
                        # Define a new array and populate it with the corrected data    
                        data_msc = np.zeros_like(self.list_of_signals[sig].spectrum[sp])
                        
                        fit = np.polyfit(reference_signal[sp], self.list_of_signals[sig].spectrum[sp], 1, full=True)
                        data_msc = (self.list_of_signals[sig].spectrum[sp] - fit[0][1]) / fit[0][0]
                        
                        #model = HuberRegressor()
                        #model.fit(reference_signal[sp].reshape(-1,1),self.list_of_signals[sig].spectrum[sp])
                        #print(model.coef_)
                        #print(model.intercept_)
                        #print(fit)
                        # Apply correction
                        
                        #Apply correction
                        #data_msc = (self.list_of_signals[sig].spectrum[sp] - model.intercept_) / model.coef_
                        
                        self.list_of_signals[sig].spectrum[sp] = data_msc
        
        self.update_mean_signal()
    
    def clean_outliers(self,zscore_threshold=2,percentage=30):
        
        list_above_threshold=[]
        for sp in range(0,len(self.list_of_signals[0].spectrum)):
            spectrums = np.array([self.list_of_signals[i].spectrum[sp] for i in range(0,len(self.list_of_signals))])
            
            
            for i in range(0,len(self.list_of_signals)):
                if np.where((np.abs(zscore(spectrums, axis=0))>zscore_threshold)[i]==True)[0].size/(spectrums[0].size)*100>percentage:
                    if i not in list_above_threshold:
                        list_above_threshold.append(i)
        
        list_above_threshold.sort(reverse=True)

        for i in range(0,len(list_above_threshold)):
            self.list_of_signals.pop(list_above_threshold[i])

        self.update_mean_signal()
        
        
    def make_map(self, line_to_find, specific_wavelength = None, ratio_of_maximum=0.5, radius=0.3, Plotline=False):
        
        x_values=[]
        y_values=[]
        map_values=[]
        
        if specific_wavelength == None:
        
            ritz = line_to_find.ritz
        
        else:
            
            ritz = specific_wavelength
        
        for i in range(0,len(self.list_of_signals)):
            print(i, ' of ', len(self.list_of_signals),end='\r') 
            current_signal = self.list_of_signals[i]
            
            new_wavelengths = np.concatenate([wl for wl in current_signal.wavelengths])
            new_spectrum = np.concatenate([sp for sp in current_signal.spectrum])
            
            if i == 0 :
                if Plotline:
                    subplots()
                intensity = get_peak_area(ritz, ratio_of_maximum, 
                                          new_wavelengths, new_spectrum, radius=radius, 
                                          Plot=Plotline, Title = "Normalization line")
            
            else:
                
                intensity = get_peak_area(ritz, ratio_of_maximum, 
                                          new_wavelengths, new_spectrum, radius=radius, 
                                          Plot=False, Title = "Normalization line")
                    
            x_values.append(current_signal.position[0])
            y_values.append(current_signal.position[1])
            map_values.append(intensity)
        
        
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        map_values = np.array(map_values)
        
        xy_values = np.stack([x_values,y_values],axis=1)
        unique_x = np.unique(x_values)
        unique_y = np.unique(y_values)
        new_x_values=[]
        new_y_values=[]
        new_map_values=[]
        for i in range(0,len(unique_x)):
            for j in range(0,len(unique_y)):
                current_x = unique_x[i] 
                current_y = unique_y[j]
                temp = (xy_values == [current_x,current_y])[:,0]*(xy_values == [current_x,current_y])[:,1]
                indexes = np.where(temp == True )[0]
                #print(indexes)
                new_x_values.append(current_x)
                new_y_values.append(current_y)
                new_map_values.append(np.mean(map_values[indexes]))
        
        f = interp2d(new_x_values,new_y_values,new_map_values,kind='cubic')
        
        xnew=np.arange(min(x_values),max(x_values),0.5)
        ynew=np.arange(min(y_values),max(y_values),0.5)
        subplots()
        imshow(f(xnew,ynew),cmap='plasma',extent=[min(x_values),max(x_values),min(y_values),max(y_values)])
        colorbar()
        ylabel('y (mm)')
        xlabel('x (mm)')
        title('Intensity of line ' + str(round(ritz,2))+ ' nm')
        
        
        return new_x_values, new_y_values, new_map_values





        
#folder = "D:\\Data_LIBS\\Miguel\\Exp Pulse\\2021-05-31\\Target Fe 99,95% LOT005133\\50 ns\\"
#current_experiment = experiment(folder)
#print(current_experiment.list_of_signals)
#current_experiment.clean_outliers()
#print(current_experiment.list_of_signals)


#folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_23 #1\\340spot3\\"
#current_experiment = experiment(folder,ignore='0000',normalize='True')
#plot(current_experiment.list_of_signals[0].spectrum[2],label="Normalized")

#folder = "D:\\Data_LIBS\\Furo PNR RC 005 - SciAps\\Converted\\DRI0000001\\"
#current_experiment = experiment(folder,Specialtype="SciAPS")
#print(current_experiment.list_of_signals)
#plot(current_experiment.list_of_signals[0].spectrum[0],label="Normalized")

"""

folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_23 #1\\340spot3\\"
current_experiment = experiment(folder,ignore='0000')
plot(current_experiment.list_of_signals[2].spectrum[3],label="")
current_experiment.apply_msc_correction()
plot(current_experiment.list_of_signals[2].spectrum[3],label="MSC")
legend()
subplots()
folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_23 #1\\340spot3\\"
current_experiment = experiment(folder,ignore='0000')
plot(current_experiment.mean_signal.spectrum[2],label="Normal")
current_experiment.apply_msc_correction(n_iter=10)
current_experiment.update_mean_signal()
plot(current_experiment.mean_signal.spectrum[2],label="MSC")
legend()

subplots()
folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_23 #1\\340spot3\\"
current_experiment = experiment(folder,ignore='0000')
plot(current_experiment.mean_signal.spectrum[2],label="Normal")
current_experiment.apply_msc_correction()
plot(current_experiment.mean_signal.spectrum[2],label="MSC")
legend()


subplot(211)
folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_23 #1\\340spot3\\"
current_experiment = experiment(folder,ignore='0000',SNV="Piecewise")
plot(current_experiment.list_of_signals[0].spectrum[2],label="SNV Piecewise")
plot(current_experiment.mean_signal.spectrum[2],label="mean")


folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_23 #1\\340spot3\\"
current_experiment = experiment(folder,ignore='0000',SNV="Full")
plot(current_experiment.list_of_signals[0].spectrum[2],label="Full",ls='--')
legend()

folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_23 #1\\340spot3\\"
current_experiment = experiment(folder,ignore='0000')
subplot(212)
plot(current_experiment.list_of_signals[0].spectrum[2],'r',lw=0.2)
legend()

folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_23 #1\\340spot3\\"
current_experiment = experiment(folder,ignore='0000')

folder2 = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_23 #1\\340spot2\\"
current_experiment2 = experiment(folder2,ignore='0000')

folder2 = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_23 #1\\340spot3\\"
current_experiment3 = experiment(folder2,ignore='0000')


current_experiment.concatenate([current_experiment2,current_experiment3])

sg = current_experiment.mean_signal
density, x,y,p = sg.density_H_alpha()





print(density)
    
al = element('Fe')
sample = digital_twin([['P',2.3e-3],["Cr",2.0e-4],["Mn",6.96e-4],['As',2.13e-3],['Mo',4.89e-6],
                           ['Sn',1.34e-5],['Sb',1.5e-6],['W',4.59e-4],['Fe',6.31e-2]])


ll = 200.
ul = 900.
Tp = 1.*T_ref
new_wavelengths = np.concatenate([wl for wl in sg.wavelengths])
#wl,spec,label, n_ion = al.synthetic_spectrum_NIST(wl = new_wavelengths, lower_limit = ll, upper_limit = ul, electron_temperature=Tp,max_ion_state=3,d_lambda=0.1)
wl,spec,label, n_ion, specs = sample.spectrum_NIST(wl = new_wavelengths, lower_limit = ll, upper_limit = ul, electron_temperature=Tp,max_ion_state=3,d_lambda=0.01, Map=True)
wl1,spec,label, n_ion, specs = sample.spectrum_NIST(lower_limit = ll, upper_limit = ul, electron_temperature=Tp,max_ion_state=3,d_lambda=0.01, Map=True)
wl=wl[0]

Tp = 1.*T_ref

#zone 1
#ll = 371.
#ul = 378.
#ll = 256.
#ul = 300.
nlines = 10
l_num, lines = al.get_most_relevant_lines_ion_state(ion_state = 1, electron_temperature=Tp, n_lines = nlines,lower_limit = ll, upper_limit = ul)

#zone 2
#ll = 256.
#ul = 264.
nlines = 2
l_num2, lines2 = al.get_most_relevant_lines_ion_state(ion_state = 2, electron_temperature=Tp, n_lines = nlines,lower_limit = ll, upper_limit = ul)

#zone 3
#ll = 300.
#ul = 900.
nlines = 2
l_num3, lines3 = al.get_most_relevant_lines_ion_state(ion_state = 3, electron_temperature=Tp, n_lines = nlines,lower_limit = ll, upper_limit = ul)
ion_energies = al.ion_energies


#saha_boltzmann_temperature([lines,lines2,lines3], ion_energies, wl, signal, ratio_of_maximum = 0.5, radius = 0.1, Plot = True, 
#                      Plotlines = False, use_max_intensity = False)

density= density*(10**2)**3

temperature, temp_95, r2, y_s, x_s = sg.saha_boltzmann_temperature_v2([lines,lines2],ion_energies,electron_density= density, 
                                                                      ratio_of_maximum = 0.5, radius = 0.1, Plot = True, Plotlines = True, use_max_intensity = False)



#temperature = .8*T_ref
line_norm=lines[1]

sg.compare_to_digital_sample(sample,electron_temperature = temperature,electron_density= density,
                             d_lambda=0.01, log_scale=False,use_wavelengths=True,line_normalize=line_norm)

"""