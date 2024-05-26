# -*- coding: utf-8 -*-
"""
signal.py
    Implements a class signal to store 
"""
    
__author__ = "Nuno Azevedo Silva"
__mantainer__ = "Nuno Azevedo Silva"
__email__ = "nunoazevedosilva@gmail.com"

import numpy as np
import copy
from scipy import *
from matplotlib.pyplot import *
import matplotlib.ticker as ticker
from Data_Core.fundamental_constants import *
from Data_Core.scrap_databases import *
from Data_Core.line_shapes import *
from Data_Core.partition_function import *
from Data_Core.element import *
from Data_Core.digital_twin import *
from Data_Core.boltzmann_functions import *
from Data_Core.read_files_functions import *
from Data_Core.voigt_fit import *
from Data_Core.baseline import *
import itertools
from scipy import signal



class signal_spec:
    """
    An object to store a signal from a LIBS experiment
    
    
    
    Attributes:
    
        spectrum: list of 1d arrays 
            Contains the spectrums for each spectrometer
        wavelengths: list of 1d arrays
            Contains the wavelengths for which the spectrum are obtained
        dark: list of 1d arrays 
            Contains the dark channel information
        reference: list of 1d arrays 
            Contains the reference channel information
        is_multi_channel: bool
            True if it is multi_channel
        spectrometer_labels: list of strings
            Contains the spectrometer names
    
    Methods:
    
    """
    
    def __init__(self, folder, filename, start_empty = False,Specialtype=None):
        
        """
        Given a filename in folder creates an object of class signal
        """
        
        if start_empty:
            
            self.wavelengths = 0
            self.spectrum = 0
            self.dark = 0
            self.reference = 0
            self.is_multi_channel = 0
            self.spectrometer_labels = 0 
            self.shot_number = 0
            self.integration_time = 0
            self.folder = folder
            self.position = [0,0]
            
        else:
            
            if Specialtype!=None:
                
                wavelenghts, spectrum, dark, reference, is_multi_channel, spectrometer_labels, shot_number, integration_time = read_signal_from_file_sciaps(folder, filename)
                self.wavelengths = wavelenghts
                self.spectrum = spectrum
                self.dark = dark
                self.reference = reference
                self.is_multi_channel = is_multi_channel
                self.spectrometer_labels = spectrometer_labels
                self.shot_number = shot_number
                self.integration_time = integration_time
                self.folder = folder
                
                self.position = [0,0]
                
            else:
                wavelenghts, spectrum, dark, reference, is_multi_channel, spectrometer_labels, shot_number, integration_time = read_signal_from_file(folder, filename)
                
                self.wavelengths = wavelenghts
                self.spectrum = spectrum
                self.dark = dark
                self.reference = reference
                self.is_multi_channel = is_multi_channel
                self.spectrometer_labels = spectrometer_labels
                self.shot_number = shot_number
                self.integration_time = integration_time
                self.folder = folder
                
                self.position = get_position_signal(folder, filename)
                
                
                
            
    def from_data(self, read_data, folder,x_pos=0,y_pos=0):
        
        self.wavelengths = read_data[0]
        self.spectrum = read_data[1]
        self.dark = read_data[2]
        self.reference = read_data[3]
        self.is_multi_channel = read_data[4]
        self.spectrometer_labels = read_data[5]
        self.shot_number = read_data[6]
        self.integration_time = read_data[7]
        self.folder = folder
        self.position = get_position_signal = [x_pos,y_pos]
        
    
    def plot_spec(self, lower_limit_wavelength = None, upper_limit_wavelength = None, normalize = False):
        
        """
        Plots the spectrum for a given signal
        """
        
        if lower_limit_wavelength == None: 
            lower_limit_wavelength = self.find_wavelength_limits()[0]
            
        if upper_limit_wavelength == None: 
            upper_limit_wavelength = self.find_wavelength_limits()[1]
            
        ax = gca()
        
        for i in range(0, len(self.spectrum)):
            spec=self.spectrum[i]
            if normalize:
                spec=spec/self.find_intensity_max()
            wl=self.wavelengths[i]
            
            ax.plot(wl,spec,ls='-', label = "Channel "+str(i))
            
        
        ax.set_xlim(lower_limit_wavelength,upper_limit_wavelength)
        ax.legend()
        
    def plot_spec_part(self, part , lower_limit_wavelength = None, upper_limit_wavelength = None, normalize = False):
        
        """
        Plots the spectrum for a given signal
        """
        
        if lower_limit_wavelength == None: 
            lower_limit_wavelength = self.find_wavelength_limits(part)[0]
            
        if upper_limit_wavelength == None: 
            upper_limit_wavelength = self.find_wavelength_limits(part)[1]
            
        ax = gca()
        
        
        spec=self.spectrum[part]
        if normalize:
            spec=spec/self.find_intensity_max()
        wl=self.wavelengths[part]
            
        ax.plot(wl,spec,ls='-', label = "Channel "+str(part))
            
        
        ax.set_xlim(lower_limit_wavelength,upper_limit_wavelength)
        ax.legend()
        
        

    
    def find_wavelength_limits(self,subset=None):
        
        """ 
        Method to get the limits of the wavelength range
        
        Returns:
            [max,min]: list containing the maximum and minimum wavelengths  
        """
        
        current_min = 9000
        current_max = 0
        if subset==None:
            for i in range(0,len(self.wavelengths)):
                if self.wavelengths[i].max()>current_max:
                    current_max = self.wavelengths[i].max()
                if self.wavelengths[i].min()<current_min:
                    current_min = self.wavelengths[i].min()
        else:
            wl=array(self.wavelengths)[subset]
            for i in range(0,len(wl)):
                if wl[i].max()>current_max:
                    current_max = wl[i].max()
                if wl[i].min()<current_min:
                    current_min = wl[i].min()
        return [current_min,current_max]
    
    
    
    
    def find_intensity_max(self):
        """
        Method to find the maximum intensity registered on a given signal
        
        Returns:
            maximum_intensity registered in the spectrum set
        """
        
        current_max = 0
        for i in range(0,len(self.wavelengths)):
            if self.spectrum[i].max()>current_max:
                current_max = self.spectrum[i].max()
        return current_max
    
    def downsample(self, step):
        """
        Method to downsample a given signal using a moving window method
        
        Returns:
            new_signal: object signal
                A copy of the original signal object with subsequent downsample operation
        """
        
        new_signal = deepcopy(self)
        step=float(step)
        
        for i in range(0,len(self.spectrum)):
            new_spectrum = []
            new_dark = []
            new_reference = []
            new_wavelengths = []
            for j in range(0,len(self.spectrum[i])):
                try:
                    new_signal.append(spectrum[i:i+step].sum()/step)
                    new_dark.append(spectrum[i:i+step].sum()/step)
                    new_reference.append(spectrum[i:i+step].sum()/step)
                    new_wavelengths.append(spectrum[i:i+step].sum()/step)
                except:
                    None
                    
        return new_signal


        
    def __repr__(self):
        
        return 'Signal for shot ' + str(self.shot_number) + '\n'+ 'at folder \n' +str(self.folder) + '\n with ' + str(len(self.spectrometer_labels)) +  ' spectrometers'
    
            
    def __str__(self):
        
        return 'Signal for shot ' + str(self.shot_number) + '\n'+ 'at folder \n' +str(self.folder) + '\n with ' + str(len(self.spectrometer_labels)) +  ' spectrometers'
    
    
    def saha_boltzmann_temperature(self, lines, ion_energies , ratio_of_maximum = 0.5, radius = 1, electron_density=n_e_ref,  
                          Plot = True, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = False):
        """
        
        Finds the plasma temperature using the Saha-Boltzmann plot method.

        Parameters
        ----------
        lines : list of lines
            Target lines.
        ion_energies : list of floats 
            List with the referent ion energies. 
        ratio_of_maximum : float, optional
            To compute the intersection. The default is 0.5.
        radius : float,optional
            range of wavelengths for finding the maximum value around the wavelength. The default is 1.
        Plot : Bool, optional
            If you want to obtain the plot. The default is True.
        Title : String, optional
            Title for the graph. The default is "Boltzmann plot".
        Plotlines : Bool, optional
            If you want to plot the corresponding interval for each line. The default is False.
        use_max_intensity : Bool, optional
            If instead of integrated intensity you want to use the maximum intensity. The default is False.

        Returns
        -------
        temperature : float
        temp_95 : float
            confidence interval at 95 percent
        r2 : float
            r-squared for the linear regression
        y_s : array
            y-array used in the plot
        x_s : array
            x-array used in the plot

        """
        new_wavelengths = np.concatenate([wl for wl in self.wavelengths])
        new_spectrum = np.concatenate([sp for sp in self.spectrum])
        temperature, temp_95, r2, y_s, x_s = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, electron_density=electron_density, 
                          Plot = Plot, Title = "Boltzmann plot", Plotlines = Plotlines, use_max_intensity = use_max_intensity)
        
        return temperature, temp_95, r2, y_s, x_s
    
    def saha_boltzmann_temperature_v2(self, lines, ion_energies , ratio_of_maximum = 0.5, radius = 1, electron_density=n_e_ref,
                          Plot = True, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = False):
        """
        
        Finds the plasma temperature using the Saha-Boltzmann plot method.

        Parameters
        ----------
        lines : list of lines
            Target lines.
        ion_energies : list of floats 
            List with the referent ion energies. 
        ratio_of_maximum : float, optional
            To compute the intersection. The default is 0.5.
        radius : float,optional
            range of wavelengths for finding the maximum value around the wavelength. The default is 1.
        Plot : Bool, optional
            If you want to obtain the plot. The default is True.
        Title : String, optional
            Title for the graph. The default is "Boltzmann plot".
        Plotlines : Bool, optional
            If you want to plot the corresponding interval for each line. The default is False.
        use_max_intensity : Bool, optional
            If instead of integrated intensity you want to use the maximum intensity. The default is False.

        Returns
        -------
        temperature : float
        temp_95 : float
            confidence interval at 95 percent
        r2 : float
            r-squared for the linear regression
        y_s : array
            y-array used in the plot
        x_s : array
            x-array used in the plot

        """
        new_wavelengths = np.concatenate([wl for wl in self.wavelengths])
        new_spectrum = np.concatenate([sp for sp in self.spectrum])
        new_spectrum = new_spectrum * new_wavelengths
        
        current_best_guess = T_ref
        previous_best_guess = T_ref
        
        temperature, temp_95, r2, y_s, x_s = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, electron_density=electron_density, 
                          Plot = False, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = use_max_intensity)
        
        current_best_guess = abs(temperature) 
        #print(temperature)
        
        #use previous update
        temperature_new, temp_95_new, r2_new, y_s_new, x_s_new = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, guess_temperature = current_best_guess, electron_density=electron_density, 
                          Plot = False, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = use_max_intensity)
        
        iteration = 0
        #new update
        #print(r2)
        #print(r2_new)
        
        if r2_new>r2:
            previous_best_guess = abs(temperature)
            current_best_guess = abs(temperature_new)
            temperature=abs(temperature_new)
            temp_95_new=temp_95 
            r2_new=r2 
            y_s_new = y_s 
            x_s_new = x_s
            
            print(" Saha-Boltzmann iteration "+str(iteration))
            temperature_new, temp_95_new, r2_new, y_s_new, x_s_new = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, guess_temperature = current_best_guess, electron_density=electron_density, 
                          Plot = False, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = use_max_intensity)
        
        
        temperature, temp_95, r2, y_s, x_s = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, electron_density=electron_density, guess_temperature = previous_best_guess,
                          Plot = Plot, Title = "Boltzmann plot", Plotlines = Plotlines, use_max_intensity = use_max_intensity)
        
        
        
        return temperature, temp_95, r2, y_s, x_s
    
    def saha_boltzmann_temperature_v2(self, lines, element , ratio_of_maximum = 0.5, radius = 1, electron_density=n_e_ref,
                          Plot = True, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = False):
        """
        
        Finds the plasma temperature using the Saha-Boltzmann plot method.

        Parameters
        ----------
        lines : list of lines
            Target lines.
        ion_energies : list of floats 
            List with the referent ion energies. 
        ratio_of_maximum : float, optional
            To compute the intersection. The default is 0.5.
        radius : float,optional
            range of wavelengths for finding the maximum value around the wavelength. The default is 1.
        Plot : Bool, optional
            If you want to obtain the plot. The default is True.
        Title : String, optional
            Title for the graph. The default is "Boltzmann plot".
        Plotlines : Bool, optional
            If you want to plot the corresponding interval for each line. The default is False.
        use_max_intensity : Bool, optional
            If instead of integrated intensity you want to use the maximum intensity. The default is False.

        Returns
        -------
        temperature : float
        temp_95 : float
            confidence interval at 95 percent
        r2 : float
            r-squared for the linear regression
        y_s : array
            y-array used in the plot
        x_s : array
            x-array used in the plot

        """
        
        ion_energies = element.ion_energies
        
        new_wavelengths = np.concatenate([wl for wl in self.wavelengths])
        new_spectrum = np.concatenate([sp for sp in self.spectrum])
        new_spectrum = new_spectrum * new_wavelengths
        
        current_best_guess = T_ref
        previous_best_guess = T_ref
        
        temperature, temp_95, r2, y_s, x_s = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, electron_density=electron_density, 
                          Plot = False, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = use_max_intensity)
        
        current_best_guess = abs(temperature) 
        #print(temperature)
        
        #use previous update
        temperature_new, temp_95_new, r2_new, y_s_new, x_s_new = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, guess_temperature = current_best_guess, electron_density=electron_density, 
                          Plot = False, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = use_max_intensity)
        
        iteration = 0
        #new update
        #print(r2)
        #print(r2_new)
        
        if r2_new>r2:
            previous_best_guess = abs(temperature)
            current_best_guess = abs(temperature_new)
            temperature=abs(temperature_new)
            temp_95_new=temp_95 
            r2_new=r2 
            y_s_new = y_s 
            x_s_new = x_s
            
            print(" Saha-Boltzmann iteration "+str(iteration))
            temperature_new, temp_95_new, r2_new, y_s_new, x_s_new = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, guess_temperature = current_best_guess, electron_density=electron_density, 
                          Plot = False, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = use_max_intensity)
        
        
        temperature, temp_95, r2, y_s, x_s = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, electron_density=electron_density, guess_temperature = previous_best_guess,
                          Plot = Plot, Title = Title, Plotlines = Plotlines, use_max_intensity = use_max_intensity)
        
        
        
        return temperature, temp_95, r2, y_s, x_s
    
    def multi_element_saha_boltzmann_temperature(self, list_elements_lines, list_elements, ratio_of_maximum = 0.5, radius = 1, electron_density=n_e_ref,
                          Plot = True, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = False):
        """
        
        Finds the plasma temperature using the Saha-Boltzmann plot method for multiple elements

        Parameters
        ----------
        list_elements_lines : list of lists of element lines
            Target lines.
        list_elements : list of elements
            List with the referent ion energies. 
        ratio_of_maximum : float, optional
            To compute the intersection. The default is 0.5.
        radius : float,optional
            range of wavelengths for finding the maximum value around the wavelength. The default is 1.
        Plot : Bool, optional
            If you want to obtain the plot. The default is True.
        Title : String, optional
            Title for the graph. The default is "Boltzmann plot".
        Plotlines : Bool, optional
            If you want to plot the corresponding interval for each line. The default is False.
        use_max_intensity : Bool, optional
            If instead of integrated intensity you want to use the maximum intensity. The default is False.

        Returns
        -------
        temperature : float
        temp_95 : float
            confidence interval at 95 percent
        r2 : float
            r-squared for the linear regression
        y_s : array
            y-array used in the plot
        x_s : array
            x-array used in the plot

        """
        new_wavelengths = np.concatenate([wl for wl in self.wavelengths])
        new_spectrum = np.concatenate([sp for sp in self.spectrum])
        new_spectrum = new_spectrum * new_wavelengths
        
        current_best_guess = T_ref
        previous_best_guess = T_ref
        temperature_total = []
        temp_95_total =[]
        r2_total = []
        y_s_total = []
        x_s_total =[]
        intercept_total=[]
        
        for i in range(0,len(list_elements)):
            
            lines = list_elements_lines[i]
            ion_energies = list_elements[i].ion_energies
        
            temperature, temp_95, r2, y_s, x_s = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, electron_density=electron_density, 
                              Plot = False, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = use_max_intensity)
            
            current_best_guess = abs(temperature) 
            #print(temperature)
            
            #use previous update
            temperature_new, temp_95_new, r2_new, y_s_new, x_s_new = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, guess_temperature = current_best_guess, electron_density=electron_density, 
                              Plot = False, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = use_max_intensity)
            
            iteration = 0
            #new update
            #print(r2)
            #print(r2_new)
            
            if r2_new>r2:
                previous_best_guess = abs(temperature)
                current_best_guess = abs(temperature_new)
                temperature=abs(temperature_new)
                temp_95_new=temp_95 
                r2_new=r2 
                y_s_new = y_s 
                x_s_new = x_s
                
                print(" Saha-Boltzmann iteration "+str(iteration))
                temperature_new, temp_95_new, r2_new, y_s_new, x_s_new = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, guess_temperature = current_best_guess, electron_density=electron_density, 
                              Plot = False, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = use_max_intensity)
            
            
            temperature, temp_95, r2, y_s, x_s = saha_boltzmann_temperature(lines, ion_energies , new_wavelengths, new_spectrum, ratio_of_maximum, radius, electron_density=electron_density, guess_temperature = previous_best_guess,
                              Plot = False, Title = "Boltzmann plot", Plotlines =  Plotlines, use_max_intensity = use_max_intensity)
            

            temperature_total.append(temperature)
            temp_95_total.append(temp_95)
            r2_total.append(r2)
            y_s_total.append(y_s)
            x_s_total.append(x_s)
        
        
        if Plot:
            fig,ax = subplots(figsize=(5,3),constrained_layout=True)
            
            marker = itertools.cycle(('+', 'x', 'o', '*'))
            
            for i in range(0,len(temperature_total)):
                
                x_s = x_s_total[i]
                y_s = y_s_total[i]

    
                regressor = linear_model.LinearRegression()
        
                x_train=x_s.reshape(len(x_s),1)
                y_train=y_s.reshape(len(y_s),1)
                regressor.fit(x_train, y_train)
        
                r2=regressor.score(x_train,y_train)
                slope = regressor.coef_[0][0]
                temperature=-1./(kb*slope)
                tev=temperature/T_ref
    
                result = stats.linregress(x_train[:,0],y_train[:,0])
        
                tinv = lambda p, df: abs(t.ppf(p/2, df))
                ts = tinv(0.05, len(x_s)-2)
                slope_95 = ts*result.stderr
                temp_95 = (slope_95/slope)*temperature
                temp_95_ev = (slope_95/slope)*tev
    

                
                
                ax.set_title(Title)
                ax.plot(x_s,y_s, marker = next(marker), ls='none',color = 'k', 
                        label=list_elements[i].label + ' lines' + '\n'
                        r'$T_{plasma}=$'+ "%0.2f" % tev + "$\pm$" + "%0.2f" % abs(temp_95_ev) + " ev")
        
                ax.plot(sorted(x_train),regressor.predict(sorted(x_train)),ls='--',color='k')
                
                #prediction bands
                N = x_s.size
                var_n = 2
                alpha = 1.0 - 0.95 #conf
                q = t.ppf(1.0 - alpha / 2.0, N - var_n)
                # Stdev of an individual measurement
                se = result.stderr
                sx = (x_s - x_s.mean()) ** 2
                sxd = np.sum((x_s - x_s.mean()) ** 2)
                dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
                fill = np.array([np.array([xxc, yyc, yyc2]) for xxc,yyc, yyc2 in sorted(zip(x_train[:,0],regressor.predict(x_train)[:,0]-dy,
                                                                 regressor.predict(x_train)[:,0]+dy))])
                xc, yc, yc2 = fill[:,0],fill[:,1],fill[:,2]
                
                ax.fill_between(xc,yc,yc2,ls='None',color='b',alpha=0.1)
                
                ax.set_xlabel(r"$E_i (ev)$")
                ax.set_ylabel(r"$log(I_{ij}^{*}/(g_i A_{ij}))$")
                
                stdev = np.sqrt(np.sum((regressor.predict(x_train)-y_train)**2) / (len(y_train) - 2))
                
                intercept_total.append(result.intercept)
                
            ax.legend(fontsize=14)
        
        return temperature_total, temp_95_total, r2_total, y_s_total, x_s_total, intercept_total
    
    
    
    def compare_to_digital_sample(self, sample, spectrometer = None, lower_limit = 200. , upper_limit = 900., electron_temperature = T_ref, electron_density=n_e_ref, max_ion_state = 3, 
                                  d_lambda=0.01, use_wavelengths = True, log_scale=False, params_voigt = [], resolution = 3000, line_normalize = None, Plotline=False):
        
        if spectrometer == None:
            
            if use_wavelengths == True:
                new_wavelengths = np.concatenate([wl for wl in self.wavelengths])
            else:
                new_wavelengths = []
            
            wl,spec,label, n_ion, specs = sample.spectrum_NIST(wl = new_wavelengths, 
                                lower_limit = lower_limit, upper_limit = upper_limit, electron_temperature=electron_temperature, 
                                electron_density=electron_density,max_ion_state=max_ion_state,d_lambda=d_lambda,params_voigt = params_voigt,resolution = resolution)
            
            new_wavelengths = wl[0]
            
            fig, ax = subplots()
            ax.set_title('Comparing with the synthetic spectrum of an element at $T_p='+str(electron_temperature/T_ref)+"$ ev")
            lines=[]
            pols=[]
            
            if line_normalize == None:
                max_spec = array(self.spectrum).max()
    
                for i in range(0,len(self.wavelengths)):
                    line1, = ax.plot(self.wavelengths[i], self.spectrum[i]/max_spec, lw=2, color='k')
                    lines.append(line1)
                    
                max_spec = array(spec).max()
                
                for i in range(0,len(spec)):
                    for j in range(0,len(spec[i])):
                        coll = ax.fill_between(new_wavelengths, spec[i][j]/max_spec, label=label[i][j] )
                        pols.append(coll)
                leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
                leg.get_frame().set_alpha(0.4)
                
                
                # we will set up a dict mapping legend line to orig line, and enable
                # picking on the legend line
                lined = dict()
                for legline, origline in zip(leg.get_lines(), lines):
                #for legline, origline in zip(ax.get_legend_handles_labels()[0], lines):
                #for legline, origline in zip(lines, lines):
                    legline.set_pickradius(5)  # 5 pts tolerance
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                for legline, origline in zip(leg.get_patches(), pols):
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                
                def onpick(event):
                    # on the pick event, find the orig line corresponding to the
                    # legend proxy line, and toggle the visibility
                    legline = event.artist
                    origline = lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    # Change the alpha on the line in the legend so we can see what lines
                    # have been toggled
                    if vis:
                        legline.set_alpha(1.0)
                    else:
                        legline.set_alpha(0.2)
                    fig.canvas.draw()
                
                fig.canvas.mpl_connect('pick_event', onpick)
            
            else:
                
                
                ratio_of_maximum = 0.5
                radius = 0.1
                new_wavelengths_0 = np.concatenate([wl for wl in self.wavelengths])
                new_spectrum = np.concatenate([sp for sp in self.spectrum])
                
                ################################
                
                new_spectrum *= new_wavelengths_0 
                
                ################################
                
                intensity = get_peak_area(line_normalize.ritz, ratio_of_maximum, new_wavelengths_0, new_spectrum, radius, Plotline, Title = "Normalization line")
                
                max_spec0 = intensity
                
                for i in range(0,len(self.wavelengths)):
                    #line1, = ax.plot(self.wavelengths[i], self.spectrum[i]/max_spec0, lw=2, color='k')
                    
                    ##################
                    line1, = ax.plot(self.wavelengths[i], self.spectrum[i]*self.wavelengths[i]/max_spec0, lw=2, color='k')
                    ##################

                    lines.append(line1)
                    
                new_spec = specs.sum(axis=0)
                
                intensity = get_peak_area(line_normalize.ritz, ratio_of_maximum, new_wavelengths, new_spec, radius, False)
                
                
                max_spec = intensity
                
                for i in range(0,len(spec)):
                    for j in range(0,len(spec[i])):
                        coll = ax.fill_between(new_wavelengths, spec[i][j]/max_spec, label=label[i][j] )
                        pols.append(coll)
                leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
                leg.get_frame().set_alpha(0.4)
                
                if log_scale == True:
                    ax.set_yscale('log')
                    ax.set_ylim(0.1,1.)
                
                
                # we will set up a dict mapping legend line to orig line, and enable
                # picking on the legend line
                lined = dict()
                for legline, origline in zip(leg.get_lines(), lines):
                #for legline, origline in zip(ax.get_legend_handles_labels()[0], lines):
                #for legline, origline in zip(lines, lines):
                    legline.set_pickradius(5)  # 5 pts tolerance
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                for legline, origline in zip(leg.get_patches(), pols):
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                
                def onpick(event):
                    # on the pick event, find the orig line corresponding to the
                    # legend proxy line, and toggle the visibility
                    legline = event.artist
                    origline = lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    # Change the alpha on the line in the legend so we can see what lines
                    # have been toggled
                    if vis:
                        legline.set_alpha(1.0)
                    else:
                        legline.set_alpha(0.2)
                    fig.canvas.draw()
                
                fig.canvas.mpl_connect('pick_event', onpick)
        
        ##If only one spectrometer
        else:
            i = spectrometer
    
            if use_wavelengths == True:
                new_wavelengths = self.wavelengths[i]
            else:
                new_wavelengths = []
                upper_limit = self.wavelengths[i][-1]
                lower_limit = self.wavelengths[i][0]
            
            wl,spec,label, n_ion, specs = sample.spectrum_NIST(wl = new_wavelengths, 
                                lower_limit = lower_limit, upper_limit = upper_limit, electron_temperature=electron_temperature, 
                                electron_density=electron_density,max_ion_state=max_ion_state,d_lambda=d_lambda,params_voigt = params_voigt,resolution = resolution)
            
            new_wavelengths = wl[0]
            
            fig, ax = subplots()
            ax.set_title('Comparing with the synthetic spectrum of an element at $T_p='+str(electron_temperature/T_ref)+"$ ev")
            lines=[]
            pols=[]
            
            if line_normalize == None:
                max_spec = array(self.spectrum).max()
    
                
                line1, = ax.plot(self.wavelengths[i], self.spectrum[i]/max_spec, lw=1, color='k')
                lines.append(line1)
                    
                max_spec = array(spec).max()
                
                for i in range(0,len(spec)):
                    for j in range(0,len(spec[i])):
                        coll = ax.fill_between(new_wavelengths, spec[i][j]/max_spec, label=label[i][j] )
                        pols.append(coll)
                leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
                leg.get_frame().set_alpha(0.4)
                
                
                # we will set up a dict mapping legend line to orig line, and enable
                # picking on the legend line
                lined = dict()
                for legline, origline in zip(leg.get_lines(), lines):
                #for legline, origline in zip(ax.get_legend_handles_labels()[0], lines):
                #for legline, origline in zip(lines, lines):
                    legline.set_pickradius(5)  # 5 pts tolerance
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                for legline, origline in zip(leg.get_patches(), pols):
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                
                def onpick(event):
                    # on the pick event, find the orig line corresponding to the
                    # legend proxy line, and toggle the visibility
                    legline = event.artist
                    origline = lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    # Change the alpha on the line in the legend so we can see what lines
                    # have been toggled
                    if vis:
                        legline.set_alpha(1.0)
                    else:
                        legline.set_alpha(0.2)
                    fig.canvas.draw()
                
                fig.canvas.mpl_connect('pick_event', onpick)
            
            else:
                
                
                ratio_of_maximum = 0.5
                radius = 0.1
                new_wavelengths_0 = self.wavelengths[i]
                new_spectrum = copy(self.spectrum[i])
                
                ################################
                
                new_spectrum *= new_wavelengths_0 
                
                ################################
                
                fig0,ax0 = subplots()
                
                intensity = get_peak_area(line_normalize.ritz, ratio_of_maximum, new_wavelengths_0, new_spectrum, radius, Plotline, Title = "Normalization line")
                
                max_spec0 = intensity
                
                if intensity==None:
                    print('Normalization line not found in range')
                    return None
                

                ##################
                line1, = ax.plot(self.wavelengths[i], self.spectrum[i]*self.wavelengths[i]/max_spec0, lw=1, color='k')
                ##################
                    
                lines.append(line1)
                    
                new_spec = specs.sum(axis=0)
                
                intensity = get_peak_area(line_normalize.ritz, ratio_of_maximum, new_wavelengths, new_spec, radius, False)
                
                
                max_spec = intensity
                
                for i in range(0,len(spec)):
                    for j in range(0,len(spec[i])):
                        coll = ax.fill_between(new_wavelengths, spec[i][j]/max_spec, label=label[i][j] )
                        pols.append(coll)
                
                
                coll = ax.fill_between(new_wavelengths, new_spec/max_spec, alpha=0.8, label="Total")
                pols.append(coll)
                
                leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
                leg.get_frame().set_alpha(0.4)
                
                if log_scale == True:
                    ax.set_yscale('log')
                    ax.set_ylim(0.1,1.)
                
                
                # we will set up a dict mapping legend line to orig line, and enable
                # picking on the legend line
                lined = dict()
                for legline, origline in zip(leg.get_lines(), lines):
                #for legline, origline in zip(ax.get_legend_handles_labels()[0], lines):
                #for legline, origline in zip(lines, lines):
                    legline.set_pickradius(5)  # 5 pts tolerance
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                for legline, origline in zip(leg.get_patches(), pols):
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                
                def onpick(event):
                    # on the pick event, find the orig line corresponding to the
                    # legend proxy line, and toggle the visibility
                    legline = event.artist
                    origline = lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    # Change the alpha on the line in the legend so we can see what lines
                    # have been toggled
                    if vis:
                        legline.set_alpha(1.0)
                    else:
                        legline.set_alpha(0.2)
                    fig.canvas.draw()
                
                fig.canvas.mpl_connect('pick_event', onpick)
        
    def remove_baseline(self, method = 'ALS'):
        if method == 'ALS':
            for i in range(0, len(self.spectrum)):
                
                baseline = baseline_als(self.spectrum[i],100000,0.05)
                self.spectrum[i] -= baseline
            
        
        return baseline
    
    
    def integrate_region(self, lower_limit, upper_limit, Plot=False):
        
        result = 0
        
        #find where the limits are located, returns 0 if not within the limits of a single spectrometer
        
        spectrometer_number = None
        
        for i in range(0,len(self.wavelengths)):
            
            lower_index_c = find_wavelength_index(lower_limit,self.wavelengths[i])
            if lower_index_c!=None:
                
                upper_index_c = find_wavelength_index(upper_limit,self.wavelengths[i])
                if upper_index_c != None:
                    spectrometer_number = i
                    lower_index = lower_index_c
                    upper_index = upper_index_c
        
        if spectrometer_number!=None:
            
            x = self.wavelengths[spectrometer_number][lower_index:upper_index+1]
            y = self.spectrum[spectrometer_number][lower_index:upper_index+1]
            
            result = np.trapz(y,x)
            
            if Plot:
                fig=subplots()
                
                ax = gca()
                fr = 30
                
                ax.set_title("Integrated region")
                ax.plot(self.wavelengths[spectrometer_number][lower_index-fr:upper_index+1+fr],
                        self.spectrum[spectrometer_number][lower_index-fr:upper_index+1+fr],
                     ls='-',lw=0.5,color='k', label = 'Signal')
                
                ax.fill_between(x,y,
                                color='b',alpha=0.1,label='Integrated Area')
                
                ax.legend()
                            
        else:
            print('Region not found in data')
        
        return result
    
    def density_H_alpha(self,fit="Voigt",Plot=True, compare_methods = False,ratio_max=0.1):
            
        
        line_ritz = 656.3
        new_wavelengths = np.concatenate([wl for wl in self.wavelengths])
        new_spectrum = np.concatenate([sp for sp in self.spectrum])
        new_spectrum += -baseline_als(new_spectrum,100000,0.05)
        ##not correct##
        radius=0.3
        center_wavelength_peak_index = get_closest_peak_index(line_ritz, radius, new_wavelengths, new_spectrum)
        index_lr = get_closest_peak_width_index(center_wavelength_peak_index,ratio_max, new_wavelengths, new_spectrum)
        
        if compare_methods==False:
            if fit == "Voigt":
                current_profile_binder = voigt_profile_binder
            elif fit == "Gaussian":
                current_profile_binder = gaussian_profile_binder
            else:
                current_profile_binder = lorentzian_profile_binder
            
            params, pcov = binder_fit(new_wavelengths[index_lr[0]:index_lr[1]], new_spectrum[index_lr[0]:index_lr[1]],
                                     current_profile_binder,
                                     initial_guess_x0 = line_ritz, initial_guess_A = new_spectrum[center_wavelength_peak_index])
            
            xx = np.arange(new_wavelengths[index_lr[0]],new_wavelengths[index_lr[1]],0.001)
            fwhm, fwhm_x_v, fwhm_y_v = fit_fwhm(xx,params,current_profile_binder)
            
            #density = 10e17*pow(fwhm/1.31,1/0.64)
            #density = 9.77e16*pow(fwhm,1.39)
            density = (fwhm/(2.8*10**-17))**(1/0.72)*10**-6
            if Plot:
                figure()
                title(r'$H_{\alpha}$ line fit',fontsize=12)
                plot(new_wavelengths[index_lr[0]:index_lr[1]],new_spectrum[index_lr[0]:index_lr[1]],'-',color='k',label='Signal')
                fill_between(xx,current_profile_binder(xx,params[0],params[1],params[2],params[3]),color='b',alpha=0.2,label=fit+' fit')
                plot(fwhm_x_v,fwhm_y_v,ls = ':', color = 'k', marker = '|',markersize = 10)
                ax = gca()
                ax.text(0.3, 0.2,  '$n_{plasma} =$' + "%1.2e" % density +' $cm^{-3}$',
                transform=ax.transAxes,fontsize=12, bbox={'boxstyle':'round', 'facecolor': 'wheat', 'alpha': 0.5, 'pad': 0.5})
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
                legend(fontsize=12)
            

            return density, new_wavelengths[index_lr[0]:index_lr[1]], new_spectrum[index_lr[0]:index_lr[1]], params
        
        else:
            figure()
            title(r'$H_{\alpha}$ line fit',fontsize=12)
            plot(new_wavelengths[index_lr[0]:index_lr[1]],new_spectrum[index_lr[0]:index_lr[1]],'-',color='k',label='Signal')
                             
            for binder in [[voigt_profile_binder,'Voigt','b'],
                           [gaussian_profile_binder,'Gaussian','r'],
                           [lorentzian_profile_binder,'Lorentzian','g']]:
            
                current_profile_binder = binder[0]
                fit = binder[1]
                color1=binder[2]
                
            
                params, pcov = binder_fit(new_wavelengths[index_lr[0]:index_lr[1]], new_spectrum[index_lr[0]:index_lr[1]],
                                         current_profile_binder,
                                         initial_guess_x0 = line_ritz, initial_guess_A = new_spectrum[center_wavelength_peak_index])
                
                
                xx = np.arange(new_wavelengths[index_lr[0]],new_wavelengths[index_lr[1]],0.001)
                fwhm, fwhm_x_v, fwhm_y_v = fit_fwhm(xx,params,current_profile_binder)
                
                
                #density = 10e17*pow(fwhm/1.31,1/0.64)
                density = (fwhm/(2.8*10**-17))**(1/0.72)*10**-6
                
                
                fill_between(xx,current_profile_binder(xx,params[0],params[1],params[2],params[3]),color=color1,alpha=0.2,label=fit+' fit \n'+ '$n_{plasma} =$' + "%1.2e" % density +' $cm^{-3}$\n')
                plot(fwhm_x_v,fwhm_y_v,ls = ':', color = 'k', marker = '|',markersize = 10)
                ax = gca()
                
            
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            legend(fontsize=12)
                    
                    
                

            return density, new_wavelengths[index_lr[0]:index_lr[1]], new_spectrum[index_lr[0]:index_lr[1]], params
        
        
    def LTE_MW_criterium(self,delta_E, Tp):
        
            """
            Parameters
            ----------
            delta_E : float
                Largest energy difference between adjacent levels, in ev.
            Tp : float
                Plasma temperature in K.

            Returns
            -------
            lowest_density : float
                Lowest density for fulfilling the McWrither criterium.
            """
            
            lowest_density = 1.6*10**12*(Tp**0.5)*(delta_E)**3
            return lowest_density


        
    def compare_and_peaks(self, sample, spectrometer = None, lower_limit = 200. , upper_limit = 900., electron_temperature = T_ref, electron_density=n_e_ref, max_ion_state = 3, 
                                  d_lambda=0.01, use_wavelengths = True, log_scale=False, params_voigt = [], resolution = 3000, peak_prominence = 2, line_normalize = None, Plotline=False):
        
        
        def classification_line(value):
            if value>0.05 and value<0.3:
                return "Low Intensity"
            elif value>=0.3 and value<0.5:
                return "Medium Intensity"
            elif value>=0.5 and value<=1:
                return "High Intensity"
            elif value>=0.0 and value<0.05:
                return "Ultra low Intensity"
            else:
                return "NA"
        
        def find_possible_lines1(peak_wavelength):
            peak_wavelength = round(peak_wavelength, 1)
            df = pd.read_csv("d_lines.txt",sep=';')
            df['Line']=[round(f,2) for f in df['Line']]
            df.sort_values(by=['Line','Relative Intensity'], ascending=[True, False], inplace=True)
            df1 = df[df['Relative Intensity']>0.01]
            df0 = df1.copy(deep=True)
            df0['Relative Intensity'] = [round(f,3) for f in df1['Relative Intensity']]
            df0['Class'] = [classification_line(f) for f in df1['Relative Intensity']]
            df2 = df0[df0['Class']=="High Intensity"]
            df3 = df0[(df0["Line"]<(peak_wavelength+0.1))&(df0["Line"]>(peak_wavelength-0.1))]
            df4 = df3.sort_values(by=['Relative Intensity'],ascending=[False])
            return df4.to_numpy()

        if spectrometer == None:
            
            if use_wavelengths == True:
                new_wavelengths = np.concatenate([wl for wl in self.wavelengths])
            else:
                new_wavelengths = []
            
            wl,spec,label, n_ion, specs = sample.spectrum_NIST(wl = new_wavelengths, 
                                lower_limit = lower_limit, upper_limit = upper_limit, electron_temperature=electron_temperature, 
                                electron_density=electron_density,max_ion_state=max_ion_state,d_lambda=d_lambda,params_voigt = params_voigt,resolution = resolution)
            
            new_wavelengths = wl[0]
            
            fig, ax = subplots()
            ax.set_title('Comparing with the synthetic spectrum of an element at $T_p='+str(electron_temperature/T_ref)+"$ ev")
            lines=[]
            pols=[]
            
            if line_normalize == None:
                max_spec = array(self.spectrum).max()
    
                for i in range(0,len(self.wavelengths)):
                    line1, = ax.plot(self.wavelengths[i], self.spectrum[i]/max_spec, lw=2, color='k')
                    lines.append(line1)
                    
                max_spec = array(spec).max()
                
                for i in range(0,len(spec)):
                    for j in range(0,len(spec[i])):
                        coll = ax.fill_between(new_wavelengths, spec[i][j]/max_spec, label=label[i][j] )
                        pols.append(coll)
                leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
                leg.get_frame().set_alpha(0.4)
                
                
                # we will set up a dict mapping legend line to orig line, and enable
                # picking on the legend line
                lined = dict()
                for legline, origline in zip(leg.get_lines(), lines):
                #for legline, origline in zip(ax.get_legend_handles_labels()[0], lines):
                #for legline, origline in zip(lines, lines):
                    legline.set_pickradius(5)  # 5 pts tolerance
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                for legline, origline in zip(leg.get_patches(), pols):
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                
                def onpick(event):
                    # on the pick event, find the orig line corresponding to the
                    # legend proxy line, and toggle the visibility
                    legline = event.artist
                    origline = lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    # Change the alpha on the line in the legend so we can see what lines
                    # have been toggled
                    if vis:
                        legline.set_alpha(1.0)
                    else:
                        legline.set_alpha(0.2)
                    fig.canvas.draw()
                
                fig.canvas.mpl_connect('pick_event', onpick)
            
            else:
                
                
                ratio_of_maximum = 0.5
                radius = 0.1
                new_wavelengths_0 = np.concatenate([wl for wl in self.wavelengths])
                new_spectrum = np.concatenate([sp for sp in self.spectrum])
                
                ################################
                
                new_spectrum *= new_wavelengths_0 
                
                ################################
                
                intensity = get_peak_area(line_normalize.ritz, ratio_of_maximum, new_wavelengths_0, new_spectrum, radius, Plotline, Title = "Normalization line")
                
                max_spec0 = intensity
                
                for i in range(0,len(self.wavelengths)):
                    #line1, = ax.plot(self.wavelengths[i], self.spectrum[i]/max_spec0, lw=2, color='k')
                    
                    ##################
                    line1, = ax.plot(self.wavelengths[i], self.spectrum[i]*self.wavelengths[i]/max_spec0, lw=2, color='k')
                    ##################

                    lines.append(line1)
                    
                new_spec = specs.sum(axis=0)
                
                intensity = get_peak_area(line_normalize.ritz, ratio_of_maximum, new_wavelengths, new_spec, radius, False)
                
                
                max_spec = intensity
                
                for i in range(0,len(spec)):
                    for j in range(0,len(spec[i])):
                        coll = ax.fill_between(new_wavelengths, spec[i][j]/max_spec, label=label[i][j] )
                        pols.append(coll)
                leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
                leg.get_frame().set_alpha(0.4)
                
                if log_scale == True:
                    ax.set_yscale('log')
                    ax.set_ylim(0.1,1.)
                
                
                # we will set up a dict mapping legend line to orig line, and enable
                # picking on the legend line
                lined = dict()
                for legline, origline in zip(leg.get_lines(), lines):
                #for legline, origline in zip(ax.get_legend_handles_labels()[0], lines):
                #for legline, origline in zip(lines, lines):
                    legline.set_pickradius(5)  # 5 pts tolerance
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                for legline, origline in zip(leg.get_patches(), pols):
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                
                def onpick(event):
                    # on the pick event, find the orig line corresponding to the
                    # legend proxy line, and toggle the visibility
                    legline = event.artist
                    origline = lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    # Change the alpha on the line in the legend so we can see what lines
                    # have been toggled
                    if vis:
                        legline.set_alpha(1.0)
                    else:
                        legline.set_alpha(0.2)
                    fig.canvas.draw()
                
                fig.canvas.mpl_connect('pick_event', onpick)
        
        ##If only one spectrometer
        else:
            i = spectrometer
    
            if use_wavelengths == True:
                new_wavelengths = self.wavelengths[i]
            else:
                new_wavelengths = []
                upper_limit = self.wavelengths[i][-1]
                lower_limit = self.wavelengths[i][0]
            
            wl,spec,label, n_ion, specs = sample.spectrum_NIST(wl = new_wavelengths, 
                                lower_limit = lower_limit, upper_limit = upper_limit, electron_temperature=electron_temperature, 
                                electron_density=electron_density,max_ion_state=max_ion_state,d_lambda=d_lambda,params_voigt = params_voigt,resolution = resolution)
            
            new_wavelengths = wl[0]
            
            fig, ax = subplots()
            ax.set_title('Comparing with the synthetic spectrum of an element at $T_p='+str(electron_temperature/T_ref)+"$ ev")
            lines=[]
            pols=[]
            
            if line_normalize == None:
                max_spec = array(self.spectrum).max()
    
                
                line1, = ax.plot(self.wavelengths[i], self.spectrum[i]/max_spec, lw=1, color='k')
                lines.append(line1)
                    
                max_spec = array(spec).max()
                
                for i in range(0,len(spec)):
                    for j in range(0,len(spec[i])):
                        coll = ax.fill_between(new_wavelengths, spec[i][j]/max_spec, label=label[i][j] )
                        pols.append(coll)
                leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
                leg.get_frame().set_alpha(0.4)
                
                
                # we will set up a dict mapping legend line to orig line, and enable
                # picking on the legend line
                lined = dict()
                for legline, origline in zip(leg.get_lines(), lines):
                #for legline, origline in zip(ax.get_legend_handles_labels()[0], lines):
                #for legline, origline in zip(lines, lines):
                    legline.set_pickradius(5)  # 5 pts tolerance
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                for legline, origline in zip(leg.get_patches(), pols):
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                
                def onpick(event):
                    # on the pick event, find the orig line corresponding to the
                    # legend proxy line, and toggle the visibility
                    legline = event.artist
                    origline = lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    # Change the alpha on the line in the legend so we can see what lines
                    # have been toggled
                    if vis:
                        legline.set_alpha(1.0)
                    else:
                        legline.set_alpha(0.2)
                    fig.canvas.draw()
                
                fig.canvas.mpl_connect('pick_event', onpick)
            
            else:
                
                
                ratio_of_maximum = 0.5
                radius = 0.1
                new_wavelengths_0 = self.wavelengths[i]
                new_spectrum = copy(self.spectrum[i])
                
                ###############################
                
                new_spectrum *= new_wavelengths_0 
                
                ###############################
                
                fig0,ax0 = subplots()
                
                intensity = get_peak_area(line_normalize.ritz, ratio_of_maximum, new_wavelengths_0, new_spectrum, radius, Plotline, Title = "Normalization line")
                
                max_spec0 = intensity
                
                if intensity==None:
                    print('Normalization line not found in range')
                    return None
                
                peaks_index = signal.find_peaks(self.spectrum[i]*self.wavelengths[i]/max_spec0,  
                                                prominence = peak_prominence)
                peaks_index = peaks_index[0]
                
                peaks_found = self.wavelengths[i][peaks_index]
                
                #print(peaks_found)
                list_possible_lines = []
                for jj in range(0,len(peaks_found)):
                    list_possible_lines.append(find_possible_lines1(peaks_found[jj]+0.05))
                
                sc = ax.scatter(self.wavelengths[i][peaks_index],
                                self.spectrum[i][peaks_index]*self.wavelengths[i][peaks_index]/max_spec0, 
                                s=100)

                annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="w"),
                                    arrowprops=dict(arrowstyle="->"))
                annot.set_visible(False)
                
                names = array(list_possible_lines)
                
                ##################
                line1, = ax.plot(self.wavelengths[i], self.spectrum[i]*self.wavelengths[i]/max_spec0, lw=1, color='k')
                ##################
                    
                lines.append(line1)
                    
                new_spec = specs.sum(axis=0)
                
                intensity = get_peak_area(line_normalize.ritz, ratio_of_maximum, new_wavelengths, new_spec, radius, False)
                
                
                max_spec = intensity
                
                for i in range(0,len(spec)):
                    for j in range(0,len(spec[i])):
                        coll = ax.fill_between(new_wavelengths, spec[i][j]/max_spec, label=label[i][j] )
                        pols.append(coll)
                
                
                coll = ax.fill_between(new_wavelengths, new_spec/max_spec, alpha=0.8, label="Total")
                pols.append(coll)
                
                leg = ax.legend(loc='upper left', fancybox=True, shadow=True)
                leg.get_frame().set_alpha(0.4)
                
                if log_scale == True:
                    ax.set_yscale('log')
                    ax.set_ylim(0.1,1.)
                
                
                # we will set up a dict mapping legend line to orig line, and enable
                # picking on the legend line
                lined = dict()
                for legline, origline in zip(leg.get_lines(), lines):
                #for legline, origline in zip(ax.get_legend_handles_labels()[0], lines):
                #for legline, origline in zip(lines, lines):
                    legline.set_pickradius(5)  # 5 pts tolerance
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                for legline, origline in zip(leg.get_patches(), pols):
                    legline.set_picker(True)
                    lined[legline] = origline
                
                
                
                def onpick(event):
                    # on the pick event, find the orig line corresponding to the
                    # legend proxy line, and toggle the visibility
                    legline = event.artist
                    origline = lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    # Change the alpha on the line in the legend so we can see what lines
                    # have been toggled
                    if vis:
                        legline.set_alpha(1.0)
                    else:
                        legline.set_alpha(0.2)
                    fig.canvas.draw()
                
                fig.canvas.mpl_connect('pick_event', onpick)
                
                def update_annot(ind):

                    pos = sc.get_offsets()[ind["ind"][0]]
                    annot.xy = pos
                    nn =[names[n] for n in ind["ind"]]
                    text = "{}".format("\n".join(["".join(str(f)) for f in nn]))
                        #"".join(str([names[n] for n in ind["ind"]])))
                    annot.set_text(text)
                    annot.get_bbox_patch().set_facecolor("b")
                    annot.get_bbox_patch().set_alpha(0.4)


                def hover(event):
                    vis = annot.get_visible()
                    if event.inaxes == ax:
                        cont, ind = sc.contains(event)
                        if cont:
                            update_annot(ind)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                        else:
                            if vis:
                                annot.set_visible(False)
                                fig.canvas.draw_idle()
                                
                fig.canvas.mpl_connect("motion_notify_event", hover)


        
    def remove_baseline(self, method = 'ALS'):
        if method == 'ALS':
            for i in range(0, len(self.spectrum)):
                
                baseline = baseline_als(self.spectrum[i],100000,0.05)
                self.spectrum[i] -= baseline
            
        
        return baseline
    
    def get_baseline(self, method = 'ALS'):
        if method == 'ALS':
            for i in range(0, len(self.spectrum)):
                
                baseline = baseline_als(self.spectrum[i],100000,0.05)
        
        return baseline
    
    
    def normalize(self):
        norm = 0
        
        for i in range(0,len(self.spectrum)):
            norm+=np.sum(self.spectrum[i])
        
        self.spectrum=self.spectrum/norm
        
    def SNV(self, key='Full'):
        new_spectra = []
        if key == 'Full':
            #print(key)
            for i in range(0,len(self.spectrum)):
                temp = self.spectrum[i]-np.mean(np.array(self.spectrum).flatten())
                std_d = np.std(np.array(self.spectrum).flatten())
                new_spectra.append(temp/std_d)
        
        else:
            #print(key)
            for i in range(0,len(self.spectrum)):
                temp = self.spectrum[i]-np.mean(self.spectrum[i])
                std_d = np.std(np.array(self.spectrum[i]))
                new_spectra.append(temp/std_d)
        
        self.spectrum=new_spectra



def func():
    folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_27 #1\\340spot3\\"
    filename = "1703275U8_0001_1703275U8.TXT"
    sg = signal_spec(folder,filename)
    
    
    #print(p)
    #density, x,y,p = sg.density_H_alpha(fit="Lorentzian")
    #print(p)
    #density, x,y,p = sg.density_H_alpha(fit="Gaussian")
    #print(p)
    
    density, x,y,p = sg.density_H_alpha(compare_methods=True)
    
    density, x,y,p = sg.density_H_alpha()
    
    
    
    density= density*(10**2)**3

    sg = signal_spec(folder,filename)
    
    al = element('Fe')
    #
    sample = digital_twin([['P',2.3e-3],["Cr",2.0e-4],["Mn",6.96e-4],['As',2.13e-3],['Mo',4.89e-6],
                           ['Sn',1.34e-5],['Sb',1.5e-6],['W',4.59e-4],['Fe',6.31e-2]])
    ll = 230.
    ul = 900.
    Tp = 1.*T_ref
    new_wavelengths = np.concatenate([wl for wl in sg.wavelengths])
    #wl,spec,label, n_ion = al.synthetic_spectrum_NIST(wl = new_wavelengths, lower_limit = ll, upper_limit = ul, electron_temperature=Tp,max_ion_state=3,d_lambda=0.1)
    wl,spec,label, n_ion, specs = sample.spectrum_NIST(wl = new_wavelengths, lower_limit = ll, upper_limit = ul, 
                                                       electron_density=density, electron_temperature=Tp,
                                                       max_ion_state=3,d_lambda=0.01, Map=True)
    wl=wl[0]
    l_num, lines = al.get_most_relevant_lines_ion_state(ion_state = 1, electron_temperature=Tp, n_lines = 5,lower_limit = ll, upper_limit = ul)
    l_num2, lines2 = al.get_most_relevant_lines_ion_state(ion_state = 2, electron_temperature=Tp, n_lines = 2,lower_limit = ll, upper_limit = ul)
    l_num3, lines3 = al.get_most_relevant_lines_ion_state(ion_state = 3, electron_temperature=Tp, n_lines = 3,lower_limit = ll, upper_limit = ul)
    ion_energies = al.ion_energies
    
    signal = spec[0]+spec[1]+spec[2]
    
    #saha_boltzmann_temperature([lines,lines2,lines3], ion_energies, wl, signal, ratio_of_maximum = 0.5, radius = 0.1, Plot = True, 
    #                      Plotlines = False, use_max_intensity = False)
    
    
    temperature, temp_95, r2, y_s, x_s = sg.saha_boltzmann_temperature_v2([lines,lines2],ion_energies, electron_density=density, ratio_of_maximum = 0.5, radius = 0.1, Plot = True, Plotlines = True, use_max_intensity = False)
    
    #temperature = T_ref
    line_norm=lines[1]
    sg.compare_to_digital_sample(sample,
                                 electron_temperature = temperature, electron_density=density, d_lambda=0.02,
                                 params_voigt=[],
                                 use_wavelengths=False,line_normalize=line_norm)
    
    
    
    """
    al = element('Sn')
    ll = 200.
    ul = 600.
    Tp = 2*T_ref
    wl,spec,label, n_ion = al.synthetic_spectrum_NIST(lower_limit = ll, upper_limit = ul, electron_temperature=Tp,max_ion_state=4,d_lambda=0.1)
    l_num, lines = al.get_most_relevant_lines_ion_state(ion_state = 1, electron_temperature=Tp, n_lines = 3,lower_limit = ll, upper_limit = ul)
    l_num2, lines2 = al.get_most_relevant_lines_ion_state(ion_state = 2, electron_temperature=Tp, n_lines = 3,lower_limit = ll, upper_limit = ul)
    l_num3, lines3 = al.get_most_relevant_lines_ion_state(ion_state = 3, electron_temperature=Tp, n_lines = 3,lower_limit = ll, upper_limit = ul)
    ion_energies = al.ion_energies
    
    signal = spec[0]+spec[1]+spec[2]
    
    #saha_boltzmann_temperature([lines,lines2,lines3], ion_energies, wl, signal, ratio_of_maximum = 0.5, radius = 0.1, Plot = True, 
    #                      Plotlines = False, use_max_intensity = False)
    
    
    sg.saha_boltzmann_temperature([lines,lines2,lines3],ion_energies, ratio_of_maximum = 0.7, radius = 1., Plot = True, Plotlines = True, use_max_intensity = False)
    """
    print(sg.LTE_MW_criterium(3.2,temperature))
    return n_ion,temperature


def func1():
    folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_27 #1\\340spot3\\"
    filename = "1703275U8_0001_1703275U8.TXT"
    sg = signal_spec(folder,filename)
    
    density, x,y,p = sg.density_H_alpha()
    density= density*(10**2)**3

    sg = signal_spec(folder,filename)
    
    al = element('Fe')
    #
    sample = digital_twin([['P',2.3e-3],["Cr",2.0e-4],["Mn",6.96e-4],['As',2.13e-3],['Mo',4.89e-6],
                           ['Sn',1.34e-5],['Sb',1.5e-6],['W',4.59e-4],['Fe',6.31e-2]])
    
    n_spectrometer = 2
    ll = sg.wavelengths[n_spectrometer][0]
    ul = sg.wavelengths[n_spectrometer][-1]
    
    Tp = 1.*T_ref
    
    new_wavelengths = np.concatenate([wl for wl in sg.wavelengths])


    l_num, lines = al.get_most_relevant_lines_ion_state(ion_state = 1, electron_temperature=Tp, n_lines = 5,lower_limit = ll, upper_limit = ul)
    l_num2, lines2 = al.get_most_relevant_lines_ion_state(ion_state = 2, electron_temperature=Tp, n_lines = 2,lower_limit = ll, upper_limit = ul)
    l_num3, lines3 = al.get_most_relevant_lines_ion_state(ion_state = 3, electron_temperature=Tp, n_lines = 3,lower_limit = ll, upper_limit = ul)
    ion_energies = al.ion_energies    

    temperature, temp_95, r2, y_s, x_s = sg.saha_boltzmann_temperature_v2([lines,lines2],al, electron_density=density, ratio_of_maximum = 0.5, radius = 0.1, Plot = True, Plotlines = True, use_max_intensity = False)
    
    #temperature = T_ref
    line_norm=lines[1]
    sg.wavelengths[n_spectrometer] = sg.wavelengths[n_spectrometer]+0.05
    sg.compare_and_peaks(sample, spectrometer =n_spectrometer,resolution=5000,
                                 electron_temperature = temperature, electron_density=density, d_lambda=0.02,
                                 params_voigt=[],
                                 use_wavelengths=False,line_normalize=line_norm)
    
    

    print(sg.LTE_MW_criterium(3.2,temperature))
    return temperature

#func1()

def func2():
    folder = "C:\\Users\\nunoa\\LIBS - simulation and data analysis\\data_repo\\RF_SOL_27 #1\\340spot3\\"
    filename = "1703275U8_0001_1703275U8.TXT"    
    sg = signal_spec(folder,filename)

    sample = digital_twin([['Mg',80],['Zn',20],['Fe',10]])
    ll = 250
    ul = 700.
    Tp = 1.*T_ref
        
    wl,spec,label, n_ion, specs = sample.spectrum_NIST( lower_limit = ll, upper_limit = ul, d_lambda=0.01, 
                                                       electron_density=n_e_ref, electron_temperature=Tp,
                                                       max_ion_state=2)
    wl=wl[0]
    
    spec_total = specs.sum(axis=0)
    sg = signal_spec(folder,filename)
    sg.wavelengths = [wl]
    sg.spectrum = [spec_total]
    sg.is_multi_channel = False
    sg.spectrometer_labels = ['1']
    
    list_lines =[]
    list_elements = []
    for al in sample.list_of_elements:
        l_num, lines = al.get_most_relevant_lines_ion_state(ion_state = 1, electron_temperature=Tp, n_lines = 5,lower_limit = ll, upper_limit = ul)
        l_num2, lines2 = al.get_most_relevant_lines_ion_state(ion_state = 2, electron_temperature=Tp, n_lines = 5,lower_limit = ll, upper_limit = ul)
        #if al.label == 'Sn':
            #lines2 = np.delete(lines2,3)
        list_lines.append([lines,lines2])
        list_elements.append(al)
        
    
    
    temperatures, temp_95, r2, y_s, x_s, intercepts = sg.multi_element_saha_boltzmann_temperature(list_lines, 
                        list_elements, electron_density=n_e_ref, ratio_of_maximum = 0.2, radius = 0.1, Plot = True, Plotlines = True, use_max_intensity = False)
    
    temperature=temperatures[0]
    line_norm=lines[1]
    
    list_predicted_concentration=[]
    
    temperature=T_ref
    concentration0 = np.exp(intercepts[-1])*partition_function(list_elements[-1].label,str(0),temperature/T_ref)/n_ion[-1][0]
    for i in range(0, len(intercepts)):
        concentration = np.exp(intercepts[i])*partition_function(list_elements[i].label,str(0),temperature/T_ref)/n_ion[i][0]/concentration0
        list_predicted_concentration.append([list_elements[i].label,concentration,list_elements[i].ratio,r2[i]])
        
    print(list_predicted_concentration)
        
    
    
    print(sg.LTE_MW_criterium(3.2,temperature))


#n_ion,temperature=func()

#sample = digital_twin([['Fe',1.]])
#wl1, intensities, labels, n_ions, specs = sample.spectrum_NIST(electron_temperature=1*T_ref, Plot=True, Map = True)


def test_1():
    
    al = element('Fe')
    ll = 200.
    ul = 800.
    specs = []
    specs_sum = []
    temps = []
    for fx in arange(0.2,2,0.1):
        Tp = fx*T_ref
        temps.append(fx)
        wl,spec,label, n_ion = al.spectrum_NIST(lower_limit = ll, upper_limit = ul, electron_temperature=Tp,max_ion_state=4,d_lambda=0.1)
        specs_sum.append(spec.sum(axis=0))
        specs.append(spec)
    
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PolyCollection
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    
    
    fig = plt.figure(figsize=[10,10])
    ax = fig.gca(projection='3d')
    
    ind_min=0
    ind_max=-1
    xs = wl[ind_min:ind_max]
    
    samples_3d_plot=array(specs_sum)
    verts = []
    zs = array(temps)
    xs_1=[]
    ys_1=[]
    zs_1=[]
    for i in range(0,len(zs)):
        print(zs[i])
        ys = samples_3d_plot[i][ind_min:ind_max]
        ys[0], ys[-1] = 0, 0
        verts.append(list(zip(xs, ys)))
        
        xs_1.append(xs)
        ys_1.append(ones(xs.shape)*zs[i])
        zs_1.append(samples_3d_plot[i][ind_min:ind_max])
        
        
    poly = PolyCollection(verts)
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    
    for i in range(0,len(zs)):
        ax.plot(xs_1[i],ys_1[i],zs_1[i],'k-')
    
    
    ax.set_xlabel('$\lambda$ (nm)',fontsize=16)
    ax.set_xlim3d(wl[0],wl[-1])
    ax.set_ylabel('Plasma temperature $T_{p}$ (ev)',fontsize=16)
    #ax.set_ylim3d(0, 110.)#1.)
    ax.set_zlabel('Counts',fontsize=16)
    #ax.set_zlim3d(0, 1500)#25000)
    #ax.set_zticks([0,12500,25000])
    
    #ax.view_init(40, -40)
    #ax.grid(False)
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    #ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    
    figure()
    imshow((np.log10(abs(0.001+array(specs_sum)))>1)*(np.log10(abs(0.001+array(specs_sum)))),
           origin='lower',extent=[min(wl),max(wl),0,max(temps)],cmap='inferno_r',
           aspect='auto',interpolation='none')
    xlabel('Wavelength (nm)')
    ylabel(r'$T_{plasma}$')
    colorbar()
    
#test_1()
