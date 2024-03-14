# -*- coding: utf-8 -*-
"""
digital_twin.py
    Implements the class digital_twin to store a list of elements and 
    concentrations to construct a digital twin of the sample, 
    generating the expected spectrums and obtaining the relevant lines
"""
    
__author__ = "Nuno Azevedo Silva"
__mantainer__ = "Nuno Azevedo Silva"
__email__ = "nunoazevedosilva@gmail.com"

import numpy as np
from scipy import *
from matplotlib.pyplot import *
from Data_Core.fundamental_constants import *
from Data_Core.scrap_databases import *
from Data_Core.line_shapes import *
from Data_Core.partition_function import *
from Data_Core.element import *

class digital_twin:
    """
    
    Class that stores a list of elements and concentrations to construct a digital twin of the sample, 
    generating the expected spectrums and obtaining the relevant lines
    
    Attributes
    
    
    Methods
        
        
    
    """
    
    
    def __init__(self, elements , wavelenghts = None):
        
        self.list_of_elements = []
        
        for i in range(0,len(elements)):
            ##
            current_element = element(elements[i][0],elements[i][1])
            self.list_of_elements.append(current_element)
            ##
        
        self.wavelenghts = wavelenghts
    
        
    def __repr__(self):
        
        return 'Digital sample with \n'+ '\n'.join([str(elem) for elem in self.list_of_elements])
    
            
    def __str__(self):
        
        return 'Digital sample with \n'+ '\n'.join([str(elem) for elem in self.list_of_elements])
    
    
    def spectrum_NIST(self, max_ion_state=3,wl=[], lower_limit = 200., upper_limit =900., d_lambda = .1,
                      electron_density = n_e_ref, electron_temperature=T_ref, 
                      params_voigt=[],resolution=3000,
                      normalize = False, Plot = False, Map = False):
        
        wl1 = []
        intensities = []
        labels =[]
        n_ions = []
        
        for i in range(0, len(self.list_of_elements)):
            
            wavelength, intensity, label, n_ion = self.list_of_elements[i].spectrum_NIST(max_ion_state,wl, lower_limit, upper_limit, d_lambda,electron_density, electron_temperature,params_voigt, resolution,normalize)
            wl1.append(wavelength)
            intensities.append(intensity)
            labels.append(label)
            n_ions.append(n_ion)
        
        specs=[]
        for i in range(0,len(intensities)):
            specs.append(intensities[i].sum(axis=0))
        specs=np.array(specs)
        
        intensities=np.array(intensities)
        #if you want to plot
        if Plot == True:
            
            figure()
            for i in range(0, len(labels)):
                for j in range(0,len(labels[i])):
                    print(labels[i][j])
                    fill_between(wl1[i], np.zeros(intensities[i][j].shape), intensities[i][j], label = labels[i][j] + " (" + str(n_ions[i][j]*100*self.list_of_elements[i].ratio)[:5] +"%)", lw=2., ls='-')
            
            plot(wavelength, intensities.sum(axis = 0).sum(axis = 0), label = "Sum", color = 'k', lw =1., ls=':')
            grid()
            legend()
            xlabel('Wavelength (nm)')
            ylabel('Intensity (arb.un.)')
            
        if Map == True:
            figure()
            eps = 1e-4
            imshow((np.log10(abs(eps+specs))>-5)*(np.log10(abs(eps+specs))),origin='lower',aspect='auto',extent=[min(wl1[0]),max(wl1[0]),0,len(self.list_of_elements)],cmap='inferno_r', interpolation='none')
            xlabel('Wavelength (nm)')
            ylabel('Element')
            lbb=[]
            yticks(np.arange(0,len(labels))+0.5,[el.label for el in self.list_of_elements])
            ax=gca()
            for i in range(0,len(self.list_of_elements)):
                plot([min(wl1[0]),max(wl1[0])],[i+1,i+1],'-',color='k',lw=3)
            colorbar()
            
            
        
            return wl1, intensities, labels, n_ions, specs
   
        return wl1, intensities, labels, n_ions,specs
    
    def spectrum_Kurucz(self, max_ion_state=3,wl=None, lower_limit = 200., upper_limit =900., d_lambda = .1,
                      electron_density = n_e_ref, electron_temperature=T_ref, normalize = False):
        
        return "Not available yet"
    
    def get_most_relevant_lines():
        return 0
    
    

#sample = digital_twin([['Al',.3],["Cu",0.3],["W",0.1],['H',0.1],['Li',0.2]])
#sample = digital_twin([['P',.2],["Cr",0.2],["Mn",0.1],['As',0.1],['Sn',0.1],['Sb',.1],['W',.2]])
#wl1, intensities, labels, n_ions, specs = sample.spectrum_NIST(electron_temperature=1*T_ref, Plot=True, Map = True)
#intensities = sample.spectrum_NIST(Plot=True, Map = True)