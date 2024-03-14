# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:38:07 2021

@author: nunoa
"""
from numpy import *
import os
import pandas as pd

def read_data(filename):
    
    """
    For a given spectral datafile the function returns 4 arrays constaining the signal
    
    Parameters
        filename: str
            The path to the file from spectrometer
    
    Returns
    
        Wavelength: 1d array
            Array with the wavelengths in (nm)
        Sample: 1d array
            Array with the signal in (counts)
        Dark: 1d array
            Array with the dark in (counts)
        Reference: 1d array
            Array with the reference signal in (counts)
        
    
    """
    
    fl=open(filename,'r')
    wave=[]
    sample=[]
    dark=[]
    reference=[]
    
    #skips header
    for i in range(0,8):
        if i !=1:
            fl.readline()
        else:
            line = fl.readline()
            try:
                integration_time = float(line.split(':')[1][:-1].split(' ')[-1].replace(',','.'))
            except:
                integration_time = None
    
    line=fl.readline()
    
    #for all lines now
    while len(line)>1:
        #print "nl" + line
        line=line.replace(",",".")
        data=line.split(";")
        wave.append(float(data[0]))
        sample.append(float(data[1]))
        dark.append(float(data[2]))
        reference.append(float(data[3]))
        line=fl.readline()
    
    fl.close()
        
    return array(wave),array(sample),array(dark),array(reference),False,['o'],0, integration_time


def read_position(filename):
    
    """
    For a given spectral datafile the function returns 4 arrays constaining the signal
    
    Parameters
        filename: str
            The path to the file from spectrometer
    
    Returns
    
        position: list [x_pos, y_pos]
    
    """
    
    fl=open(filename,'r')
    line = fl.readline()
    x_pos = 0
    y_pos = 0
    try:
        x_pos = float(line.split('\t')[0])
        y_pos = float(line.split('\t')[1])
    except:
        None
        
    fl.close()
        
    return [x_pos,y_pos]


def read_signal_from_file(folder, filename):
    
    """
    Function that reads signal (single or multi channel spectrometers) from a given filename 
    
    Parameters:
        folder: str
            path where file is located
        filename: str
            filename to read
    
    Returns:
        
        wavelength: list of 1d arrays 
            Contain the wavelengths for which the spectrum are obtained
        spectrum: list of 1d arrays 
            Contain the specturm
        dark: list of 1d arrays 
            Contain the dark channel information
        reference: list of 1d arrays 
            Contain the reference channel information
        multi_channel: bool 
            True if it is multi_channel
        
    """
    
    
    
    #find if file has multiple channels
    
    list_folder = array([array([d.split("_")[0],d.split("_")[1],
                                d.split("_")[2]]) for d in os.listdir(folder) if d.endswith(".TXT")])
    
    shot_number = filename.split("_")[1]
    
    filename= folder + filename
    
    if len(where(unique(list_folder[:,1])==shot_number)[0])!=0:
        multi_channel = True
    
    wavelength = []
    spectrum = []
    dark = []
    reference = []
    spectrometer_labels = []
    integration_time = None
    
    for spectrometer in unique(list_folder[:,0]):
        #current filename to read
        current_file = folder + spectrometer +"_"+ shot_number+"_"+  spectrometer + ".TXT"
        
        try:
            

            
            #read data from the current file
            wavelength_current, spectrum_current, dark_current, reference_current,trash,trash1,trash2, integration_time = read_data(current_file)
            
            #append to data lists
            wavelength.append(wavelength_current)
            spectrum.append(spectrum_current)
            dark.append(dark_current)
            reference.append(reference_current)
            spectrometer_labels.append(spectrometer)
        
        #if for some reason the file does not exist
        except:
            print("Warning - Skipped file " + current_file + ", does not exists" )
    
    return wavelength, spectrum, dark, reference, multi_channel, spectrometer_labels, shot_number, integration_time





def read_folder(folder,ignore=None):
    
    """
    Function that reads signal (single or multi channel spectrometers) from a given filename 
    
    Parameters:
        folder: str
            path to the folder to read
    
    Returns:
        
        list_of_signals:
        
    """
    
    
    
    #find if file has multiple channels
    
    list_folder = array([array([d.split("_")[0],d.split("_")[1],
                                d.split("_")[2]]) for d in os.listdir(folder) if d.endswith(".TXT") or d.endswith(".txt")])
    
    shots = unique(list_folder[:,1])
    list_of_signals = []
    
    for i in range(0, len(shots)):
        
        shot_number = shots[i]
        if shot_number != ignore:
            if len(where(unique(list_folder[:,1])==shot_number)[0])!=0:
                multi_channel = True
            
            wavelength = []
            spectrum = []
            dark = []
            reference = []
            spectrometer_labels = []
            integration_time = None
            
            for spectrometer in unique(list_folder[:,0]):
                #current filename to read
                current_file = folder + spectrometer +"_"+ shot_number+"_"+  spectrometer + ".TXT"
                
                try:
                    
        
                    
                    #read data from the current file
                    wavelength_current, spectrum_current, dark_current, reference_current,trash,trash1,trash2, integration_time = read_data(current_file)
                    
                    #append to data lists
                    wavelength.append(wavelength_current)
                    spectrum.append(spectrum_current)
                    dark.append(dark_current)
                    reference.append(reference_current)
                    spectrometer_labels.append(spectrometer)
                
                #if for some reason the file does not exist
                except:
                    print("Warning - Skipped file " + current_file + ", does not exists" )
            
            list_of_signals.append([wavelength, spectrum, dark, reference, 
                                    multi_channel, spectrometer_labels, shot_number, integration_time])
    return list_of_signals


def get_position_signal(folder,ignore=None):
    
    """
    Function that signal poisition
        
    """
    
    
    
    #find if file has multiple channels
    
    list_folder = array([array([d.split("_")[0],d.split("_")[1],
                                d.split("_")[2]]) for d in os.listdir(folder) if d.endswith(".TXT") or d.endswith(".txt")])
    
    shots = unique(list_folder[:,1])
    positions = []
    
    for i in range(0, len(shots)):
        
        shot_number = shots[i]
        if shot_number != ignore:
            if len(where(unique(list_folder[:,1])==shot_number)[0])!=0:
                multi_channel = True
            
            for spectrometer in unique(list_folder[:,0][0]):
                #current filename to read
                current_file = folder + spectrometer +"_"+ shot_number+"_"+  spectrometer + ".TXT"
                
                #try:
                    
        
                    
                    #read data from the current file
                current_pos = read_position(current_file)
                    
                positions.append(current_pos)
                
                #if for some reason the file does not exist
                #except:
                #    print("Warning - Skipped file " + current_file + ", does not exists" )

                
    return positions

#folder = "D:\\Data_LIBS\\Miguel\\Exp Pulse\\2021-05-31\\Target Fe 99,95% LOT005133\\50 ns\\"
#folder = "C:\\Users\\nunoa\\LIBS - CORE\\Nova pasta\\LIBS-Drivers\\Map\\"
#signals = get_position_signal(folder)


#######################################################################################################
#######################################################################################################
###################################SCI APS routines####################################################
#######################################################################################################
#######################################################################################################

def read_signal_from_file_sciaps(folder, filename):
    
    
    #find if file has multiple channels
    
    list_folder = [d for d in os.listdir(folder) if d.endswith(".csv")]
    
    shot_number = 0
    list_of_signals = []
    multi_channel = False
    
        
           
    wavelength = []
    spectrum = []
    dark = []
    reference = []
    spectrometer_labels = []
    integration_time = None
            
    for current_file in list_folder:

        #read data from the current file
        wavelength_current, spectrum_current, dark_current, reference_current,trash,trash1,trash2, integration_time = read_data_sciaps(folder+current_file)
                    
        #append to data lists
        wavelength.append(wavelength_current)
        spectrum.append(spectrum_current)
        dark.append(dark_current)
        reference.append(reference_current)
        spectrometer_labels.append("SCIAPS")
                
    
    return wavelength, spectrum, dark, reference, multi_channel, spectrometer_labels, shot_number, integration_time





def read_folder_sciaps(folder,ignore=None):
    
    
    #find if file has multiple channels
    
    list_folder = [d for d in os.listdir(folder) if d.endswith(".csv")]
    
    shot_number = 0
    list_of_signals = []
    multi_channel = False
    
        
           
    wavelength = []
    spectrum = []
    dark = []
    reference = []
    spectrometer_labels = []
    integration_time = None
            
    for current_file in list_folder:

        #read data from the current file
        wavelength_current, spectrum_current, dark_current, reference_current,trash,trash1,trash2, integration_time = read_data_sciaps(folder+current_file)
                    
        #append to data lists
        wavelength.append(wavelength_current)
        spectrum.append(spectrum_current)
        dark.append(dark_current)
        reference.append(reference_current)
        spectrometer_labels.append("SCIAPS")
                
            
        list_of_signals.append([wavelength, spectrum, dark, reference, 
                                    multi_channel, spectrometer_labels, shot_number, integration_time])
    
    return  list_of_signals


def read_data_sciaps(filename):
    
    """
    For a given spectral datafile the function returns 4 arrays constaining the signal
    
    Parameters
        filename: str
            The path to the file from spectrometer
    
    Returns
    
        Wavelength: 1d array
            Array with the wavelengths in (nm)
        Sample: 1d array
            Array with the signal in (counts)
        Dark: 1d array
            Array with the dark in (counts)
        Reference: 1d array
            Array with the reference signal in (counts)
        
    
    """
    
    df=pd.read_csv(filename)
    wave=df['wavelength'].to_numpy()
    sample=df['intensity'].to_numpy()   
    dark=zeros(wave.shape)
    reference=zeros(wave.shape)
    integration_time=1
        
    return wave,sample,dark,reference,False,['o'],0, integration_time

#folder = "D:\\Data_LIBS\\Furo PNR RC 005 - SciAps\\Converted\\DRI0000001\\"
#signals = read_signal_from_file_sciaps(folder,'')

#######################################################################################################
#######################################################################################################


#folder = "D:\\Data_LIBS\\Furo\\Furo\\DRI0000052\\340 Spot 1\\"
#signals = read_folder(folder)

#filename = "1703275U8_0002_1703275U8.TXT"
#wavelength, spectrum, dark, reference, multi_channel, spectrometer_labels, shot, integration_time = read_signal_from_file(folder, filename)