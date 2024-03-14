# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:42:42 2021

@author: nunoa
"""



from scipy import *
from scipy import stats
from scipy.stats import t
from sklearn import *
import numpy as np 
from Data_Core.element import *
from Data_Core.digital_twin import *



def find_wavelength_index(center_wavelength,wavelengths):
    """
    Find the index corresponding to the wavelength in wavelenghts that is 
    the closest to a given center_wavelength
    """
    
    for i in range(0,len(wavelengths)-1):
        if center_wavelength >= wavelengths[i] and center_wavelength < wavelengths[i+1]:
            return i
    
    return None

def get_closest_peak_index(center_wavelength_peak, radius, wavelengths, spectrum):
    
    #find the index of the closest wavelength
    index_center = find_wavelength_index(center_wavelength_peak, wavelengths)
    
    #find the closest peak around a given radius
    index_right = find_wavelength_index(center_wavelength_peak+radius, wavelengths)
    index_left = find_wavelength_index(center_wavelength_peak-radius, wavelengths)
    
    if index_right != None and index_left != None:
    
        #get the max
        index_max = np.argmax(spectrum[index_left:index_right])+index_left
        
        return index_max
    
    #index right or left outside the wavelength range
    else:
        return None
    
    
    
def get_closest_peak_width_index(center_wavelength_peak_index,ratio_of_maximum, wavelengths, spectrum):
    
    peak_value = spectrum[center_wavelength_peak_index]
    
    threshold = ratio_of_maximum * peak_value
    
    left_found = False
    right_found = False
    
    #find the intersection at left
    i = center_wavelength_peak_index
    try:
        while spectrum[i]>threshold:
            i-=1
        index_left = i+1
    
        #find intersection at rigth
        i = center_wavelength_peak_index
        while spectrum[i]>threshold:
            i+=1
        index_right = i-1
        
        
        return [index_left,index_right]
    
    except:
        return None

def get_peak_area(center_wavelength, ratio_of_maximum, wavelengths, spectrum, radius = 1, Plot = False, Title = "", fig = None, subp =(1,1,1) ):
    
    #get closest peak index
    center_wavelength_peak_index = get_closest_peak_index(center_wavelength, radius, wavelengths, spectrum)
    
    if center_wavelength_peak_index == None:
        return None
    
    #get limits at which spectrum < ratio* peak value
    peak_limits = get_closest_peak_width_index(center_wavelength_peak_index,ratio_of_maximum, wavelengths, spectrum)
    
    if peak_limits == None:
        return None
    
    peak_value = spectrum[center_wavelength_peak_index]
    
    #correction at left
    x1=wavelengths[peak_limits[0]-1]
    y1=spectrum[peak_limits[0]-1]
    x3=wavelengths[peak_limits[0]]
    y3=spectrum[peak_limits[0]]
    
    b=(y1-x1/x3*y3)/(1-x1/x3)
    m=(y3-b)/x3
    
    y2_l=ratio_of_maximum*peak_value
    x2_l=(y2_l-b)/m
    dx2_l=x3-x2_l
    correction_area_left = 0.5*(y2_l+y3)*(dx2_l)
    
    #correction at right
    x1=wavelengths[peak_limits[1]]
    y1=spectrum[peak_limits[1]]
    x3=wavelengths[peak_limits[1]+1]
    y3=spectrum[peak_limits[1]+1]
    
    b=(y1-x1/x3*y3)/(1-x1/x3)
    m=(y3-b)/x3
    
    y2_r=ratio_of_maximum*peak_value
    x2_r=(y2_r-b)/m
    dx2_r=x3-x2_r
    correction_area_right = 0.5*(y2_r+y3)*(dx2_r)
    
    #wavelength interval for correctly weighted integration by trapezoidal rule
    dwavelengths = np.roll(wavelengths, -1) - wavelengths
    
    
    #plot if you desire to
    if Plot:
        
        
        cw = wavelengths[center_wavelength_peak_index]
        
        ax = subplot(subp[0],subp[1],subp[2])
        
        #plot signal
        fr = 80
        ax.set_title(Title)
        ax.plot(wavelengths[peak_limits[0]-fr:peak_limits[1]+fr],
                spectrum[peak_limits[0]-fr:peak_limits[1]+fr],
             ls='-',lw=0.5,color='k', label = 'Signal')
        
        #plot centerwavelenght, determined peak and radius
        ax.plot([center_wavelength,center_wavelength],[0,peak_value],ls='--',lw=0.5,alpha=0.5,color='r')
        ax.plot([center_wavelength-radius,center_wavelength-radius],[0,peak_value],ls='--',lw=0.5,alpha=1,color='r')
        ax.plot([center_wavelength+radius,center_wavelength+radius],[0,peak_value],ls='--',lw=0.5,alpha=1,color='r')
        ax.plot([cw,cw],[0,peak_value],ls=':',lw=1.0,alpha=0.5,color='k', label = 'Peak Found')
        
        ax.plot([center_wavelength-radius,center_wavelength+radius],[ratio_of_maximum*peak_value,ratio_of_maximum*peak_value],ls=':',lw=1.0,alpha=0.5,color='k', label = 'Peak Found')
        
        
        y=np.concatenate((np.array([y2_l]),spectrum[peak_limits[0]:peak_limits[1]+1],np.array([y2_r])))
        x=np.concatenate((np.array([x2_l]),wavelengths[peak_limits[0]:peak_limits[1]+1],np.array([x2_r])))
        
        ax.fill_between(x,y,
                        color='b',alpha=0.1,label='Peak Area')
        
        
        if subp[2]==subp[1]*subp[0]:
            ax.legend()
        

        
        return np.trapz(y,x)
    
    
    else:
        y=np.concatenate((np.array([y2_l]),spectrum[peak_limits[0]:peak_limits[1]+1],np.array([y2_r])))
        x=np.concatenate((np.array([x2_l]),wavelengths[peak_limits[0]:peak_limits[1]+1],np.array([x2_r])))
        
        return np.trapz(y,x)


def boltzmann_temperature(lines_ion, wavelengths, spectrum, ratio_of_maximum = 0.5, radius = 1, 
                          Plot = True, Title = "Boltzmann plot", Plotlines = False, use_max_intensity = False):
    #log(I/(gj*Aij))
    y_s = []
    
    #log(Ej)
    x_s = []
    

    
    
    for line_i in lines_ion:
        if use_max_intensity:
            index_max = get_closest_peak_index(line_i.ritz, radius, wavelengths, spectrum)
            if index_max != None:
                intensity = spectrum[index_max]
                y_s.append(np.log(intensity/(line_i.g_j*line_i.A_ji)))
                x_s.append(line_i.e_upper)
            else:
                print( " *** Warning - line " + str(line_i.ritz) +" for " + line_i.label +" " + str(int(line_i.ion_state)) + " not found within the given range")
        else:
            intensity = get_peak_area(line_i.ritz, ratio_of_maximum, wavelengths, spectrum, radius, Plotlines)
            if intensity != None:
                y_s.append(np.log(intensity/(line_i.g_j*line_i.A_ji)))
                x_s.append(line_i.e_upper)
            else:
                print( " *** Warning - line " + str(line_i.ritz) +" for " + line_i.label +" " + str(int(line_i.ion_state)) +  " not found within the given range")


    y_s=np.array(y_s)
    x_s=np.array(x_s)
    
    if len(x_s) == 0:
        return 'No lines found within the wavelenght range'
        
    
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
        
    if Plot:  
        fig,ax = subplots(figsize=(5,3),constrained_layout=True)
        
        ax.set_title(Title)
        ax.plot(x_s,y_s,'o', fillstyle = 'none',ls='none',color = 'k')
        

        
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
        
        ax.fill_between(xc,yc,yc2,ls='None',color='b',alpha=0.2)
        
        ax.set_xlabel(r"$E_i (ev)$")
        ax.set_ylabel(r"$log(I_{ij}/(g_i A_{ij}))$")
        
        stdev = np.sqrt(np.sum((regressor.predict(x_train)-y_train)**2) / (len(y_train) - 2))

        ax.text(0.6, 0.7,  '$r^2 =$' + "%0.4f" % r2 + '\n' + 
                r'$T_{plasma}=$'+ "%0.0f" % temperature + "$\pm$" + "%0.0f" % abs(temp_95) + " K"+
                "\n" + r'$T_{plasma}=$'+ "%0.2f" % tev + "$\pm$" + "%0.2f" % abs(temp_95_ev) + " ev",
                transform=ax.transAxes,fontsize=8, bbox={'boxstyle':'round', 'facecolor': 'wheat', 'alpha': 0.5, 'pad': 0.5})

        return temperature, temp_95, r2, y_s,x_s
        
    else:
        
        return temperature, temp_95, r2, y_s, x_s


def saha_boltzmann_temperature(lines, ion_energies , wavelengths, spectrum, ratio_of_maximum = 0.5, radius = 1, guess_temperature = T_ref,
                          Plot = True, electron_density=n_e_ref, Title = "Saha-Boltzmann plot", Plotlines = False, use_max_intensity = False):
    
    #log(I/(gj*Aij))
    y_s = []
    
    #log(Ej)
    x_s = []
    
    ##################################
    #auxiliary variables to plot
    
    #plasma_reduction_energy_factor
    energy_correction = plasma_reduction_energy_factor(electron_density, T_ref)

    
    plot_y = len(lines)
    plot_x = max([len(lines_ion) for lines_ion in lines])
    
    if Plotlines:
        fig1,ax = subplots(plot_y,plot_x,figsize=(plot_x*3,plot_y*3))
    else:
        fig1 = None
    ##################################
    
    ion_plot_lines=0
    for lines_ion in lines:
        
        index_plot=ion_plot_lines*plot_x+1
        
        for line_i in lines_ion:
            if use_max_intensity:
                index_max = get_closest_peak_index(line_i.ritz, radius, wavelengths, spectrum)
                if index_max != None:
                    intensity = spectrum[index_max]
                    y0 = np.log(intensity/(line_i.g_j*line_i.A_ji))-(line_i.ion_state
                                                                      -1)*np.log(2*(2*pi*m_e*kb_si*guess_temperature)**(3/2.)/h**3/electron_density)
                    y_s.append(y0)
            
                    energy = line_i.e_upper+ion_energies[0:int(line_i.ion_state)-1].sum()
                    x_s.append(energy-energy_correction)
                
                else:
                    print( " *** Warning - line " + str(line_i.ritz)+ " for " + line_i.label +" " + str(int(line_i.ion_state)) +  " not found within the given range")


            else:
                intensity = get_peak_area(line_i.ritz, ratio_of_maximum, wavelengths, spectrum, radius, Plotlines, 
                                          str(line_i.label + ' ' + str(int(line_i.ion_state))+ ' - ' + str(line_i.ritz)),
                                          fig = fig1, subp =[plot_y,plot_x,index_plot] )
                
                intensity=abs(intensity)
                
                if intensity != None:
                    #print(intensity)
                    
                    ####################
                    #y0 = np.log(intensity/(line_i.g_j*line_i.A_ji))-(line_i.ion_state
                     #                                                     -1)*np.log(2*(2*pi*m_e*kb_si*guess_temperature)**(3/2.)/h**3/electron_density)
                    
                    y0 = np.log(intensity*line_i.ritz/(line_i.g_j*line_i.A_ji))-(line_i.ion_state
                                                                          -1)*np.log(2*(2*pi*m_e*kb_si*guess_temperature)**(3/2.)/h**3/electron_density)
                    
                    ################
                    y_s.append(y0)
                
                    energy = line_i.e_upper+ion_energies[0:int(line_i.ion_state)-1].sum()
                    x_s.append(energy-energy_correction)
                else:
                    print( " *** Warning - line " + str(line_i.ritz)+ " for " + line_i.label +" " + str(int(line_i.ion_state)) +  " not found within the given range")
            
            index_plot += 1
            
        ion_plot_lines+=1
    
    y_s=np.array(y_s)
    x_s=np.array(x_s)
        
    
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
    
    if Plot:
        fig,ax = subplots(figsize=(5,3),constrained_layout=True)
        
        ax.set_title(Title)
        ax.plot(x_s,y_s,'o', fillstyle = 'none',ls='none',color = 'k')
        

        
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
        
        ax.fill_between(xc,yc,yc2,ls='None',color='b',alpha=0.2)
        
        ax.set_xlabel(r"$E_i (ev)$")
        ax.set_ylabel(r"$log(I_{ij}^{*}/(g_i A_{ij}))$")
        
        stdev = np.sqrt(np.sum((regressor.predict(x_train)-y_train)**2) / (len(y_train) - 2))

        ax.text(0.6, 0.7,  '$r^2 =$' + "%0.4f" % r2 + '\n' + 
                r'$T_{plasma}=$'+ "%0.0f" % temperature + "$\pm$" + "%0.0f" % abs(temp_95) + " K"+
                "\n" + r'$T_{plasma}=$'+ "%0.2f" % tev + "$\pm$" + "%0.2f" % abs(temp_95_ev) + " ev",
                transform=ax.transAxes,fontsize=8, bbox={'boxstyle':'round', 'facecolor': 'wheat', 'alpha': 0.5, 'pad': 0.5})

        return temperature, temp_95, r2, y_s,x_s
        
    else:
        
        return temperature, temp_95, r2, y_s, x_s
    

    
    

"""
al = element('Al')
l_num, lines = al.get_most_relevant_lines_ion_state(ion_state = 1, n_lines = 5,lower_limit = 200., upper_limit =400.)
wl,spec,label, n_ion = al.spectrum_NIST(lower_limit = 200., upper_limit =400., electron_temperature=T_ref,max_ion_state=4,d_lambda=0.1)
signal = spec[0]
signal = spec[0]+0*spec[1]
ion_energies = al.ion_energies
saha_boltzmann_temperature([lines], ion_energies, wl,signal, ratio_of_maximum = 0.5, radius = 0.1, Plot = True, 
                      Plotlines = True, use_max_intensity = False)
"""

"""
sample = digital_twin([['P',.2],["Al",0.2],["Mn",0.1],['As',0.1],['Sn',0.1],['Sb',.1],['W',.2]])

al = element('P')
l_num, lines = al.get_most_relevant_lines_ion_state(ion_state = 1, n_lines = 5)
l_num2, lines2 = al.get_most_relevant_lines_ion_state(ion_state = 2, n_lines = 5)
l_num3, lines3 = al.get_most_relevant_lines_ion_state(ion_state = 3, n_lines = 5)
ion_energies = al.ion_energies

wl,spec,label, n_ion, specs =sample.spectrum_NIST(electron_temperature=2*T_ref, lower_limit = 200., upper_limit =600.,max_ion_state=4,d_lambda=0.02, Map = True)
signal = specs[0]+specs[1]+specs[2]

saha_boltzmann_temperature([lines,lines2], ion_energies, wl[0], signal, ratio_of_maximum = 0.5, radius = 0.2, Plot = True, 
                      Plotlines = True, use_max_intensity = False)
"""

"""
sample = digital_twin([['P',.2],["Al",0.2],['As',0.1],['Sn',0.1],['Sb',.1],['W',.2],["Fe",0.1]])

al = element('Mn')
l_num, lines = al.get_most_relevant_lines_ion_state(ion_state = 1, n_lines = 5)
l_num2, lines2 = al.get_most_relevant_lines_ion_state(ion_state = 2, n_lines = 5)
l_num3, lines3 = al.get_most_relevant_lines_ion_state(ion_state = 3, n_lines = 5)
ion_energies = al.ion_energies

wl,spec,label, n_ion, specs =sample.spectrum_NIST(electron_temperature=2*T_ref, lower_limit = 200., upper_limit =600.,max_ion_state=4,d_lambda=0.1, Map = True)
signal = specs[0]+specs[1]+specs[2]

saha_boltzmann_temperature([lines,lines2], ion_energies, wl[0], signal, ratio_of_maximum = 0.9, radius = 0.2, Plot = True, 
                      Plotlines = True, use_max_intensity = False)
"""