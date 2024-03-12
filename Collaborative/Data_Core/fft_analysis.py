import numpy as np
from scipy.signal import find_peaks
from matplotlib import *
from tqdm import tqdm
from Data_Core.experiment import *
from Data_Core.digital_twin import *

def fft_feature(signals, indexes1, indexes2):
    fft_map = np.fft.fftshift(np.fft.fft2(signals[:, :]))
    fft_map = np.array(fft_map)
    fft_map1 = np.array(fft_map)

    fft_map[indexes1[1], indexes1[0]] = 0
    fft_map[indexes2[1], indexes2[0]] = 0
    fft_map1[indexes2[1], indexes2[0]] = 0

    sum1 = np.sum(np.abs(fft_map), axis = (0, 1))
    max1 = np.sum(np.abs(fft_map1), axis = (0, 1))
    sums = sum1/max1

    return sums

def generate_mask(maps, radius1, radius2):
    x_size = maps.shape[0]
    y_size = maps.shape[1]

    ix = np.arange(0, x_size)
    iy = np.arange(0, y_size)

    XX, YY = np.meshgrid(ix, iy)

    center = [x_size/2, y_size/2]

    indexes1 = np.where(( (XX - center[0])**2 + (YY - center[1])**2 ) > radius1**2)
    indexes2 = np.where(( (XX - center[0])**2 + (YY - center[1])**2 ) < radius2**2)

    return indexes1, indexes2


# def find_element_index(wavelength, number = 4):
#     database = r'Data_Core/database/libs_data_Kurucz/db_total.txt'
#     dados = np.genfromtxt(database, usecols = (0, 4, 5), skip_header=1, dtype = str).T
#     ionization_level = np.where(dados[2, :] == 'I')
#     wls_database = np.array(dados[0, ionization_level], dtype = float)[0]
#     dif = np.abs(wavelength - wls_database)
#     element_database = dados[1, ionization_level][0]
#     idxs = np.argsort(dif)[0:number]
#     element = element_database[idxs]
#     # idx = np.abs(wavelength - wls_database).argmin()
#     return element

def find_element_index(wavelength, number = 4):
    database = r'Data_Core/database/'
    dados = np.genfromtxt(database + 'nist_data.csv',  usecols = (0, 1, 2), dtype = str, skip_header = 1).T
    ionization_level = np.where(dados[1, :] == '1')
    ionization_database = np.array(dados[:, ionization_level])
    wanted_elements = np.isin(ionization_database[0, 0, :], ['Li', 'Co', 'Al', 'Rb', 'O', 'Fe', 'Si', 'Cr', 'Cu', 'Na', 'K', 'Cl', 'Ca', 'Mg'])
    wls_database = np.array(ionization_database[:, 0, wanted_elements])

    dif = np.abs(wavelength - np.array(wls_database[2], dtype = float64))
    # element_database = wls_database[1, :]
    idxs = np.argsort(dif)[0: number]
    elements = wls_database[0, idxs]
    return elements

def analyse_fft(data_cube, wavelengths, n_wavelegnths = 10, inner_radius = 2, outer_radius = 40, vmax = 1):
    col = 0
    row = 0
    indexes1, indexes2 = generate_mask(data_cube, outer_radius, inner_radius)
    map_scores = np.array([fft_feature(data_cube[:, :, i], indexes1, indexes2) for i in tqdm(range(0, len(wavelengths)))])

    peaks, _ = find_peaks(map_scores, distance = 6, prominence = 0.05)
    score_peaks = map_scores[peaks]
    indexes_sorted = np.argsort(score_peaks)
    wls_of_interest = wavelengths[peaks][indexes_sorted][-n_wavelegnths:]
    lns = n_wavelegnths//5
    fig, ax = subplots(n_wavelegnths//5, 5, figsize = (15, lns*5))

    for i in wls_of_interest:
        element = find_element_index(i)
        if lns == 1:
            axs = ax[col]
        else:
            axs = ax[row, col]
        axs.imshow(data_cube[:, :, find_wavelength_index(i, wavelengths)], cmap = 'twilight', vmax = vmax*np.max(data_cube[:, :, find_wavelength_index(i, wavelengths)]))
        axs.set_title(element[0] + '/' + element[1] + '/' + element[2] + '/' + element[3] + ' - ' + str(np.round(i, 2)) + 'nm')
        axs.grid(False)
        col += 1
        if col == 5:
            col = 0
            row += 1
    fig.set_facecolor("white")
    tight_layout()
    return map_scores