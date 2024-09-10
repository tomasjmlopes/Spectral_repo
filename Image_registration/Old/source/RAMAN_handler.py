import os
from typing import List, Union, Optional, Tuple
from collections import defaultdict
import numpy as np
import h5py
import yaml
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple
import pandas as pd

class Raman_toolkit:
    """
    Python Toolkit designed to handle Raman datasets.

    This class provides tools for loading data and performing various manipulations,
    including feature extraction (manual and automatic), baseline removal, and normalization.

    Attributes:
        fname (str): The filename of the Raman dataset.
        config (dict): Configuration parameters.
        dataset (np.ndarray): The loaded Raman dataset.
        wavelengths (np.ndarray): The wavelengths corresponding to the spectral dimension.
        positions (np.ndarray): The positions of each spectrum.
        x_size (int): The size of the x dimension.
        y_size (int): The size of the y dimension.
        spectral_size (int): The size of the spectral dimension.
        features (np.ndarray): Extracted features.
        x_features (List[float]): Wavelengths of extracted features.
    """

    def __init__(self, fname: str, config_file: Optional[str] = None, overwrite: bool = False):
        if not os.path.exists(fname):
            raise FileNotFoundError(f"The file {fname} does not exist.")
        self.fname = fname
        self._overwrite = overwrite
        self.config = self._load_config(config_file)
        self.dataset = None
        self.wavelengths = None
        self.positions = None
        self.x_size = None
        self.y_size = None
        self.fft_metric = None
        self.ft_features = None
        self.int_features = None
        self.spectral_size = None
        self.features = None
        self.x_features = None
        self.ids_features = None
        self.classifier = None
        self.scaler = None
        self.debug = None

    def _load_config(self, config_file: Optional[str]) -> dict:
        if config_file is None:
            return {'resolution': 0.5}
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def load_dataset(self, init_wv: Optional[int] = None, final_wv: Optional[int] = None, 
                     baseline_corrected: bool = True) -> None:
        """
        Loads the dataset from the file, optionally focusing on a specified wavelength range.

        Args:
            init_wv (int, optional): Initial wavelength index.
            final_wv (int, optional): Final wavelength index.
            baseline_corrected (bool): If True, applies baseline correction.
            return_pos (bool): If True, loads and stores spatial positions of spectra.
        """
        try:
            with h5py.File(self.fname, 'a') as output_file:
                properties = output_file['properties']
                exp_properties = {'step_size' : np.array(properties['step_size'])[0],
                                'speed' : np.array(properties['speed']),
                                'n_points' : np.array(properties['n_points'])
                    }
                
                wavelengths = np.array(output_file['properties']['x_data'])
                
                
                spot_numbers = [int(s.split('_')[-1]) for s in list(output_file['data'].keys()) if 'spot' in s ]
                
                Nx,Ny = output_file['properties']['n_points'][0], output_file['properties']['n_points'][1]
                Nl = len(wavelengths)
                spectral_signal = np.zeros([Nx,Ny,Nl])
                
                for _i, spot_number in enumerate(spot_numbers):
                    ix, iy = int(spot_number//Ny), int(spot_number%Ny)                   
                    spot = 'spot_'+str(spot_number)
                    data = np.array(output_file['data'][spot]['raw_data'])                           
                    spectral_signal[ix,iy,:] = data

                self.dataset = spectral_signal
                self.wavelengths = wavelengths
                self.x_size, self.y_size, self.spectral_size = spectral_signal.shape
        except Exception as e:
            raise IOError(f"Error loading dataset: {str(e)}")

    def baseline_correct(self) -> np.ndarray:
        """
        Subtracts the baselines from the spectra.

        Returns:
            np.ndarray: Baseline-corrected dataset
        """
        flat_spectra = self.dataset.reshape(-1, self.spectral_size)
        baselines = np.zeros_like(flat_spectra)
        for pos, spec in enumerate(flat_spectra):
            print(f"ALS Baseline: {pos+1}/{flat_spectra.shape[0]}", end = '\r')
            baselines[pos] = self.baseline_als_optimized(spec, lam = 1e2, p = 1e-1)

        corrected_spectra = (flat_spectra - baselines).reshape(self.x_size, self.y_size, self.spectral_size)
        if self._overwrite:
            self.dataset = corrected_spectra
        else:
            return corrected_spectra
        
    def baseline_als_optimized(self, y, lam, p, niter = 10):
        L = len(y)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w) # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z
        
    def wavelength_to_index(self, WoI: float) -> int:
        """
        Find index closest to Wavelength of Interest "WoI"

        Args:
            WoI (float): Wavelength of interest

        Returns:
            int: Index of closest wavelength
        """
        return np.argmin(np.abs(self.wavelengths - WoI))

    def normalize_to_sum(self) -> np.ndarray:
        """
        Normalize each spectrum to its sum.

        Returns:
            np.ndarray: The normalized dataset.
        """
        normalized = self.dataset / np.sum(self.dataset, axis=2)[:,:,np.newaxis]
        if self._overwrite:
            self.dataset = normalized
        else:
            return normalized
        
    def manual_features(self, list_of_wavelengths: List[float], sigma: Optional[float] = None) -> np.ndarray:
        """
        Extract the wavelengths provided in list of wavelengths

        Args:
            list_of_wavelengths (List[float]): List of wavelengths to extract
            sigma (float, optional): Sigma for Gaussian filter. If None, no filtering is applied.

        Returns:
            np.ndarray: Extracted features
        """
        features = np.array([self.dataset[:, :, self.wavelength_to_index(wl)] for wl in list_of_wavelengths])
        if sigma is not None:
            features = np.array([gaussian_filter(f, sigma=sigma) for f in features])
        self.features = features
        self.x_features = list_of_wavelengths
        return features
    
    def _fft_features(self, smallest_dim_pixels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Automatically extract N features from dataset using the FT Feature Finder.

        Args:
            n_features (int): Number of features to extract
            smallest_dim_pixels (int): Smallest pixel dimension
            prominence (float or 'auto'): Prominence for peak finding. If 'auto', it's calculated from the data.
            sigma (float, optional): Sigma for Gaussian filter. If None, no filtering is applied.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Extracted features and feature metric
        """
        freqs_x = 2*np.pi*np.fft.fftfreq(self.y_size, self.config['resolution'])
        freqs_y = 2*np.pi*np.fft.fftfreq(self.x_size, self.config['resolution'])

        fft_map = np.array([np.fft.fftshift(np.fft.fft2(self.dataset[:, :, i])) for i in range(self.spectral_size)])
        fft_map[:, fft_map.shape[1]//2, fft_map.shape[2]//2] = 0  # Remove DC Component
    
        kxx, kyy = np.meshgrid(np.fft.fftshift(freqs_x), np.fft.fftshift(freqs_y))
    
        object_size_small = smallest_dim_pixels * self.config['resolution']
        size_kspace_small = np.pi / object_size_small
    
        R = np.sqrt(kxx**2 + kyy**2)

        sum1 = np.sum(np.abs(fft_map[:, (R < size_kspace_small)]), axis=(1))
        max1 = np.sum(np.abs(fft_map), axis=(1, 2))
    
        sums = np.divide(sum1, max1, out=np.zeros(sum1.shape, dtype=float))
        sums = np.nan_to_num((sums - np.nanmin(sums))/(np.nanmax(sums) - np.nanmin(sums)), nan=0.0)

        self.fft_metric = sums

    def extract_fft_features(self, n_features: int = 20, prominence: Union[float, str] = 'auto', pixel_dist = 5):
        if self.fft_metric is None:
            self.fft_metric = self._fft_features(smallest_dim_pixels = pixel_dist)
        if prominence == 'auto':
            prominence = np.mean(self.fft_metric) + np.std(self.fft_metric)

        inds, _ = find_peaks(self.fft_metric, distance=3, prominence=prominence)

        fft_wavelengths = self.wavelengths[inds]
        fft_wavelengths = fft_wavelengths[np.argsort(self.fft_metric[inds])[::-1][:n_features]]
        self.ft_features = fft_wavelengths

    def intensity_features(self, n_features: int = 20, prominence: Union[float, str] = 'auto'):
        mean_spec = self.calculate_average_spectrum()
        if prominence == 'auto':
            prominence = np.mean(mean_spec) + np.std(mean_spec)
        inds, _ = find_peaks(mean_spec, distance = 5, prominence = prominence)

        mean_intensities = mean_spec[inds]
        mean_features = self.wavelengths[inds]
        mean_features = mean_features[np.argsort(mean_intensities)[::-1][:n_features]]
        self.int_features = mean_features

    def automatic_feature_extraction(self, fft_features: int = 20, fft_prominence: float = 'auto', min_pixel_dist: float = 5, 
                                           intens_features: int = 20, int_prominence: float = 'auto', sigma: float = 0.5):

        if fft_features != 0:
            self.extract_fft_features(n_features = fft_features, prominence = fft_prominence, pixel_dist = min_pixel_dist)

        if (intens_features != 0) & (self.int_features is None):
            self.intensity_features(n_features = intens_features, prominence = int_prominence)

        tolerance = 0.25
        result_array = self.ft_features.copy() if not(self.fft_metric is None) else np.array([])
        to_check = self.int_features.copy() if not(self.int_features is None) else np.array([])
        for value in to_check:
            if np.all(np.abs(result_array - value) > tolerance):
                result_array = np.append(result_array, value)

        self.x_features = result_array
        self.features = self.manual_features(self.x_features, sigma = sigma)
    
    def standard_analysis(self, radius = 3, cmap = 'inferno'):
        mean_signal = np.mean(self.dataset, axis = (0, 1))
        min_signal = np.min(self.dataset, axis = (0, 1))
        max_signal = np.max(self.dataset, axis = (0, 1))


        fig = plt.figure(tight_layout = True, figsize = (10, 5))
        gs = gridspec.GridSpec(1, 2)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        x_center, y_center = self.y_size//2, self.x_size//2

        fig.suptitle('Raman Signal Analysis', fontsize = 20)

        #######################################################
        #                                                     #
        #       Plot Spectrum (Average, MinMax, Point)        #
        #                                                     #
        #######################################################

        axs = ax1
        axs.plot(self.wavelengths, mean_signal, lw = 2, ls = '-', color = 'lightblue', label = 'Mean')
        meanr, = axs.plot(self.wavelengths, self.dataset[x_center - radius:x_center + radius, y_center - radius:y_center + radius].mean(axis = (0, 1)),
                        color = 'darkblue',
                        label = 'Point Mean',
                        lw = 2)
        axs.fill_between(self.wavelengths, min_signal, max_signal, color = 'steelblue', alpha = 0.2)

        wn = 120
        line = axs.axvline(self.wavelengths[wn], lw = '1', alpha = 0.5, color = 'red', label = 'Mapped Wavenumber')
        axs.set_xlabel(r'Wavenumber $(cm^{-1})$')
        axs.set_ylabel(r'Intensity (arb.un.)')
        axs.legend(fancybox = True, shadow = True)
        axs.grid(False)

        #######################################################
        #                                                     #
        #  Spatial Distribuition of selected emission line    #
        #                                                     #
        #######################################################

        axs = ax2
        axs.set_title('Spatial Distribuition')
        spatial_dist = axs.imshow(self.dataset[:, :, wn], cmap = cmap, interpolation = 'gaussian')
        sca = axs.scatter(x_center, y_center, color = 'k', s = 40)
        axs.set_xlabel(r'$x(mm)$')
        axs.set_ylabel(r'$y(mm)$')
        axs.grid(False)

        #######################################################
        #                                                     #
        #             Functions for Interaction               #
        #                                                     #
        #######################################################

        def update_map(wn):
            spatial_dist.set_data(self.dataset[:, :, wn]) 
            spatial_dist.set_clim(vmin = self.dataset[:, :, wn].min(), vmax = self.dataset[:, :, wn].max())
            line.set_xdata(self.wavelengths[wn])

        def onclick(event):
            if event.dblclick:
                if event.inaxes == ax1:
                    ix, _ = event.xdata, event.ydata
                    wn = self.wavelength_to_index(ix)
                    update_map(wn)
                    fig.canvas.draw_idle()
                elif event.inaxes == ax2:
                    xx, yy = int(event.xdata), int(event.ydata)
                    sca.set_offsets([xx, yy])
                    data_region = self.dataset[yy - radius:yy + radius, xx - radius:xx + radius]
                    if data_region.shape == (2*radius, 2*radius, self.spectral_size) and radius != 0:
                        meanr.set_data(self.wavelengths, data_region.mean(axis = (0, 1)))
                    else:
                        meanr.set_data(self.wavelengths, self.dataset[yy, xx])
                    fig.canvas.draw_idle()
                
        cid = fig.canvas.mpl_connect('button_press_event', onclick)


        fig.tight_layout()