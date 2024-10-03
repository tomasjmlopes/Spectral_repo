import os
import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt

from typing import List, Optional
from numpy.lib.stride_tricks import sliding_window_view


class LibsLoader:
    """
    Python Toolkit designed to handle LIBS datasets.

    This class provides tools for loading data and basic preprocessing of spectral data
    (normalization and baseline removal).

    Attributes:
        fname (str): The filename of the LIBS dataset.
        config (dict): Configuration parameters.
        dataset (np.ndarray): The loaded LIBS dataset.
        wavelengths (np.ndarray): The wavelengths corresponding to the spectral dimension.
        positions (np.ndarray): The positions of each spectrum.
        x_size (int): The size of the x dimension.
        y_size (int): The size of the y dimension.
        spectral_size (int): The size of the spectral dimension.
    """

    def __init__(self, fname: str, config_file: Optional[str] = None):
        if not os.path.exists(fname):
            raise FileNotFoundError(f"The file {fname} does not exist.")
        self.fname = fname
        self.config = self._load_config(config_file)
        self.dataset = None
        self.wavelengths = None
        self.positions = None
        self.x_size = None
        self.y_size = None
        self.spectral_size = None

    def _load_config(self, config_file: Optional[str]) -> dict:
        if config_file is None:
            return {'resolution': 0.5}
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def load_dataset(self, init_wv: Optional[int] = None, final_wv: Optional[int] = None,
                     baseline_corrected: bool = True, return_pos: bool = False) -> None:
        """
        Loads the dataset from the file, optionally focusing on a specified wavelength range.

        Args:
            init_wv (int, optional): Initial wavelength index.
            final_wv (int, optional): Final wavelength index.
            baseline_corrected (bool): If True, applies baseline correction.
            return_pos (bool): If True, loads and stores spatial positions of spectra.
        """
        try:
            with h5py.File(self.fname, 'r') as hf:
                sample = list(hf.keys())[0].split(' ')[-1]
                baseline = 'Pro' if baseline_corrected else "raw_spectrum"

                spectrums = [np.array(hf[f'Sample_ID: {sample}/Spot_{i}/Shot_0/{baseline}']) for i in range(len(hf[f'Sample_ID: {sample}']))]
                positions = [np.array(hf[f'Sample_ID: {sample}/Spot_{i}/position']) for i in range(len(hf[f'Sample_ID: {sample}']))]

                self.wavelengths = np.array(hf['System properties']['wavelengths']).flatten()

                if init_wv is not None and final_wv is not None:
                    spectrums = [s[init_wv:final_wv] for s in spectrums]
                    self.wavelengths = self.wavelengths[init_wv:final_wv]

                self.x_size = len(np.unique([p[1] for p in positions]))
                self.y_size = len(np.unique([p[0] for p in positions]))
                self.spectral_size = len(self.wavelengths)

                # Sort spectrums and positions
                sorted_indices = np.lexsort(([p[0] for p in positions], [p[1] for p in positions]))
                spectrums = [spectrums[i] for i in sorted_indices]
                positions = [positions[i] for i in sorted_indices]

                self.dataset = np.array(spectrums).reshape(self.x_size, self.y_size, self.spectral_size)
                if return_pos:
                    self.positions = np.array(positions)
        except Exception as e:
            raise IOError(f"Error loading dataset: {str(e)}")

    def wavelength_to_index(self, WoI: float) -> int:
        """
        Find index closest to Wavelength of Interest "WoI"

        Args:
            WoI (float): Wavelength of interest

        Returns:
            int: Index of closest wavelength
        """
        return np.argmin(np.abs(self.wavelengths - WoI))

    def normalize_to_sum(self) -> None:
        """
        Normalize each spectrum to its sum.
        """
        normalized = self.dataset / np.sum(self.dataset, axis=2)[:, :, np.newaxis]
        self.dataset = normalized

    def baseline_correct(self) -> None:
        """
        Subtracts the baselines from the spectra.
        """
        flat_spectra = self.dataset.reshape(-1, self.spectral_size)
        baselines = self._get_baseline(flat_spectra)
        baselines = baselines[:, :self.spectral_size]  # Align baselines with spectra
        corrected_spectra = (flat_spectra - baselines).reshape(self.x_size, self.y_size, self.spectral_size)
        self.dataset = corrected_spectra

    def plot_spectrum(self, x: int, y: int) -> None:
        """
        Plot the spectrum at the given x, y coordinates.

        Args:
            x (int): x-coordinate
            y (int): y-coordinate
        """
        spectrum = self.dataset[x, y, :]
        plt.figure(figsize=(10, 5))
        plt.plot(self.wavelengths, spectrum)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title(f'Spectrum at position ({x}, {y})')
        plt.show()

    def _get_baseline(self, dataset: np.ndarray, min_window_size: int = 50, smooth_window_size: Optional[int] = None) -> np.ndarray:
        """
        Calculate baseline using rolling window method.

        Args:
            dataset (np.ndarray): Input dataset
            min_window_size (int): Minimum window size for rolling minimum
            smooth_window_size (int, optional): Window size for smoothing

        Returns:
            np.ndarray: Calculated baselines
        """
        if smooth_window_size is None:
            smooth_window_size = 2 * min_window_size

        local_minima = self._rolling_min(
            arr=np.hstack(
                [dataset[:, 0][:, np.newaxis]] *
                ((min_window_size + smooth_window_size) // 2)
                + [dataset]
                + [dataset[:, -1][:, np.newaxis]] *
                ((min_window_size + smooth_window_size) // 2)
            ),
            window_width=min_window_size
        )
        return np.apply_along_axis(arr=local_minima, func1d=np.convolve, axis=1,
                                   v=self._get_smoothing_kernel(smooth_window_size), mode='valid')

    @staticmethod
    def _rolling_min(arr: np.ndarray, window_width: int) -> np.ndarray:
        """
        Calculates the moving minima in each row of the provided array.

        Args:
            arr (np.ndarray): Input array
            window_width (int): Width of the rolling window

        Returns:
            np.ndarray: Array of rolling minimums
        """
        window = sliding_window_view(arr, (window_width,), axis=len(arr.shape) - 1)
        return np.amin(window, axis=len(arr.shape))

    @staticmethod
    def _get_smoothing_kernel(window_width: int) -> np.ndarray:
        """
        Generates a Gaussian smoothing kernel of the desired width.

        Args:
            window_width (int): Width of the smoothing window

        Returns:
            np.ndarray: Gaussian smoothing kernel
        """
        kernel = np.arange(-window_width // 2, window_width // 2 + 1, 1)
        sigma = window_width // 4
        kernel = np.exp(-(kernel ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()
