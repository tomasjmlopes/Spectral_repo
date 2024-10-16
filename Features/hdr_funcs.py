import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import laplace
from skimage.morphology import disk
from skimage.filters import rank
from scipy.ndimage import convolve

from scipy.fftpack import fft2, ifft2
from skimage.transform import resize

from tqdm import tqdm

KERNEL = (1.0/256) * np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])

def normalize_spectral_images(data, use_gaussian=False, sigma=0.5, batch_norm = True):
    """
    Normalizes spectral images using a min-max scaling scheme with optional Gaussian filtering.

    Parameters:
        data (np.ndarray): The input spectral image data.
        use_gaussian (bool): If True, apply Gaussian filter to the data.
        sigma (float): The sigma value for Gaussian filtering, used if use_gaussian is True.

    Returns:
        np.ndarray: The normalized spectral images.
    """
    normalized_data = np.zeros_like(data)
    if use_gaussian:
        filtered_data = gaussian_filter(data, sigma=sigma, axes = (1, 2))
    else:
        filtered_data = data

    if batch_norm:
        data_min = filtered_data.min()
        data_max = filtered_data.max()
        
        for wv in tqdm(range(data.shape[0]), desc="Normalizing images"):
            if data_max > data_min:
                normalized_data[wv] = (filtered_data[wv] - data_min) / (data_max - data_min)
            else:
                normalized_data[wv] = 0
    else: 
        for wv in tqdm(range(data.shape[0]), desc="Normalizing images"):  
            data_min = filtered_data[wv].min()
            data_max = filtered_data[wv].max()
            
            if data_max > data_min:
                normalized_data[wv] = (filtered_data[wv] - data_min) / (data_max - data_min)
            else:
                normalized_data[wv] = 0

    return normalized_data

def assemble_hdr(labels, dataset, use_gaussian=False, sigma=0.5, batch_norm=True):
    """
    Assembles a high dynamic range (HDR) dataset from labeled data, applying normalization to each unique label group.

    Parameters:
        labels (np.ndarray): An array of labels corresponding to each entry in the dataset.
        dataset (np.ndarray): The dataset to process, where each entry corresponds to a label in 'labels'.
        use_gaussian (bool): If True, apply Gaussian filter during normalization.
        sigma (float): The sigma value for Gaussian filtering, used if use_gaussian is True.

    Returns:
        np.ndarray: An array of HDR processed data groups, each corresponding to a unique label.
    """
    unique_labels = np.unique(labels)
    hdr_dataset = [normalize_spectral_images(np.array(dataset[labels == label]), use_gaussian=use_gaussian, sigma=sigma, batch_norm=batch_norm)
                   for label in unique_labels]
    
    return np.array(hdr_dataset, dtype=object)

def gauss(x, sigma, val):
    """ Gaussian function for calculating exposedness. """
    return np.exp(-((x - val) ** 2) / (2 * sigma ** 2))

def saliency_map(img):
    """
    Calculate a simple saliency map based on local contrast.
    """

    local_mean = gaussian_filter(img, sigma = 30)
    local_contrast = np.abs(img - local_mean)
    saliency = (local_contrast - local_contrast.min()) / (local_contrast.max() - local_contrast.min() + 1e-8)
    
    return saliency

def calculate_weights(hdr_dataset, sigma=0.2, ideal_val=0.5, beta=0, weight_exp = 1, weight_sharp = 1, weight_saliency = 1):
    """
    Calculate weights for HDR assembly based on various metrics.
    This function now handles hdr_dataset that may include object arrays.
    """
    weights_norm = []
    for data in hdr_dataset:
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = np.stack(data).astype(float)

        # Ideal value for vision is 0.5. However, by introducing the beta
        # parameter, we can balance this ideal value with the average intensity
        # values of the image.
        # https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.846580/full
        avg_int = np.mean(data, axis = 0)
        mu = (1 - beta) * ideal_val + beta * avg_int

        w_exposedness = gauss(data, sigma, mu)
        metric_sharpness = np.array([abs(laplace(data[i])) for i in range(data.shape[0])])
        metric_saliency = np.array([saliency_map(data[i]) for i in range(data.shape[0])])

        weights = w_exposedness**weight_exp * metric_sharpness**weight_sharp * metric_saliency**weight_saliency

        w_sum = np.sum(weights, axis = 0, keepdims = True)
        weights_each = weights / w_sum
        
        weights_norm.append(weights_each)

    return np.array(weights_norm, dtype = object)

def process_type(scaled_pro_type, weights_norm_type, n_layers, x_padding, y_padding):
    def add_padding(data):
        return np.pad(data, ((x_padding, x_padding), (y_padding, y_padding)), mode='mean')

    def downscale(img):
        return convolve(img, KERNEL, mode='constant')[::2, ::2]

    def upscale(img):
        img_up = np.zeros((2*img.shape[0], 2*img.shape[1]))
        img_up[::2, ::2] = img
        return convolve(img_up, 4*KERNEL, mode='constant')

    def create_pyramid(img, is_laplacian=True):
        pyramid = []
        for _ in range(n_layers):
            down = downscale(img)
            if is_laplacian:
                up = upscale(down)
                pyramid.append(img - up)
            else:
                pyramid.append(img)
            img = down
        pyramid.append(down)
        return pyramid

    n_wavelengths = len(scaled_pro_type)
    laplacian_pyramids = []
    gaussian_pyramids = []

    for w in range(n_wavelengths):
        image_p = add_padding(scaled_pro_type[w])
        weights_p = add_padding(weights_norm_type[w])

        laplacian_pyramids.append(create_pyramid(image_p))
        gaussian_pyramids.append(create_pyramid(weights_p, is_laplacian=False))

    # Merge pyramids
    merged = []
    for l_ps, g_ps in zip(zip(*laplacian_pyramids), zip(*gaussian_pyramids)):
        merged.append(sum(l_p * g_p for l_p, g_p in zip(l_ps, g_ps)))

    # Reconstruct image
    result = merged[-1]
    for layer in merged[-2::-1]:
        result = upscale(result)[:layer.shape[0], :layer.shape[1]] + layer

    return result[x_padding:-x_padding, y_padding:-y_padding]

def hdr_processing(scaled_pro, weights_norm, n_layers, x_padding, y_padding):
    n_types = len(scaled_pro)
    output_shape = scaled_pro[0][0].shape
    new_data = np.zeros((n_types, *output_shape))

    for i in tqdm(range(n_types)):
        new_data[i] = process_type(scaled_pro[i], weights_norm[i], n_layers, x_padding, y_padding)

    return new_data