import numpy as np

def fft_feature(signals, indexes_big, indexes_small):
    fft_map = np.fft.fftshift(np.fft.fft2(signals[:, :]))
    fft_map = np.array(fft_map)
    fft_map1 = np.array(fft_map)

    fft_map[indexes_small[1], indexes_small[0]] = 0
    fft_map[indexes_big[1], indexes_big[0]] = 0
    fft_map1[indexes_big[1], indexes_big[0]] = 0

    sum1 = np.sum(np.abs(fft_map), axis = (0, 1))
    max1 = np.sum(np.abs(fft_map1), axis = (0, 1))
    sums = sum1/max1

    return sums
    
def generate_mask(maps, size_big, size_small, freqs_x, freqs_y, dx):
    kxx, kyy = np.meshgrid(np.fft.fftshift(freqs_x), np.fft.fftshift(freqs_y))
    
    object_size_big = size_big*dx
    size_kspace_big = 2*np.pi/object_size_big
    filter_kspace_big = np.where((kxx**2 + kyy**2 < (size_kspace_big)**2))

    object_size_small = size_small*dx
    size_kspace_small = 2*np.pi/object_size_small
    filter_kspace_small = np.where((kxx**2 + kyy**2 > (size_kspace_small)**2))

    return filter_kspace_big, filter_kspace_small
    