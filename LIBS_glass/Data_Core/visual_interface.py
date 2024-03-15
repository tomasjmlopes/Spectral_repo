from matplotlib.pyplot import *
import numpy as np
from matplotlib import gridspec

find_index = lambda x, wvls: np.argmin(np.abs(wvls - x), axis = 0)

def standard_analysis(spectrums, wavelengths, radius = 3):
    mean_signal = np.mean(spectrums, axis = (0, 1))
    min_signal = np.min(spectrums, axis = (0, 1))
    max_signal = np.max(spectrums, axis = (0, 1))


    fig = figure(tight_layout = True, figsize = (10, 5))
    gs = gridspec.GridSpec(1, 2)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    x_center, y_center = spectrums.shape[1]//2, spectrums.shape[0]//2

    fig.suptitle('LIBS Signal Analysis', fontsize = 20)

    #######################################################
    #                                                     #
    #       Plot Spectrum (Average, MinMax, Point)        #
    #                                                     #
    #######################################################

    axs = ax1
    axs.plot(wavelengths, mean_signal, lw = 2, ls = '-', color = 'lightblue', label = 'Mean')
    meanr, = axs.plot(wavelengths, spectrums[x_center - radius:x_center + radius, y_center - radius:y_center + radius].mean(axis = (0, 1)),
                    color = 'darkblue',
                    label = 'Point Mean',
                    lw = 2)
    axs.fill_between(wavelengths, min_signal, max_signal, color = 'steelblue', alpha = 0.2)

    wn = 120
    line = axs.axvline(wavelengths[wn], lw = '1', alpha = 0.5, color = 'red', label = 'Mapped Wavelength')
    axs.set_xlabel(r'Wavelength $(nm)$')
    axs.set_ylabel(r'Intensity (arb.un.)')
    axs.legend(fancybox = True, shadow = True)
    axs.grid(False)

    #######################################################
    #                                                     #
    #  Spatial Distribuition of selected emission line    #
    #                                                     #
    #######################################################

    axs = ax2
    axs.set_title('Sample')
    # axs.imshow(image_var, extent = (0, 90, 0, 80))
    # axs.imshow(spectrum[:, :, wn], cmap = my_cmap, extent = (0, 90, 0, 80), interpolation = 'gaussian')
    axs.imshow(spectrums[:, :, wn], cmap = 'gist_earth', interpolation = 'nearest')
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
        im1 = ax2.imshow(spectrums[:, :, wn], cmap = 'gist_earth', interpolation = 'nearest') 
        line.set_xdata(wavelengths[wn])

    def onclick(event):
        if event.dblclick:
            if event.inaxes == ax1:
                ix, iy = event.xdata, event.ydata
                wn = find_index(ix, wavelengths)
                update_map(wn)
                fig.canvas.draw_idle()
            elif event.inaxes == ax2:
                xx, yy = int(event.xdata), int(event.ydata)
                sca.set_offsets([xx, yy])
                data_region = spectrums[yy - radius:yy + radius, xx - radius:xx + radius]
                if data_region.shape == (2*radius, 2*radius, spectrums.shape[-1]) and radius != 0:
                    meanr.set_data(wavelengths, data_region.mean(axis = (0, 1)))
                    # print('Mean')
                else:
                    meanr.set_data(wavelengths, spectrums[yy, xx])
                    # print('Single Point')
                fig.canvas.draw_idle()
            
    cid = fig.canvas.mpl_connect('button_press_event', onclick)


    fig.tight_layout()