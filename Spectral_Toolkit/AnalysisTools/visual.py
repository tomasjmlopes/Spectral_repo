import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go

from matplotlib import gridspec
from plotly.subplots import make_subplots

def dash_periodic_table(element_counts: dict, figsize: tuple = (1000, 500)):
    """
    Plots a periodic table using Plotly and for each element in element_counts, classifies
    it as present (3 or more emission lines found), yellow (between 1 and 3 
    emission lines found), and red (no lines found).
    Returns:
        go.Figure: Plotly figure object of the periodic table.
    """
    
    # Define groups and color mappings (unchanged)
    element_groups = {
        'Alkali Metals': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
        'Alkaline Earth Metals': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
        'Transition Metals': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                            'Rf', 'Db', 'Sg', 'Bh', 'Hs'],
        'Post-Transition Metals': ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'At'],
        'Metalloids': ['B', 'Si', 'Ge', 'As', 'Sb', 'Te'],
        'Reactive Non-Metals': ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I'],
        'Noble Gases': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'],
        'Lanthanides': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
        'Actinides': ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'],
        'Unknown Properties': ['Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    }

    element_to_group = {element: group for group, elements in element_groups.items() for element in elements}

    group_colors = {
        'Alkali Metals': 'rgb(173, 216, 230)',
        'Alkaline Earth Metals': 'rgb(255, 36, 0)',
        'Transition Metals': 'rgb(158, 67, 179)',
        'Post-Transition Metals': 'rgb(64, 130, 109)',
        'Metalloids': 'rgb(255, 204, 153)',
        'Reactive Non-Metals': 'rgb(0, 49, 83)',
        'Noble Gases': 'rgb(128, 0, 0)',
        'Lanthanides': 'rgb(0, 127, 255)',
        'Actinides': 'rgb(138, 51, 36)',
        'Unknown Properties': 'rgb(128, 128, 128)'
    }

    # Define the periodic table layout
    periodic_table = [
    ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],  # Empty row at the top
    ['H', '', '','', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
    ['Li', 'Be', '','', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
    ['Na', 'Mg', '','', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
    ['K', 'Ca', 'Sc','', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
    ['Rb', 'Sr', 'Y','', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
    ['Cs', 'Ba', 'La','', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
    ['Fr', 'Ra', 'Ac','', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'],
    ['', '', '', '','', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
    ['', '', '','Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', '', ''],
    ['', '', '','Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', '', '']
    ]

    fig = make_subplots(rows=1, cols=1)

    # Function to create a rounded rectangle path
    def rounded_rect(x, y, w, h, r):
        return f' M {x+r},{y} L {x+w-r},{y} Q {x+w},{y} {x+w},{y+r} L {x+w},{y+h-r} Q {x+w},{y+h} {x+w-r},{y+h} L {x+r},{y+h} Q {x},{y+h} {x},{y+h-r} L {x},{y+r} Q {x},{y} {x+r},{y} Z'

    shapes = []
    annotations = []

    for i, row in enumerate(periodic_table):
        for j, element in enumerate(row):
            if element:
                count = element_counts.get(element)
                group = element_to_group.get(element, 'Unknown')
                # print(element, count, group)

                if count is not None:
                    if count >= 3:
                        color = 'rgba(152, 251, 152, 1)'  # Light green
                        fontcolor = 'rgb(0, 100, 0)'  # Dark sage
                    elif 0 < count < 3:
                        color = 'rgba(255, 255, 224, 1)'  # Light yellow
                        fontcolor = 'rgb(184, 134, 11)'  # Dark yellow
                    else:
                        color = 'rgba(255, 192, 203, 1)'  # Light pink
                        fontcolor = 'rgb(139, 0, 0)'  # Dark red
                    opacity = 1
                    fontweight = 'bold'
                else:
                    color = group_colors.get(group, 'rgb(245, 245, 220)')  # Beige for unknown
                    fontcolor = 'black'
                    opacity = 0.3
                    fontweight = 'normal'

                # Add element box with rounded corners and spacing
                x, y = j + 0.05, -i - 0.95
                w, h = 0.9, 0.9
                r = 0.1  # Corner radius
                path = rounded_rect(x, y, w, h, r)

                shapes.append(
                    dict(
                        type='path',
                        path=path,
                        fillcolor=color,
                        line=dict(color='black', width=1),
                        opacity=opacity,
                        xref='x',
                        yref='y'
                    )
                )

                # Add element text
                annotations.append(
                    dict(
                        x=j+0.5,
                        y=-i-0.5,
                        text=element,
                        showarrow=False,
                        font=dict(
                            family="Arial, sans-serif",
                            size=12,
                            color=fontcolor,
                            weight=fontweight
                        ),
                        opacity=opacity, 
                        align="center",
                        xref='x',
                        yref='y'
                    )
                )

    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        shapes=shapes,
        annotations=annotations
    )

    # Adjust the axis ranges to fit the entire table, with extra space for Hydrogen
    fig.update_xaxes(range=[10, 10])  # Adding more space to the left for Hydrogen
    fig.update_yaxes(range=[-11, -1], scaleanchor="x", scaleratio=1)  # Adding more space at the top

    return fig

def wavelength_to_index(WoI: float, wavelengths: np.ndarray) -> int:
        """
        Find index closest to Wavelength of Interest "WoI"

        Args:
            WoI (float): Wavelength of interest

        Returns:
            int: Index of closest wavelength
        """
        return np.argmin(np.abs(wavelengths - WoI))

def plot_preiodic_table(element_counts: dict, figsize: tuple = (10, 5), fname = 'cached.png'):
    """
    Plots a periodic table and for each element in element_counts, classifies
    it as present (3 or more emission lines found), yellow (between 1 and 3 
    emission lines found), and red (no lines found).

    Args:
        element_counts Dict(str, int): Dictionary with element names and correponding
        number of emission lines found
        figsize (tuple): Size of the figure
    """


    element_groups = {
        'Alkali Metals': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
        'Alkaline Earth Metals': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
        'Transition Metals': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                            'Rf', 'Db', 'Sg', 'Bh', 'Hs'],
        'Post-Transition Metals': ['Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'Po', 'At'],
        'Metalloids': ['B', 'Si', 'Ge', 'As', 'Sb', 'Te'],
        'Reactive Non-Metals': ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I'],
        'Noble Gases': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'],
        'Lanthanides': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
        'Actinides': ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'],
        'Unknown Properties': ['Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    }

    element_to_group = {}
    for group, elements in element_groups.items():
        for element in elements:
            element_to_group[element] = group

    group_colors = {
        'Alkali Metals': 'xkcd:light aqua',
        'Alkaline Earth Metals': 'xkcd:scarlet',
        'Transition Metals': 'xkcd:muted purple',
        'Post-Transition Metals': 'xkcd:viridian',
        'Metalloids': 'xkcd:apricot',
        'Reactive Non-Metals': 'xkcd:prussian blue',
        'Noble Gases': 'xkcd:blood',
        'Lanthanides': 'xkcd:azure',
        'Actinides': 'xkcd:umber',
        'Unknown Properties': 'xkcd:grey'
    }

    fig, ax = plt.subplots(figsize=figsize)

    periodic_table = [
        ['H', '', '','', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
        ['Li', 'Be', '','', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
        ['Na', 'Mg', '','', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
        ['K', 'Ca', 'Sc','', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
        ['Rb', 'Sr', 'Y','', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
        ['Cs', 'Ba', 'La','', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
        ['Fr', 'Ra', 'Ac','', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'],
        ['', '', '', '','', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '','Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', ''],
        ['', '', '','Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', '']
    ]

    for i, row in enumerate(periodic_table):
        for j, element in enumerate(row):
            if element: 
                count = element_counts.get(element, None)
                group = element_to_group.get(element, 'Unknown')

                if count is not None:
                    if count >= 3:
                        color = '#98FB98' 
                        fontcolor = 'xkcd:dark sage'
                    elif 0 < count < 3:
                        color = '#FFFFE0'  
                        fontcolor = 'xkcd:dark yellow'
                    else:
                        color = '#FFC0CB'  
                        fontcolor = 'xkcd:burnt red'
                    fontweight = 'bold'
                    alpha = 1.0  
                else:
                    color = group_colors.get(group, '#F5F5DC')  
                    fontweight = 'normal'
                    fontcolor = 'black'
                    alpha = 0.15 

                fancy_box = patches.FancyBboxPatch(
                    (j + 0.05, -i + 0.05), 0.9, 0.9, 
                    boxstyle="round,pad=0.02", 
                    edgecolor='black', 
                    facecolor=color, 
                    alpha=alpha
                )

                ax.add_patch(fancy_box)
                ax.text(j + 0.5, -i + 0.5, element, ha='center', va='center', fontsize=12, 
                        weight=fontweight, color=fontcolor, alpha=alpha)

    ax.set_xlim(0, 20) 
    ax.set_ylim(-10, 1)
    ax.set_aspect('equal')
    ax.axis('off') 
    fig.suptitle('Periodic Table of Elements', fontsize=20)
    fig.tight_layout()
    fig.savefig(fname, transparent = True, dpi = 300)


def standard_analysis(data_cube, wavelengths, radius = 3, cmap = 'turbo'):
    mean_signal = np.mean(data_cube, axis = (0, 1))
    min_signal = np.min(data_cube, axis = (0, 1))
    max_signal = np.max(data_cube, axis = (0, 1))


    fig = plt.figure(tight_layout = True, figsize = (10, 5))
    gs = gridspec.GridSpec(1, 2)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    x_center, y_center = data_cube.shape[1]//2, data_cube.shape[0]//2

    # Plot Spectrum (Average, MinMax, Point)
    axs = ax1
    axs.plot(wavelengths, mean_signal, lw = 2, ls = '-', color = 'lightblue', label = 'Mean')
    meanr, = axs.plot(wavelengths, data_cube[x_center - radius:x_center + radius, y_center - radius:y_center + radius].mean(axis = (0, 1)),
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

    # Spatial Distribuition of selected emission line
    axs = ax2
    axs.set_title('Spatial Distribuition')
    spatial_dist = axs.imshow(data_cube[:, :, wn], cmap = cmap, interpolation = 'gaussian')
    sca = axs.scatter(x_center, y_center, color = 'k', s = 40)
    axs.set_xlabel(r'$x(mm)$')
    axs.set_ylabel(r'$y(mm)$')
    axs.grid(False)

    # Functions for Interaction
    def update_map(wn):
        spatial_dist.set_data(data_cube[:, :, wn]) 
        spatial_dist.set_clim(vmin = data_cube[:, :, wn].min(), vmax = data_cube[:, :, wn].max())
        line.set_xdata(wavelengths[wn])

    def onclick(event):
        if event.dblclick:
            if event.inaxes == ax1:
                ix, _ = event.xdata, event.ydata
                wn = wavelength_to_index(ix, wavelengths)
                update_map(wn)
                fig.canvas.draw_idle()
            elif event.inaxes == ax2:
                xx, yy = int(event.xdata), int(event.ydata)
                sca.set_offsets([xx, yy])
                data_region = data_cube[yy - radius:yy + radius, xx - radius:xx + radius]
                if data_region.shape == (2*radius, 2*radius, data_cube.shape[-1]) and radius != 0:
                    meanr.set_data(wavelengths, data_region.mean(axis = (0, 1)))
                else:
                    meanr.set_data(wavelengths, data_cube[yy, xx])
                fig.canvas.draw_idle()
            
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.suptitle('LIBS Signal Analysis', fontsize = 20)
    fig.tight_layout()