import numpy as np
import plotly.graph_objects as go
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
    hover_x = []
    hover_y = []
    hover_text = []

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
                        yref='y',
                        # label={'color': 'Counts'},
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

                hover_x.append(j + 0.5)  # Center of the element box horizontally
                hover_y.append(-i - 0.5)  # Center of the element box vertically
                hover_text.append(f"Element: {element}<br>Count: {count if count is not None else 'N/A'}")
    
    fig.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode='markers',
            marker=dict(size=0.1, color='rgba(0,0,0,0)'),  # Invisible markers
            text=hover_text,
            hoverinfo='text'  # Display the hovertext when hovering over each element
        )
    )

    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        shapes=shapes,
        annotations=annotations
    )

    # Adjust the axis ranges to fit the entire table
    fig.update_xaxes(range=[0, 19])
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig
