import dash
import base64
import plotly.graph_objs as go
import numpy as np

from dash import dcc, html, Input, Output
from LoaderTools.libs import LibsLoader
from AnalysisTools.analytics import AnalyticsToolkit
from AnalysisTools import visual


# Initialize the Dash app
app = dash.Dash(__name__)

# Load and process the spectral data
fname = r"E:/Data/Data_LIBS/ForHolo/wrench_map"
data_handler = LibsLoader(fname)
data_handler.load_dataset(baseline_corrected=True)
data_handler.normalize_to_sum()
data_handler.automatic_feature_extraction(fft_features=20, intens_features=20, sigma=1)

analytics = AnalyticsToolkit()
labels = analytics.clustering(model='kmeans', n_clusters=4, feature_cube=data_handler.features, scaler='minmax', random_state=10)

# Define styles
CONTENT_STYLE = {
    'margin-left': '2rem',
    'margin-right': '2rem',
    'padding': '2rem 1rem',
    'background-color': 'darkgrey',
    'min-height': '100vh',
    'font-family': 'Arial, sans-serif',
}

HEADER_STYLE = {
    'text-align': 'center',
    'padding': '20px 0',
    'font-size': '40px',
    'font-weight': 'bold',
    'color': '#333333',
    'background-color': 'light blue',
    'font-family': 'Arial, sans-serif',
}

# Define the layout for the dashboard
app.layout = html.Div([
    html.H1("Spectral UI", style=HEADER_STYLE),
    
    # Top row
    html.Div([
        # Periodic Table (top left)
        html.Div([
            dcc.Graph(id='imported-figure', config={'displayModeBar': False}, style={'height': '300px'})
        ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        # Classification Image (top right)
        html.Div([
            dcc.Graph(id='classification-image', config={'displayModeBar': False}, style={'height': '300px'})
        ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top', 'float': 'right'}),
    ], style={'margin-bottom': '20px'}),
    
    # Bottom row (centered)
    html.Div([
        # Mean Spectrum Graph
        html.Div([
            dcc.Graph(id='spectrum-graph', config={'displayModeBar': False}, style={'height': '450px'})
        ], style={'width': '60%', 'margin': '0 auto'}),
    ], style={'text-align': 'center'}),
    
], style=CONTENT_STYLE)

@app.callback(
    Output('spectrum-graph', 'figure'),
    Input('spectrum-graph', 'clickData')
)
def update_spectrum(clickData):
    spectrum = np.mean(data_handler.dataset, axis=(0, 1))
    title = "Mean Spectrum"
    
    fig_spectrum = go.Figure(data=[go.Scatter(x=data_handler.wavelengths, y=spectrum, mode='lines')])
    fig_spectrum.update_layout(
        title=dict(text=title, font=dict(size=20, color='black', family='Arial, sans-serif')),
        xaxis_title=dict(text="Wavelength (nm)", font=dict(size=16, color='black', family='Arial, sans-serif')),
        yaxis_title=dict(text="Intensity", font=dict(size=16, color='black', family='Arial, sans-serif')),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='black',
            linewidth=2,
            zeroline=False,
            ticks='outside',
            tickfont=dict(size=12, color='black'),
            title_font=dict(size=16, color='black')
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='black',
            linewidth=2,
            zeroline=False,
            ticks='outside',
            tickfont=dict(size=12, color='black'),
            title_font=dict(size=16, color='black'),
            exponentformat='e',
            showexponent='all'
        ),
    )
    
    if clickData:
        selected_wv = clickData['points'][0]['x']
        spectral_image = data_handler.dataset[:, :, np.where(data_handler.wavelengths == selected_wv)[0][0]]
        
        # Add the vertical line on top of the main plot but below the heatmap
        fig_spectrum.add_shape(
            type="line",
            x0=selected_wv, x1=selected_wv,
            y0=spectrum.min(), y1=spectrum.max(),
            line=dict(color='red', width=2, dash="dash"),
            layer='above'
        )
        
        # Add the heatmap on top of everything
        fig_spectrum.add_trace(go.Heatmap(
            z=spectral_image,
            xaxis='x2', yaxis='y2',
            showscale=False,
            hoverinfo='none'
        ))
        
        fig_spectrum.update_layout(
            xaxis2=dict(
                domain=[0.65, 1],
                anchor='y2',
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                scaleanchor='y2',
                scaleratio=1
            ),
            yaxis2=dict(
                domain=[0.65, 1],
                anchor='x2',
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                scaleanchor='x2',
                scaleratio=1
            ),
        )
    
    return fig_spectrum

@app.callback(
    Output('classification-image', 'figure'),
    Output('imported-figure', 'figure'),
    Input('classification-image', 'clickData')
)
def update_classification(clickData):
    selected_pixel = 0 if clickData is None else labels[clickData['points'][0]['y'], clickData['points'][0]['x']]

    counts = analytics.identify_on_cluster(spectral_cube=data_handler.dataset,
                                           wavelengths=data_handler.wavelengths,
                                           tolerance=0.05,
                                           min_intensity=0.1,
                                           return_counts=True,
                                           cluster_number=selected_pixel)
    
    # Assuming visual.dash_periodic_table() returns a base64 encoded image
    img_base64 = visual.dash_periodic_table(counts)
    
    fig_classification = go.Figure(data=[go.Heatmap(
        z=labels,
        colorscale=[[0, 'black'], [0.33, 'red'], [0.66, 'orange'], [1, 'yellow']],
        showscale=False
    )])
    
    fig_classification.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', scaleratio=1),
        margin=dict(t=0, l=0, b=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial, sans-serif', size=14, color='black'),
    )
    
    fig_periodic_table = go.Figure(go.Image(source=f'data:image/png;base64,{img_base64}'))
    fig_periodic_table.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', scaleratio=1),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, l=0, b=0, r=0),
        font=dict(family='Arial, sans-serif', size=14, color='black'),
    )
    
    return fig_classification, fig_periodic_table

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)

# if __name__ == '__main__':
#     app.run_server(debug=True)

# cd Desktop/Tomás/GitHub/Spectral_repo/
# 172.17.17.171