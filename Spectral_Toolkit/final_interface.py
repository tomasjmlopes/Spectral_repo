from dash import Dash, html, dcc
from AnalysisTools import visual
from LoaderTools.libs import LibsLoader
from AnalysisTools.analytics import AnalyticsToolkit
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np

class SpectralDashboard:
    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.SAMPLE_ID = "[sample-id]"
        self.PROJECT_ID = "[project-id]"
        
        # Load and process data
        self.data_handler = self._initialize_data()
        self.analytics = AnalyticsToolkit()
        self.labels = self._perform_clustering()
        
        # Create figures
        self.figures = {
            'periodic_table': self._create_periodic_table_plot(),
            'spectral_image': self._create_spectral_image_plot(),
            'spectrum': self._create_spectrum_plot(),
            'classification': self._create_labels_image_plot()
        }
        
        # Set up layout
        self.app.layout = self._create_layout()

    def _initialize_data(self):
        data_handler = LibsLoader(r"E:/Data/Data_LIBS/ForHolo/wrench_map")
        data_handler.load_dataset(baseline_corrected=True)
        data_handler.normalize_to_sum()
        data_handler.automatic_feature_extraction(fft_features=0, intens_features=20, sigma=1)
        return data_handler

    def _perform_clustering(self):
        return self.analytics.clustering(
            model='kmeans',
            n_clusters=4,
            feature_cube=self.data_handler.features,
            scaler='minmax',
            random_state=10
        )

    def _create_spectral_image_plot(self):
        wv = 670.76
        slice_data = self.data_handler.dataset[:, :, self.data_handler.wavelength_to_index(wv)]
        
        fig = px.imshow(
            slice_data,
            color_continuous_scale='turbo',
            labels={'color': 'Intensity'}
        )
        
        common_settings = self._get_common_axis_settings(slice_data)
        colorbar_settings = self._get_colorbar_settings(slice_data)
        
        fig.update_layout(
            width=365, height=355,
            title={
                'text': "Spatial Distribution",
                'x': 0.45, 'y': 0.9,
                'xanchor': 'center',
                'font': {'size': 22}
            },
            margin=dict(l=0, r=0, t=0, b=0),
            **common_settings,
            **colorbar_settings
        )
        return fig

    def _get_colorbar_settings(self, data):
        return {
            'coloraxis_colorbar': dict(
                thickness=10,
                len=0.64,
                y=0.5,
                x=1.05,
                yanchor='middle',
                ticks='outside',
                tickvals=[data.min(), data.max()],
                ticktext=[f"{data.min():.3f}", f"{data.max():.3f}"],
                tickfont=dict(size=14, color='black'),
                tickcolor='black',
                title={'text': ''}
            )
        }

    def _create_spectrum_plot(self):
        max_spectrum = np.max(self.data_handler.dataset, axis=(0, 1))
        
        fig = px.line(
            x=self.data_handler.wavelengths,
            y=max_spectrum,
            color_discrete_sequence=['steelblue']
        )

        min_wv = np.floor(self.data_handler.wavelengths.min() / 100) * 100
        max_wv = np.ceil(self.data_handler.wavelengths.max() / 100) * 100
        tick_values = np.arange(min_wv, max_wv + 100, 100)

        fig.update_layout(
            width=550, height=360,
            title={
                'text': "Spectrum",
                'x': 0.5, 'y': 0.9,
                'xanchor': 'center',
                'font': {'size': 22}
            },
            margin=dict(l=70, r=70, t=70, b=70),
            **self._get_spectrum_axis_settings(tick_values)
        )
        return fig

    def _create_periodic_table_plot(self):
        counts = self.analytics.identify_from_elements(
            spectral_cube=self.data_handler.dataset,
            wavelengths=self.data_handler.wavelengths,
            operation='average',
            min_intensity=0.1,
            return_counts=True
        )
        
        fig = visual.dash_periodic_table(counts, figsize=(900, 350))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=25, r=25, t=25, b=25)
        )
        fig.update_xaxes(range=[0, 19])
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    def _create_labels_image_plot(self):
        fig = px.imshow(
            self.labels,
            color_continuous_scale='turbo',
            labels={'color': 'Class'}
        )
        
        fig.update_layout(
            width=340, height=340,
            title={
                'text': "Final Classification",
                'x': 0.53, 'y': 0.95,
                'xanchor': 'center',
                'font': {'size': 22}
            },
            margin=dict(l=5, r=5, t=5, b=5),
            **self._get_common_axis_settings(self.labels),
            coloraxis_showscale=False
        )
        return fig

    def _get_common_axis_settings(self, data):
        return {
            'xaxis': dict(
                title={'text': "x (mm)", 'font': {'size': 18}, 'standoff': 0},
                range=[0, data.shape[1] - 1],
                tickvals=[1, data.shape[1] - 1],
                ticktext=[0, data.shape[1]],
                tickfont=dict(size=16, color='black'),
                linewidth=2, linecolor='black', mirror=True
            ),
            'yaxis': dict(
                title={'text': "y (mm)", 'font': {'size': 18}, 'standoff': 0},
                range=[0, data.shape[0] - 1],
                tickvals=[1, data.shape[0] - 1],
                ticktext=[data.shape[0], 0],
                tickfont=dict(size=16, color='black'),
                linewidth=2, linecolor='black', mirror=True
            ),
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': dict(color='black')
        }

    def _get_spectrum_axis_settings(self, tick_values):
        axis_settings = {
            'tickfont': {'size': 16, 'color': 'black'},
            'showline': True,
            'linewidth': 2,
            'linecolor': 'black',
            'mirror': False,
            'showgrid': False,
            'zeroline': False,
            'tickmode': 'array',
            'tickvals': tick_values
        }
        return {
            'xaxis': dict(title={'text': "Wavelengths (nm)", 'font': {'size': 20}}, **axis_settings),
            'yaxis': dict(title={'text': "Intensity (arb.un.)", 'font': {'size': 20}}, **axis_settings),
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': dict(color='black')
        }

    def _create_ui_components(self):
        title_section = html.Div([
            html.H1("Spectral UI"),
            html.P([
                html.Br(),
                html.Span(f"SAMPLE: {self.SAMPLE_ID}"),
                html.Br(), html.Br(),
                html.Span(f"This interface presents a resumed analysis of the finding for the spectral data collected, as part of PROJECT {self.PROJECT_ID}."),
                html.Br(), html.Br(),
                html.Span("by INESC TEC Spectroscopy Group"),
            ])
        ], className="title-section")

        radio_section = html.Div([
            dbc.RadioItems(
                options=[
                    {"label": label, "value": val} 
                    for val, label in enumerate(["Analysis", "Report", "About"], 1)
                ],
                value=1,
                className="btn-group",
                inputClassName="btn-check",
                labelClassName="btn btn-outline-light",
                labelCheckedClassName="btn btn-light",
            )
        ], className="radio-section")

        return title_section, radio_section

    def _create_layout(self):
        title_section, radio_section = self._create_ui_components()
        
        left_panel = html.Div([
            title_section,
            radio_section,
            html.Div([
                dcc.Graph(figure=self.figures['classification'], config={'displayModeBar': False})
            ], className="plot-section1")
        ], className="left-panel")

        right_panel = html.Div([
            html.Div([
                html.Div(dcc.Graph(figure=self.figures['spectrum'], config={'displayModeBar': False}), 
                         className='section2-plot'),
                html.Div(dcc.Graph(figure=self.figures['spectral_image'], config={'displayModeBar': False}), 
                         className='section2-imshow')
            ], className="plot-section2"),
            html.Div([
                dcc.Graph(figure=self.figures['periodic_table'], config={'displayModeBar': False})
            ], className="plot-section3")
        ], className="right-panel")

        return dbc.Container([
            html.Div([left_panel, right_panel], className="main-container")
        ], fluid=True, className='dashboard-container')

    def run(self, debug=True):
        self.app.run_server(debug=debug)