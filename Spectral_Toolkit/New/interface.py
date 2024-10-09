from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import dash
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from LoaderTools.libs import LibsLoader
from AnalysisTools.analysis import AnalyticsToolkit
from AnalysisTools.visual import dash_periodic_table


class SpectralDashboard:
    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.SAMPLE_ID = "[sample-id]"
        self.PROJECT_ID = "[project-id]"

        # Load data using LibsLoader
        self.data_handler = self._initialize_data()

        # Initialize AnalyticsToolkit
        self.analytics = AnalyticsToolkit()

        # Perform feature extraction using AnalyticsToolkit
        self.analytics.automatic_feature_extraction(
            dataset=self.data_handler.dataset,
            wavelengths=self.data_handler.wavelengths,
            fft_features=20,
            intens_features=20,
            sigma=1
        )

        # Perform clustering using extracted features
        self.labels = self.analytics.clustering(
            model='kmeans',
            n_clusters=4,
            feature_cube=self.analytics.features,
            scaler='minmax',
            random_state=10
        )

        # Pre-calculate max_spectrum and region-based average spectra
        self.max_spectrum = np.max(self.data_handler.dataset, axis=(0, 1))
        self.region_avg_spectra = self._calculate_region_avg_spectra()

        # Calculate region counts
        self.region_counts = self._create_region_counts()

        # Create initial figures
        self.figures = {
            'periodic_table': self._create_periodic_table_plot(),
            'spectral_image': self._create_spectral_image_plot(),
            'spectrum': self._create_spectrum_plot(),
            'classification': self._create_labels_image_plot()
        }

        # Set up layout
        self.app.layout = self._create_layout()

        # Set up callbacks
        self._setup_callbacks()

    def _initialize_data(self):
        data_handler = LibsLoader(r"E:/Data/Data_LIBS/ForHolo/wrench_map")
        data_handler.load_dataset(baseline_corrected=True)
        data_handler.normalize_to_sum()
        return data_handler

    def _calculate_region_avg_spectra(self):
        """
        Calculate the average spectrum for each classification region.
        """
        region_avg_spectra = {}
        num_regions = np.unique(self.labels).size

        for region in range(num_regions):
            region_mask = (self.labels == region)
            region_spectra = self.data_handler.dataset[region_mask]

            # Calculate the average spectrum for this region
            region_avg_spectra[region] = np.mean(region_spectra, axis=0)

        return region_avg_spectra

    def _create_region_counts(self):
        """
        Calculate element counts for the average spectrum in each classification region.
        """
        region_counts = {}
        for region, avg_spectrum in self.region_avg_spectra.items():
            counts = self.analytics.identify_from_elements(
                spectrum_or_cube=avg_spectrum,
                wavelengths=self.data_handler.wavelengths,
                min_intensity=0.1,
                return_counts=True
            )
            region_counts[region] = counts

        return region_counts

    def _setup_callbacks(self):
        # Initialize selected wavelength
        self.selected_wavelength = None

        @self.app.callback(
            [Output('periodic-table', 'figure'),
             Output('spectrum-plot', 'figure')],
            [Input('classification-plot', 'clickData'),
             Input('spectrum-plot', 'clickData')]
        )
        def update_plots(classification_clickData, spectrum_clickData):
            ctx = dash.callback_context
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

            selected_region = None
            if classification_clickData:
                point = classification_clickData['points'][0]
                selected_region = int(point['z'])

            if spectrum_clickData and triggered_id == 'spectrum-plot':
                point = spectrum_clickData['points'][0]
                self.selected_wavelength = point['x']

            # Update periodic table based on region
            if selected_region is not None:
                updated_periodic_table = dash_periodic_table(
                    self.region_counts[selected_region],
                    figsize=(900, 350)
                )
                updated_periodic_table.update_layout(
                    margin=dict(l=25, r=25, t=25, b=25)
                )
            else:
                updated_periodic_table = self.figures['periodic_table']

            # Update spectrum plot
            updated_spectrum = self._create_spectrum_plot(
                selected_region=selected_region,
                selected_wavelength=self.selected_wavelength
            )

            return updated_periodic_table, updated_spectrum

        @self.app.callback(
            Output('spectral-image', 'figure'),
            [Input('spectrum-plot', 'clickData')]
        )
        def update_spectral_image(clickData):
            if clickData is None:
                return self.figures['spectral_image']

            # Get clicked wavelength
            point = clickData['points'][0]
            wavelength = point['x']
            self.selected_wavelength = wavelength

            # Update spectral image for selected wavelength
            return self._create_spectral_image_plot(wavelength)

    def _create_spectrum_plot(self, selected_region=None, selected_wavelength=None):
        fig = go.Figure()

        # Calculate and plot average spectrum of whole dataset
        avg_spectrum = np.mean(self.data_handler.dataset, axis=(0, 1))
        fig.add_trace(go.Scatter(
            x=self.data_handler.wavelengths,
            y=avg_spectrum,
            name='Average Spectrum',
            line=dict(color='steelblue')
        ))

        # Plot max spectrum
        fig.add_trace(go.Scatter(
            x=self.data_handler.wavelengths,
            y=self.max_spectrum,
            name='Max Spectrum',
            line=dict(color='gray', dash='dot')
        ))

        # Add region spectrum if selected
        if selected_region is not None:
            fig.add_trace(go.Scatter(
                x=self.data_handler.wavelengths,
                y=self.region_avg_spectra[selected_region],
                name=f'Region {selected_region}',
                line=dict(color='green', dash='dash')
            ))

        # Add vertical line for selected wavelength
        if selected_wavelength is not None:
            fig.add_vline(
                x=selected_wavelength,
                line_width=1,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{selected_wavelength:.2f} nm",
                annotation_position="top"
            )

        min_wv = np.floor(self.data_handler.wavelengths.min() / 100) * 100
        max_wv = np.ceil(self.data_handler.wavelengths.max() / 100) * 100
        tick_values = np.arange(min_wv, max_wv + 100, 100)

        # Update layout with internal legend
        fig.update_layout(
            width=550, height=360,
            title={
                'text': "Spectrum Analysis",
                'x': 0.45, 'y': 0.9,
                'xanchor': 'center',
                'font': {'size': 22}
            },
            margin=dict(l=70, r=70, t=70, b=70),
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.4)',
                bordercolor='Black',
                borderwidth=1
            ),
            **self._get_axis_settings(tick_values)
        )

        return fig

    def _get_axis_settings(self, tick_values):
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
            'xaxis': dict(
                title={'text': "Wavelengths (nm)", 'font': {'size': 20}},
                **axis_settings
            ),
            'yaxis': dict(
                title={'text': "Intensity (arb.un.)", 'font': {'size': 20}},
                **axis_settings
            ),
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': dict(color='black')
        }

    def _create_periodic_table_plot(self):
        counts = self.analytics.identify_from_elements(
            spectrum_or_cube=self.max_spectrum,
            wavelengths=self.data_handler.wavelengths,
            min_intensity=0.1,
            return_counts=True
        )

        fig = dash_periodic_table(counts, figsize=(900, 350))
        fig.update_layout(
            margin=dict(l=25, r=25, t=25, b=25)
        )
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

    def _create_spectral_image_plot(self, wavelength=None):
        if wavelength is None:
            wavelength = 670.76  # default wavelength

        wv_index = self.data_handler.wavelength_to_index(wavelength)
        slice_data = self.data_handler.dataset[:, :, wv_index]

        fig = px.imshow(
            slice_data,
            color_continuous_scale='turbo',
            labels={'color': 'Intensity'}
        )

        fig.update_layout(
            width=365, height=355,
            title=None,
            margin=dict(l=0, r=0, t=0, b=0),
            **self._get_common_axis_settings(slice_data),
            coloraxis_colorbar=dict(
                thickness=10,
                len=0.64,
                y=0.5,
                x=1.05,
                yanchor='middle',
                ticks='outside',
                tickvals=[slice_data.min(), slice_data.max()],
                ticktext=[f"{slice_data.min():.3f}", f"{slice_data.max():.3f}"],
                tickfont=dict(size=14, color='black'),
                tickcolor='black',
                title={'text': ''}
            )
        )
        return fig

    def _create_ui_components(self):
        title_section = html.Div([
            html.H1("Spectral UI"),
            html.P([
                html.Br(),
                html.Span(f"SAMPLE: {self.SAMPLE_ID}"),
                html.Br(), html.Br(),
                html.Span(f"This interface presents a resumed analysis of the findings for the spectral data collected, as part of PROJECT {self.PROJECT_ID}."),
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
                dcc.Graph(
                    id='classification-plot',
                    figure=self.figures['classification'],
                    config={'displayModeBar': False}
                )
            ], className="plot-section1")
        ], className="left-panel")

        right_panel = html.Div([
            html.Div([
                html.Div(
                    dcc.Graph(
                        id='spectrum-plot',
                        figure=self.figures['spectrum'],
                        config={'displayModeBar': False}
                    ),
                    className='section2-plot'
                ),
                html.Div(
                    dcc.Graph(
                        id='spectral-image',
                        figure=self.figures['spectral_image'],
                        config={'displayModeBar': False}
                    ),
                    className='section2-imshow'
                )
            ], className="plot-section2"),
            html.Div([
                dcc.Graph(
                    id='periodic-table',
                    figure=self.figures['periodic_table'],
                    config={'displayModeBar': False}
                )
            ], className="plot-section3")
        ], className="right-panel")

        return dbc.Container([
            html.Div([left_panel, right_panel], className="main-container")
        ], fluid=True, className='dashboard-container')

    def run(self, debug=True):
        self.app.run_server(debug=debug)

print("Starting...")
print(f"Directory: {os.getcwd()}")
dashboard = SpectralDashboard()
dashboard.run(debug=True)