from dash import Dash, html, dcc
from AnalysisTools import visual
from LoaderTools.libs import LibsLoader
from AnalysisTools.analytics import AnalyticsToolkit
from dash.dependencies import Input, Output

import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash
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
        data_handler.automatic_feature_extraction(fft_features=20, intens_features=20, sigma=1)
        return data_handler

    def _perform_clustering(self):
        return self.analytics.clustering(
            model='kmeans',
            n_clusters=4,
            feature_cube=self.data_handler.features,
            scaler='minmax',
            random_state=10
        )

    def _calculate_region_avg_spectra(self):
        """
        Calculate the average spectrum for each classification region.
        Returns:
            A dictionary with region indices as keys and their corresponding average spectra as values.
        """
        region_avg_spectra = {}
        num_regions = np.unique(self.labels).size

        for region in range(num_regions):
            # Select data points belonging to the current region
            region_mask = (self.labels == region)
            region_spectra = self.data_handler.dataset[region_mask]

            # Calculate the average spectrum for this region
            region_avg_spectra[region] = np.mean(region_spectra, axis=0)

        return region_avg_spectra
    
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

    def _setup_callbacks(self):
        # Initialize selected wavelength
        self.selected_wavelength = None

        # Callback for classification plot click
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
                updated_periodic_table = visual.dash_periodic_table(
                    self.region_counts[selected_region], 
                    figsize=(900, 350)
                )
            else:
                updated_periodic_table = self.figures['periodic_table']
            
            updated_periodic_table.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=25, r=25, t=25, b=25)
            )
            updated_periodic_table.update_xaxes(range=[0, 19])
            updated_periodic_table.update_yaxes(scaleanchor="x", scaleratio=1)
            
            # Update spectrum plot
            updated_spectrum = self._create_spectrum_plot(
                selected_region=selected_region,
                selected_wavelength=self.selected_wavelength
            )
            
            return updated_periodic_table, updated_spectrum

        # Callback for spectrum plot click
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
                x=0.02,  # Adjust this value to move legend left/right
                y=0.98,  # Adjust this value to move legend up/down
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.4)',
                bordercolor='Black',
                borderwidth=1
            ),
            **self._get_spectrum_axis_settings(tick_values)
        )
        
        return fig

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
        # Use the pre-calculated max_spectrum for periodic table as well
        counts = self.analytics.identify_from_elements(
            spectrum_or_cube=self.max_spectrum,  # Now using the stored max_spectrum
            wavelengths=self.data_handler.wavelengths,
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

    def _create_region_counts(self):
        """
        Calculate element counts for the average spectrum in each classification region.
        Returns:
            A dictionary with region indices as keys and their corresponding element counts as values.
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
        
        common_settings = self._get_common_axis_settings(slice_data)
        colorbar_settings = self._get_colorbar_settings(slice_data)
        
        fig.update_layout(
            width=365, height=355,
            title=None,
            margin=dict(l=0, r=0, t=0, b=0),
            **common_settings,
            **colorbar_settings
        )
        return fig

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