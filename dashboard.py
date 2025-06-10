"""
Centralized dashboard for Crime Hotspot Prediction & Risk Analysis.
Combines all visualizations into a single interactive interface.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import sys
import dash_bootstrap_components as dbc

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, 
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.FLATLY])

# Custom CSS
CUSTOM_CSS = {
    'custom-container': {
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '20px'
    },
    'custom-header': {
        'background': 'linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d)',
        'color': 'white',
        'padding': '30px',
        'borderRadius': '15px',
        'marginBottom': '30px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
        'textAlign': 'center'
    },
    'main-title': {
        'fontSize': '2.5rem',
        'fontWeight': '700',
        'marginBottom': '15px',
        'textShadow': '2px 2px 4px rgba(0,0,0,0.2)',
        'letterSpacing': '1px'
    },
    'subtitle': {
        'fontSize': '1.2rem',
        'fontWeight': '400',
        'opacity': '0.9',
        'marginBottom': '0',
        'fontStyle': 'italic'
    },
    'custom-tab': {
        'backgroundColor': '#f8f9fa',
        'borderRadius': '5px',
        'padding': '10px'
    },
    'custom-graph': {
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'padding': '15px',
        'marginBottom': '20px'
    },
    'insight-box': {
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'padding': '20px',
        'marginBottom': '20px',
        'borderLeft': '4px solid #1a2a6c'
    },
    'insight-title': {
        'color': '#1a2a6c',
        'fontSize': '1.1rem',
        'fontWeight': '600',
        'marginBottom': '10px'
    },
    'insight-text': {
        'color': '#666',
        'fontSize': '0.95rem',
        'lineHeight': '1.5'
    }
}

# Load data
def load_data():
    try:
        df = pd.read_csv("data/cleaned_data.csv")
        if df.empty:
            raise ValueError("Loaded data is empty")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

# Load GeoJSON data
def load_geojson():
    try:
        gdf = gpd.read_file('India_GeoJSON.json')
        if gdf.empty:
            raise ValueError("Loaded GeoJSON is empty")
        return gdf
    except Exception as e:
        print(f"Warning: Could not load GeoJSON: {str(e)}")
        return None

# Create the layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Crime Hotspot Prediction & Risk Analysis",
                       className="text-center mb-4",
                       style=CUSTOM_CSS['main-title']),
                html.P("Interactive visualization of crime data across Indian states (2020-2022)",
                      className="text-center",
                      style=CUSTOM_CSS['subtitle'])
            ], style=CUSTOM_CSS['custom-header'])
        ])
    ]),
    
    # Key Insights Section
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Key Insights", className="text-center mb-4"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H5("Crime Rate Trends", style=CUSTOM_CSS['insight-title']),
                            html.P("Explore how crime rates have evolved across states from 2020 to 2022. "
                                  "Compare states and identify patterns in crime distribution.",
                                  style=CUSTOM_CSS['insight-text'])
                        ], style=CUSTOM_CSS['insight-box'])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H5("Geographical Patterns", style=CUSTOM_CSS['insight-title']),
                            html.P("Visualize crime hotspots across India. The map reveals regional patterns "
                                  "and helps identify areas requiring immediate attention.",
                                  style=CUSTOM_CSS['insight-text'])
                        ], style=CUSTOM_CSS['insight-box'])
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H5("Statistical Analysis", style=CUSTOM_CSS['insight-title']),
                            html.P("Dive deep into correlations between different crime metrics. "
                                  "Understand the relationships between population, crime rates, and other factors.",
                                  style=CUSTOM_CSS['insight-text'])
                        ], style=CUSTOM_CSS['insight-box'])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H5("Predictive Insights", style=CUSTOM_CSS['insight-title']),
                            html.P("Explore feature importance and anomaly detection to understand "
                                  "what factors most influence crime rates and identify unusual patterns.",
                                  style=CUSTOM_CSS['insight-text'])
                        ], style=CUSTOM_CSS['insight-box'])
                    ], width=6)
                ])
            ], style=CUSTOM_CSS['custom-container'])
        ])
    ]),
    
    # Year Selector
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Select Year:", className="mb-3"),
                dcc.Dropdown(
                    id='year-selector',
                    options=[
                        {'label': '2020', 'value': '2020'},
                        {'label': '2021', 'value': '2021'},
                        {'label': '2022', 'value': '2022'}
                    ],
                    value='2022',
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style=CUSTOM_CSS['custom-container'])
        ], width=4)
    ]),
    
    # Navigation Tabs
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                # Overview Tab
                dbc.Tab(label='Overview', children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4("Crime Rate Trends by State", className="text-center mb-4"),
                                html.P("Compare crime rates across states and years. Hover over bars for detailed information.",
                                      className="text-center text-muted mb-4"),
                                dcc.Graph(id='crime-trends-graph')
                            ], style=CUSTOM_CSS['custom-graph'])
                        ], width=12),
                        
                        dbc.Col([
                            html.Div([
                                html.H4("Top Crime States", className="text-center mb-4"),
                                html.P("Identify states with highest crime rates. The color intensity indicates severity.",
                                      className="text-center text-muted mb-4"),
                                dcc.Graph(id='top-crime-states-graph')
                            ], style=CUSTOM_CSS['custom-graph'])
                        ], width=12)
                    ])
                ]),
                
                # Geographical Analysis Tab
                dbc.Tab(label='Geographical Analysis', children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4("Crime Rate by State", className="text-center mb-4"),
                                html.P("Explore geographical distribution of crime rates. Darker colors indicate higher rates.",
                                      className="text-center text-muted mb-4"),
                                dcc.Graph(id='state-map-graph')
                            ], style=CUSTOM_CSS['custom-graph'])
                        ], width=12),
                        
                        dbc.Col([
                            html.Div([
                                html.H4("Year-over-Year Changes", className="text-center mb-4"),
                                html.P("Track changes in crime rates between years. Red indicates increase, blue indicates decrease.",
                                      className="text-center text-muted mb-4"),
                                dcc.Graph(id='year-over-year-graph')
                            ], style=CUSTOM_CSS['custom-graph'])
                        ], width=12)
                    ])
                ]),
                
                # Statistical Analysis Tab
                dbc.Tab(label='Statistical Analysis', children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4("Correlation Analysis", className="text-center mb-4"),
                                html.P("Understand relationships between different crime metrics. Darker colors indicate stronger correlations.",
                                      className="text-center text-muted mb-4"),
                                dcc.Graph(id='correlation-heatmap-graph')
                            ], style=CUSTOM_CSS['custom-graph'])
                        ], width=12),
                        
                        dbc.Col([
                            html.Div([
                                html.H4("Distribution Analysis", className="text-center mb-4"),
                                html.P("Explore the distribution of crime rates across different dimensions.",
                                      className="text-center text-muted mb-4"),
                                dcc.Graph(id='distribution-graph')
                            ], style=CUSTOM_CSS['custom-graph'])
                        ], width=12)
                    ])
                ]),
                
                # Model Analysis Tab
                dbc.Tab(label='Model Analysis', children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4("Feature Importance", className="text-center mb-4"),
                                html.P("Identify key factors influencing crime rates. Longer bars indicate greater importance.",
                                      className="text-center text-muted mb-4"),
                                dcc.Graph(id='feature-importance-graph')
                            ], style=CUSTOM_CSS['custom-graph'])
                        ], width=12),
                        
                        dbc.Col([
                            html.Div([
                                html.H4("Anomaly Detection", className="text-center mb-4"),
                                html.P("Spot unusual patterns in crime rates. Red markers indicate significant deviations.",
                                      className="text-center text-muted mb-4"),
                                dcc.Graph(id='anomaly-detection-graph')
                            ], style=CUSTOM_CSS['custom-graph'])
                        ], width=12)
                    ])
                ])
            ], style=CUSTOM_CSS['custom-tab'])
        ])
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P("Data Source: National Crime Records Bureau (NCRB)",
                      className="text-center text-muted mb-0")
            ], style={'marginTop': '20px'})
        ])
    ])
], fluid=True)

# Callback for Year Selection
@app.callback(
    [Output('crime-trends-graph', 'figure'),
     Output('top-crime-states-graph', 'figure'),
     Output('state-map-graph', 'figure'),
     Output('year-over-year-graph', 'figure')],
    [Input('year-selector', 'value')]
)
def update_year_based_graphs(selected_year):
    try:
        df = load_data()
        
        # Crime Rate Trends
        fig1 = go.Figure()
        years = ['2020', '2021', '2022']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for year, color in zip(years, colors):
            fig1.add_trace(go.Bar(
                name=year,
                x=df['State/UT'],
                y=df[f'Crime_Rate_{year}'],
                marker_color=color,
                hovertemplate="<b>%{x}</b><br>" +
                            f"Crime Rate ({year}): %{{y:.2f}}<br>" +
                            "Cases: %{customdata}<br>" +
                            "<extra></extra>",
                customdata=df[f'Cases Reported ({year})']
            ))
        
        fig1.update_layout(
            title="Crime Rate Trends by State (2020-2022)",
            xaxis_title="State/UT",
            yaxis_title="Crime Rate (cases per lakh population)",
            template="plotly_white",
            barmode='group',
            hovermode='closest',
            showlegend=True,
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Top Crime States
        top_states = df.nlargest(10, f'Crime_Rate_{selected_year}')
        fig2 = px.bar(
            top_states,
            x='State/UT',
            y=f'Crime_Rate_{selected_year}',
            title=f"Top 10 States by Crime Rate ({selected_year})",
            labels={f'Crime_Rate_{selected_year}': 'Crime Rate per Lakh Population'},
            color=f'Crime_Rate_{selected_year}',
            color_continuous_scale='Reds'
        )
        
        fig2.update_layout(
            template="plotly_white",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # State Map
        gdf = load_geojson()
        if gdf is not None:
            merged = gdf.merge(df, left_on='name', right_on='State/UT', how='left')
            
            fig3 = px.choropleth(
                merged,
                geojson=merged.geometry,
                locations=merged.index,
                color=f'Crime_Rate_{selected_year}',
                hover_name='State/UT',
                hover_data={
                    f'Crime_Rate_{selected_year}': ':.2f',
                    f'Cases Reported ({selected_year})': True,
                    'Population': True
                },
                color_continuous_scale='Reds',
                title=f'Crime Rate by State ({selected_year})'
            )
            
            fig3.update_geos(
                fitbounds="locations",
                visible=False
            )
            
            fig3.update_layout(
                height=700,
                margin={"r":0,"t":30,"l":0,"b":0},
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
        else:
            fig3 = go.Figure()
            fig3.add_annotation(
                text="Geographical data not available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
        
        # Year-over-Year Changes
        if selected_year != '2020':
            prev_year = str(int(selected_year) - 1)
            df['Change'] = df[f'Crime_Rate_{selected_year}'] - df[f'Crime_Rate_{prev_year}']
            
            fig4 = px.bar(
                df,
                x='State/UT',
                y='Change',
                title=f"Year-over-Year Change ({prev_year}-{selected_year})",
                color='Change',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            
            fig4.update_layout(
                template="plotly_white",
                height=600,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
        else:
            fig4 = go.Figure()
            fig4.add_annotation(
                text="No year-over-year comparison available for 2020",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
        
        return fig1, fig2, fig3, fig4
    except Exception as e:
        print(f"Error in update_year_based_graphs: {str(e)}")
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()

# Callback for Statistical Analysis
@app.callback(
    [Output('correlation-heatmap-graph', 'figure'),
     Output('distribution-graph', 'figure')],
    [Input('year-selector', 'value')]
)
def update_statistical_analysis(selected_year):
    try:
        df = load_data()
        
        # Correlation Heatmap
        correlation_matrix = df[['Crime_Rate_2020', 'Crime_Rate_2021', 'Crime_Rate_2022',
                               'Rate of Cognizable Crimes (IPC) (2022)',
                               'Chargesheeting Rate (2022)',
                               'Crime_Rate_Change_2021',
                               'Crime_Rate_Change_2022',
                               'Crime_Rate_3Y_Avg']].corr()
        
        fig1 = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig1.update_layout(
            title="Correlation Matrix of Crime Metrics",
            template="plotly_white",
            height=700,
            xaxis={'tickangle': 45},
            yaxis={'tickangle': 0},
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Distribution Analysis
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Distribution of Crime Rates ({selected_year})',
                'Crime Rate Distribution by Year',
                'Population vs Crime Rate',
                'Year-over-Year Changes'
            )
        )
        
        # Distribution of selected year crime rates
        fig2.add_trace(
            go.Histogram(
                x=df[f'Crime_Rate_{selected_year}'],
                name=selected_year,
                marker_color='#1f77b4'
            ),
            row=1, col=1
        )
        
        # Box plot of crime rates by year
        for year, color in zip(['2020', '2021', '2022'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
            fig2.add_trace(
                go.Box(
                    y=df[f'Crime_Rate_{year}'],
                    name=year,
                    marker_color=color
                ),
                row=1, col=2
            )
        
        # Scatter plot of population vs crime rate
        fig2.add_trace(
            go.Scatter(
                x=df['Population'],
                y=df[f'Crime_Rate_{selected_year}'],
                mode='markers',
                marker=dict(
                    color=df[f'Crime_Rate_{selected_year}'],
                    colorscale='Reds',
                    showscale=True
                ),
                text=df['State/UT'],
                name='States'
            ),
            row=2, col=1
        )
        
        # Year-over-year changes
        if selected_year != '2020':
            prev_year = str(int(selected_year) - 1)
            fig2.add_trace(
                go.Box(
                    y=df[f'Crime_Rate_{selected_year}'] - df[f'Crime_Rate_{prev_year}'],
                    name=f'{prev_year}-{selected_year}',
                    marker_color='#9467bd'
                ),
                row=2, col=2
            )
        
        fig2.update_layout(
            height=1000,
            template="plotly_white",
            showlegend=False,
            title_text="Distribution Analysis of Crime Rates",
            title_x=0.5,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig1, fig2
    except Exception as e:
        print(f"Error in update_statistical_analysis: {str(e)}")
        return go.Figure(), go.Figure()

# Callback for Model Analysis
@app.callback(
    [Output('feature-importance-graph', 'figure'),
     Output('anomaly-detection-graph', 'figure')],
    [Input('year-selector', 'value')]
)
def update_model_analysis(selected_year):
    try:
        df = load_data()
        
        # Feature Importance
        try:
            import joblib
            model = joblib.load('models/rf_model.pkl')
            feature_names = ['Population', 'Crime Rate (IPC)', 'Chargesheeting Rate',
                           'Crime Rate 2020', 'Crime Rate 2021', 'Crime Rate Change 2021',
                           '3-Year Average']
            importance_scores = model.feature_importances_
            
            fig1 = go.Figure()
            
            fig1.add_trace(go.Bar(
                x=importance_scores,
                y=feature_names,
                orientation='h',
                marker_color='#1f77b4',
                text=importance_scores.round(3),
                textposition='auto',
                hovertemplate="<b>%{y}</b><br>" +
                            "Importance: %{x:.3f}<br>" +
                            "<extra></extra>"
            ))
            
            fig1.update_layout(
                title="Feature Importance in Crime Rate Prediction",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                template="plotly_white",
                height=500,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
        except Exception as e:
            print(f"Error loading feature importance: {str(e)}")
            fig1 = go.Figure()
        
        # Anomaly Detection
        z_scores = (df[f'Crime_Rate_{selected_year}'] - df[f'Crime_Rate_{selected_year}'].mean()) / df[f'Crime_Rate_{selected_year}'].std()
        df['Z_Score'] = z_scores
        
        fig2 = go.Figure()
        
        # Normal points
        normal_mask = abs(z_scores) <= 2
        fig2.add_trace(go.Scatter(
            x=df[normal_mask]['State/UT'],
            y=df[normal_mask][f'Crime_Rate_{selected_year}'],
            mode='markers',
            marker=dict(color='#1f77b4'),
            name='Normal',
            hovertemplate="<b>%{x}</b><br>" +
                        "Crime Rate: %{y:.2f}<br>" +
                        "Z-Score: %{customdata:.2f}<br>" +
                        "<extra></extra>",
            customdata=df[normal_mask]['Z_Score']
        ))
        
        # Anomalies
        anomaly_mask = abs(z_scores) > 2
        fig2.add_trace(go.Scatter(
            x=df[anomaly_mask]['State/UT'],
            y=df[anomaly_mask][f'Crime_Rate_{selected_year}'],
            mode='markers',
            marker=dict(color='#d62728', size=10),
            name='Anomaly',
            hovertemplate="<b>%{x}</b><br>" +
                        "Crime Rate: %{y:.2f}<br>" +
                        "Z-Score: %{customdata:.2f}<br>" +
                        "<extra></extra>",
            customdata=df[anomaly_mask]['Z_Score']
        ))
        
        fig2.update_layout(
            title=f"Crime Rate Anomalies ({selected_year})",
            xaxis_title="State/UT",
            yaxis_title="Crime Rate per Lakh Population",
            template="plotly_white",
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig1, fig2
    except Exception as e:
        print(f"Error in update_model_analysis: {str(e)}")
        return go.Figure(), go.Figure()

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050) 