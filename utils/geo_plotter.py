"""
Geographical plotting utility for crime analysis.
Handles creation of maps and geographical visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.ticker as mticker

class GeoPlotter:
    def __init__(self, df):
        self.df = df
        self.output_dir = Path("output/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
        
    def plot_crime_trends(self):
        """Plot crime rate trends by state with interactive elements"""
        # Calculate crime rates
        crime_rates = self.df[['State/UT', 'Cases Reported (2020)', 'Cases Reported (2021)', 
                             'Cases Reported (2022)', 'Population']].copy()
        
        crime_rates['Crime_Rate_2020'] = crime_rates['Cases Reported (2020)'] / crime_rates['Population']
        crime_rates['Crime_Rate_2021'] = crime_rates['Cases Reported (2021)'] / crime_rates['Population']
        crime_rates['Crime_Rate_2022'] = crime_rates['Cases Reported (2022)'] / crime_rates['Population']
        
        # Create interactive plot using plotly
        fig = go.Figure()
        
        years = ['2020', '2021', '2022']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for year, color in zip(years, colors):
            fig.add_trace(go.Bar(
                name=year,
                x=crime_rates['State/UT'],
                y=crime_rates[f'Crime_Rate_{year}'],
                marker_color=color,
                hovertemplate="<b>%{x}</b><br>" +
                            f"Crime Rate ({year}): %{{y:.2f}}<br>" +
                            "Cases: %{customdata}<br>" +
                            "<extra></extra>",
                customdata=crime_rates[f'Cases Reported ({year})']
            ))
        
        fig.update_layout(
            title={
                'text': "Crime Rate Trends by State (2020-2022)",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="State/UT",
            yaxis_title="Crime Rate (cases per lakh population)",
            template=self.template,
            barmode='group',
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=600,
            width=1000
        )
        
        # Save interactive plot
        fig.write_html(self.output_dir / 'crime_rate_trends_interactive.html')
        
        # Create static version
        plt.figure(figsize=(20, 12))
        x = np.arange(len(crime_rates['State/UT']))
        width = 0.25
        
        plt.bar(x - width, crime_rates['Crime_Rate_2020'], width, label='2020', color=colors[0])
        plt.bar(x, crime_rates['Crime_Rate_2021'], width, label='2021', color=colors[1])
        plt.bar(x + width, crime_rates['Crime_Rate_2022'], width, label='2022', color=colors[2])
        
        plt.title('Crime Rate Trends by State (2020-2022)', fontsize=16, pad=30)
        plt.xlabel('State/UT', fontsize=12, labelpad=10)
        plt.ylabel('Crime Rate (cases per lakh population)', fontsize=12, labelpad=10)
        plt.xticks(x, crime_rates['State/UT'], rotation=45, ha='right', fontsize=10)
        plt.legend(['2020', '2021', '2022'], title='Year', fontsize=10, title_fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'crime_rate_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_correlation_heatmap(self):
        """Plot interactive correlation heatmap"""
        correlation_matrix = self.df[['Crime_Rate_2020', 'Crime_Rate_2021', 'Crime_Rate_2022',
                                    'Rate of Cognizable Crimes (IPC) (2022)',
                                    'Chargesheeting Rate (2022)',
                                    'Crime_Rate_Change_2021',
                                    'Crime_Rate_Change_2022',
                                    'Crime_Rate_3Y_Avg']].corr()
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
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
        
        fig.update_layout(
            title={
                'text': "Correlation Matrix of Crime Metrics",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            template=self.template,
            height=800,
            width=1000,
            xaxis={'tickangle': 45},
            yaxis={'tickangle': 0}
        )
        
        # Save interactive plot
        fig.write_html(self.output_dir / 'correlation_heatmap_interactive.html')
        
        # Create static version
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True)
        plt.title('Correlation Matrix of Crime Metrics', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300)
        plt.close()

    def plot_top_crime_states(self):
        """Plot top 10 states by crime rate"""
        plt.figure(figsize=(12, 6))
        top_states = self.df.nlargest(10, 'Crime_Rate_2022')
        sns.barplot(data=top_states, x='State/UT', y='Crime_Rate_2022')
        plt.title('Top 10 States by Crime Rate (2022)', fontsize=16)
        plt.xlabel('State/UT', fontsize=14)
        plt.ylabel('Crime Rate per Lakh Population', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_crime_states.png')
        plt.close()
        
    def plot_feature_importance(self, feature_names, importance_scores):
        """Plot interactive feature importance"""
        # Create interactive bar plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_scores,
            y=feature_names,
            orientation='h',
            marker_color=self.color_palette,
            text=importance_scores.round(3),
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>" +
                        "Importance: %{x:.3f}<br>" +
                        "<extra></extra>"
        ))
        
        fig.update_layout(
            title={
                'text': "Feature Importance in Crime Rate Prediction",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            template=self.template,
            height=600,
            width=800,
            showlegend=False
        )
        
        # Save interactive plot
        fig.write_html(self.output_dir / 'feature_importance_interactive.html')
        
        # Create static version
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title('Feature Importance in Crime Rate Prediction', fontsize=16)
        plt.xlabel('Importance Score', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300)
        plt.close()
        
    def plot_distribution_analysis(self):
        """Create interactive distribution analysis plots"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution of Crime Rates (2022)',
                'Crime Rate Distribution by Year',
                'Population vs Crime Rate (2022)',
                'Year-over-Year Changes (2021-2022)'
            )
        )
        
        # Distribution of 2022 crime rates
        fig.add_trace(
            go.Histogram(
                x=self.df['Crime_Rate_2022'],
                name='2022',
                marker_color=self.color_palette[0]
            ),
            row=1, col=1
        )
        
        # Box plot of crime rates by year
        for year, color in zip(['2020', '2021', '2022'], self.color_palette[:3]):
            fig.add_trace(
                go.Box(
                    y=self.df[f'Crime_Rate_{year}'],
                    name=year,
                    marker_color=color
                ),
                row=1, col=2
            )
        
        # Scatter plot of population vs crime rate
        fig.add_trace(
            go.Scatter(
                x=self.df['Population'],
                y=self.df['Crime_Rate_2022'],
                mode='markers',
                marker=dict(
                    color=self.df['Crime_Rate_2022'],
                    colorscale='Reds',
                    showscale=True
                ),
                text=self.df['State/UT'],
                name='States'
            ),
            row=2, col=1
        )
        
        # Year-over-year changes
        fig.add_trace(
            go.Box(
                y=self.df['Crime_Rate_Change_2022'],
                name='2021-2022',
                marker_color=self.color_palette[3]
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=1000,
            width=1200,
            template=self.template,
            showlegend=False,
            title_text="Distribution Analysis of Crime Rates",
            title_x=0.5
        )
        
        # Save interactive plot
        fig.write_html(self.output_dir / 'distribution_analysis_interactive.html')
        
        # Create static version
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        sns.histplot(data=self.df, x='Crime_Rate_2022', kde=True, ax=ax1)
        ax1.set_title('Distribution of Crime Rates (2022)')
        
        crime_rates = pd.melt(self.df, 
                            value_vars=['Crime_Rate_2020', 'Crime_Rate_2021', 'Crime_Rate_2022'],
                            var_name='Year', value_name='Crime Rate')
        sns.boxplot(data=crime_rates, x='Year', y='Crime Rate', ax=ax2)
        ax2.set_title('Crime Rate Distribution by Year')
        
        sns.scatterplot(data=self.df, x='Population', y='Crime_Rate_2022', ax=ax3)
        ax3.set_title('Population vs Crime Rate (2022)')
        
        sns.boxplot(data=self.df, y='Crime_Rate_Change_2022', ax=ax4)
        ax4.set_title('Year-over-Year Changes (2021-2022)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribution_analysis.png', dpi=300)
        plt.close()
        
    def plot_anomaly_detection(self):
        """Plot anomaly detection results"""
        plt.figure(figsize=(12, 6))
        
        # Calculate z-scores for 2022 crime rates
        z_scores = stats.zscore(self.df['Crime_Rate_2022'])
        self.df['Z_Score'] = z_scores
        
        # Plot scatter plot with anomalies highlighted
        plt.scatter(self.df['State/UT'], self.df['Crime_Rate_2022'], 
                   c=np.abs(z_scores) > 2, cmap='coolwarm')
        
        # Add labels for anomalies
        anomalies = self.df[np.abs(z_scores) > 2]
        for idx, row in anomalies.iterrows():
            plt.annotate(row['State/UT'], 
                        (row['State/UT'], row['Crime_Rate_2022']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Crime Rate Anomalies (2022)', fontsize=16)
        plt.xlabel('State/UT', fontsize=14)
        plt.ylabel('Crime Rate per Lakh Population', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_detection.png')
        plt.close()
        
    def plot_state_wise_map(self):
        """Create interactive choropleth map of crime rates"""
        try:
            gdf = gpd.read_file('India_GeoJSON.json')
            merged = gdf.merge(self.df, left_on='name', right_on='State/UT', how='left')
            
            # Create interactive choropleth map
            fig = px.choropleth(
                merged,
                geojson=merged.geometry,
                locations=merged.index,
                color='Crime_Rate_2022',
                hover_name='State/UT',
                hover_data={
                    'Crime_Rate_2022': ':.2f',
                    'Cases Reported (2022)': True,
                    'Population': True
                },
                color_continuous_scale='Reds',
                title='Crime Rate by State (2022)',
                template=self.template
            )
            
            fig.update_geos(
                fitbounds="locations",
                visible=False
            )
            
            fig.update_layout(
                height=800,
                width=1000,
                margin={"r":0,"t":30,"l":0,"b":0}
            )
            
            # Save interactive plot
            fig.write_html(self.output_dir / 'state_wise_map_interactive.html')
            
            # Create static version
            fig, ax = plt.subplots(1, 1, figsize=(14, 14))
            merged.plot(
                column='Crime_Rate_2022',
                cmap='Reds',
                linewidth=0.8,
                ax=ax,
                edgecolor='0.8',
                legend=True
            )
            ax.set_title('Crime Rate by State (2022)', fontdict={'fontsize': 18})
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'state_wise_map.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create state-wise map: {str(e)}")
            print("Continuing with other visualizations...")
        
    def plot_top_states_trend(self):
        """Plot trend lines for top 5 states by average crime rate"""
        # Calculate average crime rate for each state
        avg_crime_rates = self.df[['State/UT', 'Crime_Rate_2020', 'Crime_Rate_2021', 'Crime_Rate_2022']].copy()
        avg_crime_rates['Avg_Crime_Rate'] = avg_crime_rates[['Crime_Rate_2020', 'Crime_Rate_2021', 'Crime_Rate_2022']].mean(axis=1)
        
        # Get top 5 states
        top_states = avg_crime_rates.nlargest(5, 'Avg_Crime_Rate')['State/UT'].tolist()
        
        # Create trend plot
        plt.figure(figsize=(10, 6))
        for state in top_states:
            state_data = self.df[self.df['State/UT'] == state]
            plt.plot([2020, 2021, 2022],
                     [state_data['Crime_Rate_2020'].iloc[0],
                      state_data['Crime_Rate_2021'].iloc[0],
                      state_data['Crime_Rate_2022'].iloc[0]],
                     marker='o', label=state)
        plt.title('Crime Rate Trends for Top 5 States (2020-2022)', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Crime Rate per Lakh Population', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'top_states_trend.png')
        plt.close()
        
    def plot_pairplot(self):
        """Create pairplot of numeric features"""
        numeric_cols = ['Crime_Rate_2020', 'Crime_Rate_2021', 'Crime_Rate_2022',
                       'Rate of Cognizable Crimes (IPC) (2022)',
                       'Chargesheeting Rate (2022)']
        
        plt.figure(figsize=(15, 15))
        sns.pairplot(self.df[numeric_cols], diag_kind='kde')
        plt.savefig(self.output_dir / 'pairplot.png')
        plt.close()
        
    def plot_year_over_year_changes(self):
        """Plot heatmap of year-over-year changes"""
        changes = pd.DataFrame({
            'State/UT': self.df['State/UT'],
            '2020-2021': self.df['Crime_Rate_Change_2021'],
            '2021-2022': self.df['Crime_Rate_Change_2022']
        })
        plt.figure(figsize=(12, 10))
        sns.heatmap(changes.set_index('State/UT'), annot=True, cmap='RdBu_r', center=0, fmt='.2f')
        plt.title('Year-over-Year Changes in Crime Rates', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'year_over_year_changes.png')
        plt.close()

if __name__ == "__main__":
    # Example usage
    from scripts.preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor("data/raw_data.csv")
    df = preprocessor.clean_data()
    df = preprocessor.calculate_crime_rates()
    
    # Create visualizations
    plotter = GeoPlotter(df)
    plotter.plot_crime_trends()
    plotter.plot_correlation_heatmap()
    plotter.plot_top_crime_states()
    plotter.plot_distribution_analysis()
    plotter.plot_anomaly_detection()
    plotter.plot_state_wise_map()
    plotter.plot_top_states_trend()
    plotter.plot_pairplot()
    plotter.plot_year_over_year_changes() 