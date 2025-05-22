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
import matplotlib.ticker as mticker

class GeoPlotter:
    def __init__(self, df):
        self.df = df
        self.output_dir = Path("output/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_crime_trends(self):
        """Plot crime rate trends by state"""
        # Calculate crime rates per lakh population correctly (no multiplier needed)
        crime_rates = self.df[['State/UT', 'Cases Reported (2020)', 'Cases Reported (2021)', 'Cases Reported (2022)', 'Population']].copy()
        
        # Calculate rates per lakh population (population is already in lakhs)
        crime_rates['Crime_Rate_2020'] = crime_rates['Cases Reported (2020)'] / crime_rates['Population']
        crime_rates['Crime_Rate_2021'] = crime_rates['Cases Reported (2021)'] / crime_rates['Population']
        crime_rates['Crime_Rate_2022'] = crime_rates['Cases Reported (2022)'] / crime_rates['Population']
        
        # Sort states by average crime rate for better visualization
        crime_rates['Avg_Crime_Rate'] = crime_rates[['Crime_Rate_2020', 'Crime_Rate_2021', 'Crime_Rate_2022']].mean(axis=1)
        crime_rates = crime_rates.sort_values('Avg_Crime_Rate', ascending=False)
        
        # Create the plot with larger figure size and more vertical space
        plt.figure(figsize=(20, 12))
        
        # Create grouped bar plot
        x = np.arange(len(crime_rates['State/UT']))
        width = 0.25
        
        plt.bar(x - width, crime_rates['Crime_Rate_2020'], width, label='2020', color='#1f77b4')
        plt.bar(x, crime_rates['Crime_Rate_2021'], width, label='2021', color='#ff7f0e')
        plt.bar(x + width, crime_rates['Crime_Rate_2022'], width, label='2022', color='#2ca02c')
        
        # Customize the plot with adjusted spacing
        plt.title('Crime Rate Trends by State (2020-2022)', fontsize=16, pad=30)
        plt.suptitle('Crime Rate (cases per lakh population)', fontsize=12, y=0.98)
        
        # Format y-axis with reasonable numbers
        def format_y_axis(x, pos):
            return f'{x:.1f}'  # Show one decimal place
        
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(format_y_axis))
        
        # Set reasonable y-axis limits
        plt.ylim(0, max(crime_rates[['Crime_Rate_2020', 'Crime_Rate_2021', 'Crime_Rate_2022']].max()) * 1.1)
        
        # Customize labels and ticks
        plt.xlabel('State/UT', fontsize=12, labelpad=10)
        plt.ylabel('Crime Rate (cases per lakh population)', fontsize=12, labelpad=10)
        plt.xticks(x, crime_rates['State/UT'], rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        # Add legend with better formatting
        plt.legend(['2020', '2021', '2022'], title='Year', fontsize=10, title_fontsize=12, 
                  bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout with more padding
        plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])  # Adjust rect to make room for suptitle
        
        # Add explanatory text with adjusted position
        plt.figtext(0.5, 0.01, 
                    'Note: Crime Rate = (Number of Cases / Population in Lakhs)', 
                    ha='center', fontsize=10, color='gray')
        
        # Save the plot with adjusted padding
        plt.savefig(self.output_dir / 'crime_rate_trends.png', 
                    bbox_inches='tight',
                    pad_inches=0.5,
                    dpi=300)
        plt.close()
        
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of crime metrics"""
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[['Crime_Rate_2020', 'Crime_Rate_2021', 'Crime_Rate_2022',
                                    'Rate of Cognizable Crimes (IPC) (2022)',
                                    'Chargesheeting Rate (2022)',
                                    'Crime_Rate_Change_2021',
                                    'Crime_Rate_Change_2022',
                                    'Crime_Rate_3Y_Avg']].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, fmt='.2f', square=True)
        plt.title('Correlation Matrix of Crime Metrics', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png')
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
        """Plot feature importance from the model"""
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
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close()
        
    def plot_distribution_analysis(self):
        """Plot distribution analysis of crime rates"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution of 2022 crime rates
        sns.histplot(data=self.df, x='Crime_Rate_2022', kde=True, ax=ax1)
        ax1.set_title('Distribution of Crime Rates (2022)', fontsize=16)
        ax1.set_xlabel('Crime Rate per Lakh Population', fontsize=14)
        
        # Box plot of crime rates by year
        crime_rates = pd.melt(self.df, 
                            value_vars=['Crime_Rate_2020', 'Crime_Rate_2021', 'Crime_Rate_2022'],
                            var_name='Year', value_name='Crime Rate')
        sns.boxplot(data=crime_rates, x='Year', y='Crime Rate', ax=ax2)
        ax2.set_title('Crime Rate Distribution by Year', fontsize=16)
        
        # Scatter plot of population vs crime rate
        sns.scatterplot(data=self.df, x='Population', y='Crime_Rate_2022', ax=ax3)
        ax3.set_title('Population vs Crime Rate (2022)', fontsize=16)
        ax3.set_xlabel('Population', fontsize=14)
        ax3.set_ylabel('Crime Rate per Lakh Population', fontsize=14)
        
        # Year-over-year changes
        sns.boxplot(data=self.df, y='Crime_Rate_Change_2022', ax=ax4)
        ax4.set_title('Distribution of Year-over-Year Changes (2021-2022)', fontsize=16)
        ax4.set_ylabel('Change in Crime Rate', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribution_analysis.png')
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
        """Create choropleth map of crime rates by state"""
        try:
            gdf = gpd.read_file('india_GeoJSON.json')
            merged = gdf.merge(self.df, left_on='name', right_on='State/UT', how='left')
            fig, ax = plt.subplots(1, 1, figsize=(14, 14))
            cbar = merged.plot(column='Crime_Rate_2022', cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
            ax.set_title('Crime Rate by State (2022)', fontdict={'fontsize': 18})
            ax.axis('off')
            # Format colorbar to Indian number system
            colorbar = cbar.get_figure().axes[-1]
            def indian_fmt(x, pos):
                if x >= 1e7:
                    return f'{x/1e7:.1f} Cr'
                elif x >= 1e5:
                    return f'{x/1e5:.1f} L'
                elif x >= 1e3:
                    return f'{x/1e3:.1f} K'
                else:
                    return f'{x:.0f}'
            colorbar.yaxis.set_major_formatter(mticker.FuncFormatter(indian_fmt))
            plt.tight_layout()
            plt.savefig(self.output_dir / 'state_wise_map.png', bbox_inches='tight', pad_inches=0.3)
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