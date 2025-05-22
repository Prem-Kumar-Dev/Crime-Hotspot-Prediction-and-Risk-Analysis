"""
Main entry point for Crime Hotspot Prediction & Risk Analysis.
Coordinates the entire workflow from data preprocessing to visualization.
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scripts.preprocessing import DataPreprocessor
from scripts.train_model import CrimePredictor
from utils.geo_plotter import GeoPlotter

def main():
    # Create necessary directories
    for dir_path in ['data', 'models', 'output/plots', 'reports']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("Starting Crime Hotspot Prediction & Risk Analysis...")
    
    # Step 1: Data Preprocessing
    print("\n1. Preprocessing data...")
    preprocessor = DataPreprocessor("data/raw_data.csv")
    df = preprocessor.load_data()
    print("\nValidating data...")
    preprocessor.validate_data()
    df = preprocessor.clean_data()
    df = preprocessor.calculate_crime_rates()
    preprocessor.save_cleaned_data()
    
    # Step 2: Model Training
    print("\n2. Training ML model...")
    X, y = preprocessor.get_features()
    predictor = CrimePredictor()
    metrics = predictor.train(X, y)
    predictor.save_model()
    
    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {metrics['mse']:.2f}")
    print(f"R2 Score: {metrics['r2']:.2f}")
    
    # Step 3: Visualization
    print("\n3. Creating visualizations...")
    plotter = GeoPlotter(df)
    
    # Create various plots
    print("Generating crime rate trends...")
    plotter.plot_crime_trends()
    
    print("Generating correlation analysis...")
    plotter.plot_correlation_heatmap()
    
    print("Generating top crime states analysis...")
    plotter.plot_top_crime_states()
    
    print("Generating distribution analysis...")
    plotter.plot_distribution_analysis()
    
    print("Generating anomaly detection...")
    plotter.plot_anomaly_detection()
    
    print("Generating state-wise map...")
    plotter.plot_state_wise_map()
    
    print("Generating top states trend analysis...")
    plotter.plot_top_states_trend()
    
    print("Generating pairplot...")
    plotter.plot_pairplot()
    
    print("Generating year-over-year changes...")
    plotter.plot_year_over_year_changes()
    
    # Plot feature importance
    print("Generating feature importance plot...")
    feature_names = ['Population', 'Crime Rate (IPC)', 'Chargesheeting Rate',
                    'Crime Rate 2020', 'Crime Rate 2021', 'Crime Rate Change 2021',
                    '3-Year Average']
    importance_scores = predictor.get_feature_importance()
    plotter.plot_feature_importance(feature_names, importance_scores)
    
    print("\nAnalysis complete! Check the following outputs:")
    print("1. Cleaned data: data/cleaned_data.csv")
    print("2. Summary statistics: data/summary_statistics.txt")
    print("3. Trained model: models/rf_model.pkl")
    print("4. Visualizations in output/plots/:")
    print("   Static plots:")
    print("   - crime_rate_trends.png")
    print("   - correlation_heatmap.png")
    print("   - top_crime_states.png")
    print("   - distribution_analysis.png")
    print("   - anomaly_detection.png")
    print("   - feature_importance.png")
    print("   - pairplot.png")
    print("\n   Interactive plots:")
    print("   - crime_rate_trends_interactive.html")
    print("   - correlation_heatmap_interactive.html")
    print("   - top_crime_states_interactive.html")
    print("   - distribution_analysis_interactive.html")
    print("   - anomaly_detection_interactive.html")
    print("   - feature_importance_interactive.html")
    print("   - state_wise_map.html")
    print("   - top_states_trend.html")
    print("   - year_over_year_changes.html")

if __name__ == "__main__":
    main() 