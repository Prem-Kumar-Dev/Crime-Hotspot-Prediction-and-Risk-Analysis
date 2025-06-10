# Crime Hotspot Prediction & Risk Analysis

A comprehensive data visualization and analysis project that explores crime patterns across Indian states from 2020 to 2022. This project combines statistical analysis, geographical visualization, and machine learning to provide insights into crime trends and patterns.

## Features

### Interactive Dashboard
- **Modern UI**: Beautiful and intuitive interface built with Dash and Bootstrap
- **Year-wise Analysis**: Compare crime data across 2020, 2021, and 2022
- **Multiple Visualization Types**:
  - Crime rate trends and comparisons
  - Geographical distribution maps
  - Statistical correlations
  - Distribution analysis
  - Feature importance
  - Anomaly detection

### Key Components
1. **Overview Analysis**
   - Crime rate trends by state
   - Top crime states identification
   - Year-over-year comparisons

2. **Geographical Analysis**
   - Interactive choropleth maps
   - State-wise crime distribution
   - Regional pattern identification

3. **Statistical Analysis**
   - Correlation heatmaps
   - Distribution analysis
   - Population vs. crime rate relationships

4. **Predictive Analysis**
   - Feature importance visualization
   - Anomaly detection
   - Risk assessment

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Crime-Hotspot-Prediction-and-Risk-Analysis.git
   cd Crime-Hotspot-Prediction-and-Risk-Analysis
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   python dashboard.py
   ```

5. **Access the dashboard**
   - Open your web browser
   - Go to http://localhost:8050

## Project Structure

```
Crime-Hotspot-Prediction-and-Risk-Analysis/
├── data/
│   ├── raw_data.csv          # Original crime dataset
│   ├── cleaned_data.csv      # Preprocessed data
│   └── summary_statistics.txt # Statistical summary of the data
├── models/
│   ├── rf_model.pkl          # Trained Random Forest model
│   └── scaler.pkl            # Data scaler for model input
├── scripts/
│   ├── preprocessing.py      # Data cleaning and preprocessing
│   ├── train_model.py        # Model training script
│   └── predict.py            # Prediction script
├── utils/
│   └── geo_plotter.py        # Geographical plotting utilities
├── output/
│   └── plots/                # Generated visualizations
├── reports/
│   └── analysis/             # Analysis reports and findings
├── dashboard.py              # Interactive dashboard application
├── main.py                   # Main application entry point
├── India_GeoJSON.json        # Geographical data for India
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Dependencies

The project requires the following Python packages:

```
dash==2.14.2
dash-bootstrap-components==1.5.0
plotly==5.18.0
pandas==2.1.4
geopandas==0.14.1
scikit-learn==1.3.2
joblib==1.3.2
numpy==1.26.2
matplotlib>=3.4.0
seaborn>=0.11.0
kaleido>=0.2.1
```

## Usage Guide

1. **Dashboard Navigation**
   - Use the year selector to filter data (2020-2022)
   - Navigate between tabs for different analyses
   - Hover over visualizations for detailed information
   - Use zoom and pan features on the map

2. **Understanding Visualizations**
   - Color intensity indicates severity/importance
   - Hover tooltips provide detailed information
   - Interactive elements allow for deeper exploration

3. **Key Features**
   - Real-time data filtering
   - Interactive geographical maps
   - Statistical correlation analysis
   - Anomaly detection
   - Feature importance visualization

## Data Source

- **Source**: National Crime Records Bureau (NCRB)
- **Time Period**: 2020-2022
- **Coverage**: All Indian states and union territories
- **Data Types**: Crime rates, population statistics, geographical data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development

### Prerequisites
- Python 3.8 or higher
- Git
- pip (Python package installer)

### Local Development
1. Clone the repository
2. Create and activate virtual environment
3. Install dependencies
4. Run the dashboard locally
5. Make changes and test
6. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- National Crime Records Bureau for providing the crime data
- Dash and Plotly for the visualization framework
- Contributors and maintainers of the project
- Open source community for various tools and libraries

## Support

For support, please:
1. Check the existing issues
2. Create a new issue if needed
3. Provide detailed information about the problem
4. Include steps to reproduce if applicable
