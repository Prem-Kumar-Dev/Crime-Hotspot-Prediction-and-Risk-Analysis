# Crime Hotspot Prediction & Risk Analysis

This project analyzes crime data across Indian states and predicts crime risk using machine learning.

## Project Structure
```
📁 crime-hotspot-prediction/
├── 📄 README.md ← Project overview & instructions
├── 📄 requirements.txt ← List of Python dependencies
├── 📄 main.py ← Entry point
├── 📁 data/
│   ├── 📄 raw_data.csv ← Original crime dataset
│   └── 📄 cleaned_data.csv ← Preprocessed data
├── 📁 notebooks/
│   ├── 📄 1_data_cleaning.ipynb
│   ├── 📄 2_feature_engineering.ipynb
│   ├── 📄 3_eda.ipynb
│   ├── 📄 4_model_training.ipynb
│   └── 📄 5_evaluation.ipynb
├── 📁 scripts/
│   ├── 📄 preprocessing.py
│   ├── 📄 train_model.py
│   └── 📄 predict.py
├── 📁 models/
│   └── 📄 rf_model.pkl
├── 📁 output/
│   └── 📁 plots/
├── 📁 utils/
│   └── 📄 geo_plotter.py
└── 📁 reports/
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis:
```bash
python main.py
```

## Project Components

1. **Data Preprocessing**
   - Cleaning missing values
   - Converting data types
   - Calculating crime rates

2. **Feature Engineering**
   - Temporal features
   - Population-based metrics
   - Crime rate calculations

3. **Exploratory Data Analysis**
   - Crime rate trends
   - State-wise analysis
   - Correlation analysis

4. **Model Training**
   - Random Forest Regressor
   - Feature importance analysis
   - Model evaluation

5. **Visualization**
   - Crime rate trends
   - Correlation heatmaps
   - Top crime states
   - Feature importance plots

## Output

The project generates:
- Preprocessed data in `data/cleaned_data.csv`
- Trained model in `models/rf_model.pkl`
- Visualizations in `output/plots/`
- Analysis notebooks in `notebooks/`

## Requirements
- Python 3.8+
- Dependencies listed in requirements.txt 