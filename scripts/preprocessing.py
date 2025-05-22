"""
Data preprocessing module for crime analysis.
Handles data cleaning, feature engineering, and data validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy import stats

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.output_dir = Path("data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and perform initial data validation"""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded data with {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df
        
    def validate_data(self):
        """Validate data integrity and consistency"""
        # Check for missing values
        missing_values = self.df.isnull().sum()
        print("\nMissing values per column:")
        print(missing_values[missing_values > 0])
        
        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        print(f"\nNumber of duplicate rows: {duplicates}")
        
        # Check data types
        print("\nData types:")
        print(self.df.dtypes)
        
        return missing_values, duplicates
        
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.df is None:
            self.load_data()
            
        # Remove total rows
        self.df = self.df[~self.df['State/UT'].str.contains('Total', case=False, na=False)]
        
        # Rename columns for consistency
        self.df = self.df.rename(columns={
            '2020': 'Cases Reported (2020)',
            '2021': 'Cases Reported (2021)',
            '2022': 'Cases Reported (2022)',
            'Mid-Year Projected Population (in Lakhs) (2022)': 'Population'
        })
        
        # Convert columns to numeric, handling errors
        numeric_columns = [
            'Population', 'Rate of Cognizable Crimes (IPC) (2022)',
            'Chargesheeting Rate (2022)', 'Cases Reported (2020)',
            'Cases Reported (2021)', 'Cases Reported (2022)'
        ]
        
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
        # Handle missing values
        self.df = self.handle_missing_values()
        
        # Handle outliers
        self.df = self.handle_outliers()
        
        return self.df
        
    def handle_missing_values(self):
        """Handle missing values using appropriate strategies"""
        # For numeric columns, use median imputation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
            
        # For categorical columns, use mode imputation
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            
        return self.df
        
    def handle_outliers(self):
        """Handle outliers using IQR method"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            self.df[col] = self.df[col].clip(lower_bound, upper_bound)
            
        return self.df
        
    def calculate_crime_rates(self):
        """Calculate crime rates and additional features"""
        # Calculate crime rates per lakh population (population is already in lakhs)
        for year in [2020, 2021, 2022]:
            self.df[f'Crime_Rate_{year}'] = self.df[f'Cases Reported ({year})'] / self.df['Population']
        
        # Calculate year-over-year changes
        self.df['Crime_Rate_Change_2021'] = self.df['Crime_Rate_2021'] - self.df['Crime_Rate_2020']
        self.df['Crime_Rate_Change_2022'] = self.df['Crime_Rate_2022'] - self.df['Crime_Rate_2021']
        
        # Calculate rolling average of crime rates
        self.df['Crime_Rate_3Y_Avg'] = self.df[['Crime_Rate_2020', 'Crime_Rate_2021', 
                                               'Crime_Rate_2022']].mean(axis=1)
        
        return self.df
        
    def get_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'basic_stats': self.df[numeric_cols].describe(),
            'correlation_matrix': self.df[numeric_cols].corr(),
            'skewness': self.df[numeric_cols].skew(),
            'kurtosis': self.df[numeric_cols].kurtosis()
        }
        
        return summary
        
    def get_features(self):
        """Extract features and target variable for machine learning"""
        feature_columns = [
            'Population', 'Rate of Cognizable Crimes (IPC) (2022)',
            'Chargesheeting Rate (2022)', 'Crime_Rate_2020',
            'Crime_Rate_2021', 'Crime_Rate_Change_2021',
            'Crime_Rate_3Y_Avg'
        ]
        
        X = self.df[feature_columns]
        y = self.df['Crime_Rate_2022']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
        
        return X_scaled, y
        
    def save_cleaned_data(self):
        """Save cleaned data and summary statistics"""
        # Save cleaned data
        output_path = self.output_dir / "cleaned_data.csv"
        self.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        
        # Save summary statistics
        summary = self.get_summary_statistics()
        summary_path = self.output_dir / "summary_statistics.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=== Basic Statistics ===\n")
            f.write(str(summary['basic_stats']))
            f.write("\n\n=== Correlation Matrix ===\n")
            f.write(str(summary['correlation_matrix']))
            f.write("\n\n=== Skewness ===\n")
            f.write(str(summary['skewness']))
            f.write("\n\n=== Kurtosis ===\n")
            f.write(str(summary['kurtosis']))
            
        print(f"Summary statistics saved to {summary_path}")

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor("data/raw_data.csv")
    df = preprocessor.load_data()
    preprocessor.validate_data()
    df = preprocessor.clean_data()
    df = preprocessor.calculate_crime_rates()
    preprocessor.save_cleaned_data() 