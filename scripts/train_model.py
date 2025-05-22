"""
Model training module for crime analysis.
Handles model training, evaluation, and saving.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
from pathlib import Path

class CrimePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def train(self, X, y):
        """Train the ML model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        evaluation_metrics = {
            'mse': mse,
            'r2': r2,
            'X_test': X_test_scaled,
            'y_test': y_test
        }
        
        return evaluation_metrics
    
    def save_model(self):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
            
        # Save model
        model_path = self.models_dir / "rf_model.pkl"
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = self.models_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
            
        return self.model.feature_importances_

if __name__ == "__main__":
    # Example usage
    from scripts.preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor("data/raw_data.csv")
    df = preprocessor.clean_data()
    df = preprocessor.calculate_crime_rates()
    
    # Get features
    X, y = preprocessor.get_features()
    
    # Train model
    predictor = CrimePredictor()
    metrics = predictor.train(X, y)
    
    # Save model
    predictor.save_model()
    
    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {metrics['mse']:.2f}")
    print(f"R2 Score: {metrics['r2']:.2f}") 