import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def create_sample_data():
    """Create sample insurance data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    age = np.random.randint(18, 65, n_samples)
    sex = np.random.choice(['male', 'female'], n_samples)
    bmi = np.random.normal(30, 6, n_samples)
    bmi = np.clip(bmi, 15, 50)  # Clip to reasonable range
    children = np.random.poisson(1, n_samples)
    children = np.clip(children, 0, 5)
    smoker = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
    region = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples)
    
    # Generate charges with some realistic relationships
    charges = (
        250 * age +  # Base cost increases with age
        (bmi - 25) * 50 +  # BMI impact
        children * 500 +  # Cost per child
        np.where(smoker == 'yes', 20000, 0) +  # Smoking penalty
        np.random.normal(0, 5000, n_samples)  # Random variation
    )
    charges = np.maximum(charges, 1000)  # Minimum charge
    
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region,
        'charges': charges
    })
    
    return df

def preprocess_data(df):
    """Preprocess the data for training"""
    # Encode categorical variables
    df_processed = df.copy()
    
    # Binary encoding
    df_processed['sex'] = (df_processed['sex'] == 'male').astype(int)
    df_processed['smoker'] = (df_processed['smoker'] == 'yes').astype(int)
    
    # One-hot encoding for region
    region_dummies = pd.get_dummies(df_processed['region'], prefix='region')
    df_processed = pd.concat([df_processed, region_dummies], axis=1)
    df_processed.drop('region', axis=1, inplace=True)
    
    return df_processed

def train_model():
    """Train the insurance cost prediction model"""
    print("Creating sample data...")
    df = create_sample_data()
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Separate features and target
    X = df_processed.drop('charges', axis=1)
    y = df_processed['charges']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training model...")
    # Try different models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = float('-inf')
    best_name = None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name}:")
        print(f"  MSE: {mse:.2f}")
        print(f"  R²: {r2:.4f}")
        print()
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
    
    print(f"Best model: {best_name} (R² = {best_score:.4f})")
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/insurance_model.pkl')
    print("Model saved to models/insurance_model.pkl")
    
    return best_model

if __name__ == "__main__":
    train_model()