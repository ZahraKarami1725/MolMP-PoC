# mol_predictor.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Cheminformatics Libraries
from rdkit import Chem
from rdkit.Chem import Descriptors 

# ML Libraries
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# --- Configuration (CRITICAL) ---
USE_EXTERNAL_DATA = True 
DATA_PATH = 'data/train.csv'

# **CRITICAL: Column names derived from your uploaded image**
SMILES_COL = 'SMILES'       # Column B
TARGET_PROPERTY = 'Tm'      # Column C (Assumed to be Melting Point in K or C)
# --------------------

if not os.path.exists('results'):
    os.makedirs('results')

# === 1. Data Loading and Preparation (External Data Mode) ===
def load_and_clean_kaggle_data(path, smiles_col, target_col):
    """Loads Kaggle CSV and selects ONLY the necessary SMILES and Target columns."""
    print("1.1 Loading External Data...")
    try:
        df = pd.read_csv(path)
        
        # **ACTION: Select only the required columns and discard ALL Group columns**
        df = df[[smiles_col, target_col]].copy()
        
        # Rename columns to standardized names for internal consistency
        df = df.rename(columns={smiles_col: 'SMILES', target_col: 'TARGET'})
        
        # Initial cleaning: drop rows with missing values in key columns
        df = df.dropna(subset=['SMILES', 'TARGET'])
        
        # Convert Target (Tm) to numeric, dropping invalid rows
        df['TARGET'] = pd.to_numeric(df['TARGET'], errors='coerce')
        df = df.dropna(subset=['TARGET'])
        
        print(f"   Loaded and cleaned {len(df)} samples, keeping ONLY SMILES and Tm.")
        return df
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found at {path}. Please check your path.")
        return None
    except KeyError as e:
        print(f"FATAL ERROR: Column {e} not found. Check if SMILES_COL and TARGET_PROPERTY are correct.")
        return None

df_raw = load_and_clean_kaggle_data(DATA_PATH, SMILES_COL, TARGET_PROPERTY)
if df_raw is None:
    exit()

# === 2. Feature Generation: RDKit Descriptors (Preprocessing Core) ===
def generate_rdkit_descriptors(df):
    """Generates RDKit descriptors for the SMILES column."""
    print("2.1 Generating RDKit Descriptors (Features)...")
    
    descriptor_names = [d[0] for d in Descriptors.descList]
    descriptor_list = []
    
    for i, smiles in enumerate(df['SMILES']):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            descriptors = [np.nan] * len(descriptor_names)
        else:
            try:
                descriptors = [d[1](mol) for d in Descriptors.descList]
            except:
                descriptors = [np.nan] * len(descriptor_names)

        descriptor_list.append(descriptors)
        
    X_desc_df = pd.DataFrame(descriptor_list, columns=descriptor_names)
    X_data = pd.concat([df.reset_index(drop=True), X_desc_df], axis=1)
    
    # Final cleaning: drop samples where descriptor calculation resulted in NaN
    X_data = X_data.dropna() 

    X = X_data.drop(columns=['SMILES', 'TARGET']).select_dtypes(include=np.number)
    y = X_data['TARGET']
    
    print(f"   Generated {X.shape[1]} descriptors. Final samples for modeling: {X.shape[0]}")
    return X, y

X, y = generate_rdkit_descriptors(df_raw)
print("-" * 50)

# === 3. Data Splitting and Scaling ===
print("3. Splitting and Scaling Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

print(f"   Training set size: {X_train_scaled.shape[0]}")
print("-" * 50)


# === 4. Model 1: XGBoost Regression ===
print("4. Training XGBoost Model...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=6, 
    random_state=42, 
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)
xgb_r2 = r2_score(y_test, xgb_predictions)
print(f"   XGBoost Results: R2 Score={xgb_r2:.3f}")


# === 5. Model 2: Artificial Neural Network (ANN) ===
print("5. Training Neural Network (ANN) Model...")
N_FEATURES = X_train_scaled.shape[1]

ann_model = Sequential([
    Dense(512, activation='relu', input_shape=(N_FEATURES,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(1) 
])

ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

history = ann_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    verbose=0
)

ann_predictions = ann_model.predict(X_test_scaled).flatten()
ann_r2 = r2_score(y_test, ann_predictions)
print(f"   ANN Results: R2 Score={ann_r2:.3f}")
print("-" * 50)


# === 6. Visualization and Saving ===
print("6. Generating Comparison Plot and Saving Models...")

plt.figure(figsize=(10, 5))

# Plot 1: XGBoost Actual vs Predicted
plt.scatter(y_test, xgb_predictions, alpha=0.7, label=f'XGBoost (R2={xgb_r2:.3f})', color='darkorange')
plt.scatter(y_test, ann_predictions, alpha=0.7, label=f'ANN (R2={ann_r2:.3f})', color='darkblue')

min_val = min(y_test.min(), xgb_predictions.min(), ann_predictions.min())
max_val = max(y_test.max(), xgb_predictions.max(), ann_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

plt.title('Melting Point Prediction Comparison (Actual vs. Predicted)')
plt.xlabel('Actual Tm (Melting Point)')
plt.ylabel('Predicted Tm (Melting Point)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/model_comparison.png')

# Save trained models
xgb_model.save_model('results/xgb_mp_predictor.json')
ann_model.save('results/ann_mp_predictor.h5')
print("7. Project Complete. Models and plot saved successfully.")