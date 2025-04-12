import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load dataset
df = pd.read_csv("../data/irrigation_dataset.csv")

# Simulate user-defined inputs (normally you'd have this in your dataset)
df['Crop Type'] = np.random.choice(['wheat', 'rice', 'maize'], size=len(df))
df['Growth Stage'] = np.random.choice(['early', 'mid', 'late'], size=len(df))
df['Irrigation Type'] = np.random.choice(['drip', 'sprinkler', 'surface'], size=len(df))

# Preprocessing
le_crop = LabelEncoder()
le_stage = LabelEncoder()
le_irrigation = LabelEncoder()
le_growth = LabelEncoder()

df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
df['Growth Stage'] = le_stage.fit_transform(df['Growth Stage'])
df['Irrigation Type'] = le_irrigation.fit_transform(df['Irrigation Type'])
df['Growth Stage'] = le_growth.fit_transform(df['Growth Stage'])


df['Day of Year'] = pd.to_datetime(df['Date'], dayfirst=True).dt.dayofyear

features = [
    'Mean Temp (C)', 'Max Temp (C)', 'Min Temp (C)', 'R. Humidity (%)',
    'Wind Speed (m/s)', 'S. Pressure (kPa)', 'Net Radiation (MJ/m^2/day)',
    'Day of Year', 'Crop Type', 'Growth Stage', 'Irrigation Type'
]
target = 'ET0 (mm/day)'

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
xgb_model = GradientBoostingRegressor()
xgb_model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(xgb_model, '../model/xgb_model.pkl')
joblib.dump(scaler, '../model/scaler.pkl')

# Save encoders
joblib.dump(le_crop, '../model/le_crop.pkl')
joblib.dump(le_stage, '../model/le_stage.pkl')
joblib.dump(le_irrigation, '../model/le_irrigation.pkl')
joblib.dump(le_growth, '../model/le_growth.pkl')
