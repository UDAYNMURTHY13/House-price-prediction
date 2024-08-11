from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load and preprocess data
file_path = r'C:\Users\Admin\Desktop\house price prediction\Bengaluru_House_Data.csv'
df = pd.read_csv(file_path)

# Convert 'total_sqft' to numeric
def convert_sqft_to_numeric(sqft_str):
    if isinstance(sqft_str, str):
        try:
            if '-' in sqft_str:
                parts = sqft_str.split(' - ')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(sqft_str)
        except ValueError:
            return None
    return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_numeric)
df['bath'] = pd.to_numeric(df['bath'], errors='coerce')
df['balcony'] = pd.to_numeric(df['balcony'], errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical features
categorical_features = ['location', 'size']
numerical_features = ['total_sqft', 'bath', 'balcony']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the model pipeline with Linear Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Define features and target
X = df[['location', 'size', 'total_sqft', 'bath', 'balcony']]
y = df['price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the model and preprocessor
try:
    joblib.dump(model, 'house_price_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("Model and preprocessor saved successfully!")
except Exception as e:
    print(f"Error saving model or preprocessor: {e}")
