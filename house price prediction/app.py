from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and preprocessor
model = joblib.load('house_price_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

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
label_encoder_location = joblib.load('label_encoder_location.pkl')
label_encoder_size = joblib.load('label_encoder_size.pkl')

# Get unique values for dropdowns
locations = label_encoder_location.inverse_transform(df['location'].unique())
sizes = label_encoder_size.inverse_transform(df['size'].unique())
num_bathrooms = sorted(df['bath'].dropna().unique())
num_balconies = sorted(df['balcony'].dropna().unique())

@app.route('/')
def home():
    return render_template('index.html', locations=locations, sizes=sizes, num_bathrooms=num_bathrooms, num_balconies=num_balconies)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from form
        location = request.form['location']
        size = request.form['size']
        total_sqft = float(request.form['total_sqft'])
        bath = float(request.form['bath'])
        balcony = float(request.form['balcony'])

        # Check if location and size are valid
        if location not in label_encoder_location.classes_ or size not in label_encoder_size.classes_:
            return jsonify({'error': 'Invalid location or size'})

        # Encode the categorical features
        location_encoded = label_encoder_location.transform([location])[0]
        size_encoded = label_encoder_size.transform([size])[0]

        # Create a DataFrame for the input
        input_features = pd.DataFrame([[location_encoded, size_encoded, total_sqft, bath, balcony]],
                                      columns=['location', 'size', 'total_sqft', 'bath', 'balcony'])

        # Apply the same preprocessor
        input_features_scaled = preprocessor.transform(input_features)

        # Predict the price using the model
        prediction = model.predict(input_features_scaled)
        predicted_price = prediction[0]

        # For demonstration purposes, find the nearest area type (if needed)
        area_type = label_encoder_size.inverse_transform([size_encoded])[0]

        return jsonify({
            'predicted_price': predicted_price,
            'total_sqft': total_sqft,
            'area_type': area_type
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
