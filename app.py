from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained model and feature column order
model = joblib.load('xgb_sales_model.pkl')
feature_columns = joblib.load('xgb_feature_columns.pkl')

# Prepare label encoders (fit them on known values from training)
# These are hardcoded based on training set categories
fat_content_encoder = LabelEncoder()
fat_content_encoder.classes_ = np.array(['Low Fat', 'Regular'])

outlet_size_encoder = LabelEncoder()
outlet_size_encoder.classes_ = np.array(['High', 'Medium', 'Small'])

location_type_encoder = LabelEncoder()
location_type_encoder.classes_ = np.array(['Tier 1', 'Tier 2', 'Tier 3'])

outlet_type_encoder = LabelEncoder()
outlet_type_encoder.classes_ = np.array(['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])

outlet_id_encoder = LabelEncoder()
outlet_id_encoder.classes_ = np.array([
    'OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019',
    'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'
])

item_type_encoder = LabelEncoder()
item_type_encoder.classes_ = np.array([
    'Baking Goods', 'Breads', 'Breakfast', 'Canned', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables',
    'Hard Drinks', 'Health and Hygiene', 'Household', 'Meat', 'Others',
    'Seafood', 'Snack Foods', 'Soft Drinks', 'Starchy Foods'
])

@app.route('/predict', methods=['POST'])
def predict_sales():
    try:
        input_data = request.get_json()
        df = pd.DataFrame([input_data])

        # Standardize Fat Content
        df['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}, inplace=True)

        # Fill missing values
        df['Item_Weight'].fillna(12.8576451841, inplace=True)  # example training mean
        df['Outlet_Size'].fillna('Medium', inplace=True)       # example training mode

        # Feature Engineering
        df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']

        # Label Encode
        df['Item_Fat_Content'] = fat_content_encoder.transform(df['Item_Fat_Content'])
        df['Outlet_Size'] = outlet_size_encoder.transform(df['Outlet_Size'])
        df['Outlet_Location_Type'] = location_type_encoder.transform(df['Outlet_Location_Type'])
        df['Outlet_Type'] = outlet_type_encoder.transform(df['Outlet_Type'])
        df['Outlet_Identifier'] = outlet_id_encoder.transform(df['Outlet_Identifier'])
        df['Item_Type'] = item_type_encoder.transform(df['Item_Type'])

        # Drop unused columns
        df.drop(['Item_Identifier', 'Outlet_Establishment_Year'], axis=1, inplace=True)

        # Ensure column order
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]

        # Prediction
        prediction = model.predict(df)[0]
        return jsonify({'predicted_sales': round(float(prediction), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
