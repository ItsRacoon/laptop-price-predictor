from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and metadata
print("Loading model...")
try:
    model_package = joblib.load("laptop_price_model_complete.pkl")
    model = model_package['model']
    feature_names = model_package['feature_names']
    numeric_cols = model_package['numeric_cols']
    categorical_cols = model_package['categorical_cols']
    categorical_info = model_package['categorical_info']
    numeric_info = model_package['numeric_info']
    metrics = model_package['metrics']
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model information and feature options"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'categorical_info': categorical_info,
        'numeric_info': numeric_info,
        'metrics': metrics,
        'feature_names': feature_names
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction based on input features"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        # Create a dictionary for all features
        feature_dict = {}
        
        # Process numeric features (excluding engineered ones)
        base_numeric_cols = ['num_cores', 'num_threads', 'ram_memory', 
                            'primary_storage_capacity', 'secondary_storage_capacity',
                            'display_size', 'resolution_width', 'resolution_height',
                            'year_of_warranty']
        
        for col in base_numeric_cols:
            feature_dict[col] = float(data.get(col, 0))
        
        # Calculate engineered features
        feature_dict['total_storage'] = (
            feature_dict['primary_storage_capacity'] + 
            feature_dict['secondary_storage_capacity']
        )
        feature_dict['resolution_total'] = (
            feature_dict['resolution_width'] * 
            feature_dict['resolution_height']
        )
        feature_dict['cores_per_thread_ratio'] = (
            feature_dict['num_cores'] / 
            (feature_dict['num_threads'] + 1)
        )
        feature_dict['ram_per_core'] = (
            feature_dict['ram_memory'] / 
            (feature_dict['num_cores'] + 1)
        )
        feature_dict['has_secondary_storage'] = int(
            feature_dict['secondary_storage_capacity'] > 0
        )
        
        # Process categorical features
        for col in categorical_cols:
            feature_dict[col] = data.get(col, 'Unknown')
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([feature_dict])[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Calculate confidence interval (rough approximation using RMSE)
        rmse = metrics['test_rmse']
        lower_bound = max(0, prediction - rmse)
        upper_bound = prediction + rmse
        
        return jsonify({
            'prediction': float(prediction),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'formatted_prediction': f"₹{prediction:,.2f}",
            'formatted_range': f"₹{lower_bound:,.2f} - ₹{upper_bound:,.2f}"
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Make predictions for multiple laptops"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        laptops = data.get('laptops', [])
        
        predictions = []
        for laptop_data in laptops:
            # Process each laptop similar to single prediction
            feature_dict = {}
            
            base_numeric_cols = ['num_cores', 'num_threads', 'ram_memory', 
                                'primary_storage_capacity', 'secondary_storage_capacity',
                                'display_size', 'resolution_width', 'resolution_height',
                                'year_of_warranty']
            
            for col in base_numeric_cols:
                feature_dict[col] = float(laptop_data.get(col, 0))
            
            # Calculate engineered features
            feature_dict['total_storage'] = (
                feature_dict['primary_storage_capacity'] + 
                feature_dict['secondary_storage_capacity']
            )
            feature_dict['resolution_total'] = (
                feature_dict['resolution_width'] * 
                feature_dict['resolution_height']
            )
            feature_dict['cores_per_thread_ratio'] = (
                feature_dict['num_cores'] / 
                (feature_dict['num_threads'] + 1)
            )
            feature_dict['ram_per_core'] = (
                feature_dict['ram_memory'] / 
                (feature_dict['num_cores'] + 1)
            )
            feature_dict['has_secondary_storage'] = int(
                feature_dict['secondary_storage_capacity'] > 0
            )
            
            for col in categorical_cols:
                feature_dict[col] = laptop_data.get(col, 'Unknown')
            
            input_df = pd.DataFrame([feature_dict])[feature_names]
            prediction = model.predict(input_df)[0]
            
            predictions.append({
                'specs': laptop_data,
                'prediction': float(prediction),
                'formatted_prediction': f"₹{prediction:,.2f}"
            })
        
        return jsonify({'predictions': predictions})
        
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)