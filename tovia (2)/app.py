from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

app = Flask(__name__)

# Global variables to store models and preprocessor
models = {}
preprocessor = None

def load_models_and_preprocessor():
    """Load the trained models and preprocessor"""
    global models, preprocessor
    
    try:
        # Set TensorFlow to use CPU only to avoid GPU compatibility issues
        tf.config.set_visible_devices([], 'GPU')
        
        # Load models with custom_objects to handle metric compatibility
        custom_objects = {
            'mse': tf.keras.metrics.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError(),
            'mape': tf.keras.metrics.MeanAbsolutePercentageError()
        }
        
        # Try to load models with different approaches
        model_files = ['models/ann_model.h5', 'models/dnn_model.h5', 'models/cnn_model.h5']
        model_names = ['ann', 'dnn', 'cnn']
        
        for model_file, model_name in zip(model_files, model_names):
            try:
                # First try with custom_objects and compile=False
                models[model_name] = tf.keras.models.load_model(
                    model_file, 
                    custom_objects=custom_objects, 
                    compile=False
                )
                print(f"Successfully loaded {model_name} model")
            except Exception as e:
                print(f"Error loading {model_name} model: {e}")
                # Try alternative loading method
                try:
                    models[model_name] = tf.keras.models.load_model(
                        model_file, 
                        compile=False,
                        options=tf.saved_model.LoadOptions(experimental_io_device='/cpu:0')
                    )
                    print(f"Successfully loaded {model_name} model with alternative method")
                except Exception as e2:
                    print(f"Failed to load {model_name} model with alternative method: {e2}")
                    return False
        
        # Recompile models with current TensorFlow version
        for model_name, model in models.items():
            try:
                model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae', 'mape']
                )
                print(f"Successfully compiled {model_name} model")
            except Exception as e:
                print(f"Error compiling {model_name} model: {e}")
        
        # Load preprocessor
        try:
            with open('models/preprocessor.pkl', 'rb') as f:
                global preprocessor
                preprocessor = pickle.load(f)
            print("Preprocessor loaded successfully!")
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
            return False
            
        print("Models and preprocessor loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def make_prediction(model_type, frequency, distance, tx_height, rx_height, environment):
    """Make prediction using the specified model"""
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Frequency_MHz': [frequency],
            'Distance_km': [distance],
            'Tx_Height_m': [tx_height],
            'Rx_Height_m': [rx_height],
            'Environment': [environment]
        })
        print(f"Input data created: {input_data}")
        
        if preprocessor is None:
            print("Error: Preprocessor not loaded")
            return None
            
        # Preprocess the input
        try:
            input_processed = preprocessor.transform(input_data)
            print(f"Input preprocessed successfully. Shape: {input_processed.shape}")
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None
        
        # Get the model
        if model_type not in models:
            print(f"Error: Model {model_type} not found in loaded models")
            return None
            
        model = models[model_type]
        
        # For CNN model, we need to reshape the input
        if model_type == 'cnn':
            input_processed = np.expand_dims(input_processed, axis=2)
            print(f"CNN input shape after reshape: {input_processed.shape}")
        
        # Make prediction
        try:
            prediction = model.predict(input_processed)[0][0]
            print(f"Prediction successful: {prediction}")
            return float(prediction)
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return None
    
    except Exception as e:
        print(f"Error in make_prediction: {e}")
        return None

@app.route('/')
def index():
    """Main page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        frequency = float(request.form['frequency'])
        distance = float(request.form['distance'])
        tx_height = float(request.form['tx_height'])
        rx_height = float(request.form['rx_height'])
        environment = request.form['environment']
        model_type = request.form['model_type']
        
        # Validate inputs
        if frequency <= 0 or distance <= 0 or tx_height <= 0 or rx_height <= 0:
            return render_template('result.html', 
                                 error="All numeric values must be positive")
        
        # Make prediction
        prediction = make_prediction(model_type, frequency, distance, 
                                   tx_height, rx_height, environment)
        
        if prediction is None:
            # Check what might be wrong
            error_msg = "Error making prediction. "
            if not models:
                error_msg += "Models not loaded. "
            if preprocessor is None:
                error_msg += "Preprocessor not loaded. "
            if model_type not in models:
                error_msg += f"Model '{model_type}' not available. "
            error_msg += "Please check the logs or try again."
            
            return render_template('result.html', error=error_msg)
        
        # Prepare result data
        result_data = {
            'prediction': round(prediction, 2),
            'model_type': model_type.upper(),
            'inputs': {
                'frequency': frequency,
                'distance': distance,
                'tx_height': tx_height,
                'rx_height': rx_height,
                'environment': environment
            }
        }
        
        return render_template('result.html', result=result_data)
        
    except ValueError:
        return render_template('result.html', 
                             error="Please enter valid numeric values")
    except Exception as e:
        return render_template('result.html', 
                             error=f"An error occurred: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON response)"""
    try:
        data = request.get_json()
        
        prediction = make_prediction(
            data['model_type'],
            data['frequency'],
            data['distance'],
            data['tx_height'],
            data['rx_height'],
            data['environment']
        )
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'prediction': round(prediction, 2),
            'model_type': data['model_type'],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint to verify models are loaded"""
    try:
        status = {
            'models_loaded': len(models) > 0,
            'preprocessor_loaded': preprocessor is not None,
            'available_models': list(models.keys()),
            'status': 'healthy' if len(models) > 0 and preprocessor is not None else 'unhealthy'
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/test')
def test_prediction():
    """Test endpoint to verify prediction functionality"""
    try:
        # Test with sample data
        test_data = {
            'frequency': 900.0,
            'distance': 5.0,
            'tx_height': 30.0,
            'rx_height': 1.5,
            'environment': 'Urban',
            'model_type': 'ann'
        }
        
        prediction = make_prediction(
            test_data['model_type'],
            test_data['frequency'],
            test_data['distance'],
            test_data['tx_height'],
            test_data['rx_height'],
            test_data['environment']
        )
        
        if prediction is not None:
            return jsonify({
                'status': 'success',
                'test_prediction': prediction,
                'test_data': test_data,
                'message': 'Prediction system is working correctly!'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed - check logs for details'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Test failed: {str(e)}'
        }), 500

# Load models and preprocessor when the module is imported (for Hugging Face Spaces)
print("Starting Pathloss Prediction App...")
print("Loading models and preprocessor...")

if load_models_and_preprocessor():
    print("✅ All models loaded successfully!")
    print("🚀 Flask application ready!")
else:
    print("❌ Failed to load models. Please ensure model files are present.")
    print("Required files:")
    print("  - models/ann_model.h5")
    print("  - models/dnn_model.h5") 
    print("  - models/cnn_model.h5")
    print("  - models/preprocessor.pkl")

if __name__ == '__main__':
    # Only run the Flask app directly if this script is executed
    app.run(host='0.0.0.0', port=7860, debug=True)
