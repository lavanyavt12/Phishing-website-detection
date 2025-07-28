import numpy as np
import pandas as pd
import os
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import save_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.cnn_model import create_cnn_model
from models.lstm_model import create_lstm_model
from models.xgboost_model import create_xgboost_model
from preprocessing.feature_extraction import URLFeatureExtractor
from utils.metrics import evaluate_model, save_results

def prepare_data():
    """
    Prepare data for model training by loading dataset, extracting features,
    splitting data, and scaling features.
    """
    try:
        # Load data
        data_path = os.path.join(project_root, 'data', 'raw_data', 'PhiUSIIL_Phishing_URL_Dataset.csv')
        
        print(f"Loading dataset from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at: {data_path}")
            
        df = pd.read_csv(data_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        print("Original columns:", df.columns.tolist())
        
        # Get the URL column name
        url_column = 'URL'
        if url_column not in df.columns:
            raise ValueError(f"URL column '{url_column}' not found in dataset")
            
        print(f"\nLabel distribution:\n{df['label'].value_counts()}\n")
        
        # Extract features
        extractor = URLFeatureExtractor()
        features_list = []
        
        print("Extracting features from URLs...")
        total_urls = len(df)
        for idx, url in enumerate(df[url_column], 1):
            if idx % 1000 == 0:
                print(f"Processing URL {idx}/{total_urls}")
            try:
                features = extractor.extract_url_features(url)
                features_list.append(features)
            except Exception as e:
                print(f"Error processing URL at index {idx}: {str(e)}")
                features_list.append({})
        
        X = pd.DataFrame(features_list)
        y = df['label']
        
        print("\nFeature extraction completed.")
        print(f"Features shape: {X.shape}")
        
        # Split data
        print("\nSplitting dataset into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save the scaler
        try:
            scaler_path = os.path.join(project_root, 'models', 'saved', 'scaler.pkl')
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
        except Exception as e:
            print(f"Warning: Failed to save scaler: {str(e)}")
        
        print("\nFinal data shapes:")
        print(f"X_train: {X_train_scaled.shape}")
        print(f"X_test: {X_test_scaled.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        raise

def train_models():
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Reshape data for CNN and LSTM
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Manual hyperparameter tuning for CNN
    print("\nPerforming manual hyperparameter tuning for CNN...")
    param_combinations = [
        {'learning_rate': lr, 'dropout_rate': dr, 'filters': f}
        for lr in [0.01, 0.001]
        for dr in [0.2, 0.3]
        for f in [16, 32]
    ]
    
    best_val_acc = 0
    best_params = None
    best_model = None
    
    # Split training data for validation
    val_split = int(0.1 * len(X_train_reshaped))
    X_train_tune = X_train_reshaped[:-val_split]
    X_val_tune = X_train_reshaped[-val_split:]
    y_train_tune = y_train[:-val_split]
    y_val_tune = y_train[-val_split:]
    
    for params in param_combinations:
        print(f"\nTrying parameters: {params}")
        model = create_cnn_model(
            input_shape=(X_train.shape[1], 1),
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate'],
            filters=params['filters']
        )
        
        history = model.fit(
            X_train_tune, y_train_tune,
            epochs=3,
            batch_size=32,
            validation_data=(X_val_tune, y_val_tune),
            verbose=1
        )
        
        val_acc = max(history.history['val_accuracy'])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            best_model = model
    
    print(f"\nBest parameters found: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Train final CNN model with best parameters
    print("\nTraining final CNN model with best parameters...")
    cnn_model = create_cnn_model(
        input_shape=(X_train.shape[1], 1),
        learning_rate=best_params['learning_rate'],
        dropout_rate=best_params['dropout_rate'],
        filters=best_params['filters']
    )
    
    cnn_history = cnn_model.fit(
        X_train_reshaped, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    # Train LSTM model
    print("\nTraining LSTM model...")
    lstm_model = create_lstm_model(input_shape=(X_train.shape[1], 1))
    lstm_history = lstm_model.fit(
        X_train_reshaped, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    xgb_model = create_xgboost_model()
    xgb_model.fit(X_train, y_train)
    
    return cnn_model, lstm_model, xgb_model, (X_test, X_test_reshaped, y_test), cnn_history, lstm_history

def save_models(cnn_model, lstm_model, xgb_model):
    """Save trained models to disk"""
    output_dir = os.path.join(project_root, 'models', 'saved')
    
    # Remove directory if it exists and recreate it
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("\nSaving models...")
    try:
        # Save CNN model
        cnn_path = os.path.join(output_dir, 'cnn_model')
        os.makedirs(cnn_path, exist_ok=True)
        save_model(cnn_model, cnn_path)
        
        # Save LSTM model
        lstm_path = os.path.join(output_dir, 'lstm_model')
        os.makedirs(lstm_path, exist_ok=True)
        save_model(lstm_model, lstm_path)
        
        # Save XGBoost model
        xgb_model.save_model(os.path.join(output_dir, 'xgboost_model.json'))
        
        print("Models saved successfully!")
    except Exception as e:
        print(f"Error saving models: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting training pipeline...")
    
    try:
        # Train models and get test data
        cnn_model, lstm_model, xgb_model, (X_test, X_test_reshaped, y_test), cnn_history, lstm_history = train_models()
        
        # Save trained models
        save_models(cnn_model, lstm_model, xgb_model)
        
        # Evaluate and save results for each model
        print("\nEvaluating models...")
        
        # CNN evaluation
        cnn_pred = (cnn_model.predict(X_test_reshaped) > 0.5).astype(int)
        evaluate_model(y_test, cnn_pred, "CNN")
        save_results("CNN", 
                    accuracy_score(y_test, cnn_pred),
                    classification_report(y_test, cnn_pred),
                    confusion_matrix(y_test, cnn_pred),
                    cnn_history)
        
        # LSTM evaluation
        lstm_pred = (lstm_model.predict(X_test_reshaped) > 0.5).astype(int)
        evaluate_model(y_test, lstm_pred, "LSTM")
        save_results("LSTM", 
                    accuracy_score(y_test, lstm_pred),
                    classification_report(y_test, lstm_pred),
                    confusion_matrix(y_test, lstm_pred),
                    lstm_history)
        
        # XGBoost evaluation
        xgb_pred = xgb_model.predict(X_test)
        evaluate_model(y_test, xgb_pred, "XGBoost")
        save_results("XGBoost", 
                    accuracy_score(y_test, xgb_pred),
                    classification_report(y_test, xgb_pred),
                    confusion_matrix(y_test, xgb_pred))
                    
        print("\nTraining pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise