import numpy as np
from tensorflow.keras.models import load_model
import joblib
from preprocessing.feature_extraction import URLFeatureExtractor
from sklearn.preprocessing import StandardScaler

class PhishingDetector:
    def __init__(self, models_dir='models/saved'):
        # Load models
        self.cnn_model = load_model(f'{models_dir}/cnn_model')
        self.lstm_model = load_model(f'{models_dir}/lstm_model')
        self.xgb_model = joblib.load(f'{models_dir}/xgboost_model.pkl')
        self.extractor = URLFeatureExtractor()
        self.scaler = StandardScaler()
    
    def predict(self, url):
        try:
            # Extract features
            features = self.extractor.extract_url_features(url)
            X = np.array(list(features.values())).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            
            # Get predictions
            cnn_pred = (self.cnn_model.predict(X_reshaped) > 0.5).astype(int)[0][0]
            lstm_pred = (self.lstm_model.predict(X_reshaped) > 0.5).astype(int)[0][0]
            xgb_pred = self.xgb_model.predict(X_scaled)[0]
            
            # Ensemble prediction (majority voting)
            predictions = [cnn_pred, lstm_pred, xgb_pred]
            final_prediction = 1 if sum(predictions) >= 2 else 0
            
            return {
                'prediction': 'Phishing' if final_prediction == 1 else 'Legitimate',
                'cnn_pred': bool(cnn_pred),
                'lstm_pred': bool(lstm_pred),
                'xgb_pred': bool(xgb_pred)
            }
        except Exception as e:
            raise Exception(f"Error processing URL: {str(e)}")