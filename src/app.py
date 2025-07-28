from flask import Flask, render_template, request, jsonify
import os
from predict import PhishingDetector

app = Flask(__name__)

# Initialize detector
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved')
detector = PhishingDetector(models_dir=model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'})
    
    try:
        result = detector.predict(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)