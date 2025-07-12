# api_cors_fix.py - Ajoutez ces headers CORS à votre API

from flask import Flask, request, jsonify
from flask_cors import CORS  # pip install flask-cors

app = Flask(__name__)

# Configuration CORS pour l'extension Chrome
CORS(app, resources={
    r"/predict": {
        "origins": ["chrome-extension://*", "https://mail.google.com", "https://*.mail.yahoo.com"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Gestion explicite des requêtes OPTIONS (preflight)
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'OK'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response

    # Votre logique d'analyse existante
    try:
        data = request.get_json()
        text = data.get('text', '')

        # REMPLACEZ PAR VOTRE LOGIQUE D'ANALYSE
        # Exemple simple pour tester
        is_phishing = 'phishing' in text.lower() or 'urgent' in text.lower()

        result = {
            'prediction': 'phishing' if is_phishing else 'safe',
            'confidence': 0.95 if is_phishing else 0.85
        }

        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        error_response = jsonify({'error': str(e), 'prediction': 'safe', 'confidence': 0.0})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500


if __name__ == '__main__':
    print("🚀 API démarrée avec support CORS pour Extension Chrome")
    app.run(host='localhost', port=8000, debug=True)