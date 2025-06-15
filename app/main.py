# ✅ Imports corrects en tête
from fastapi import FastAPI, HTTPException  # ✅ HTTPException depuis fastapi
from pydantic import BaseModel
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sqlite3
import threading
import time

# ✅ Import système feedback corrigé
from feedbacks_collector import FeedbackCollector, start_feedback_monitoring

# ✅ Créer l'app FastAPI AVANT de l'utiliser
app = FastAPI(title="Email Spam Detection API with Feedback")
print("tensorflow version:", tf.__version__)

# ✅ Initialiser le système de feedback
feedback_collector = FeedbackCollector()

# ✅ Démarrer le monitoring automatique
start_feedback_monitoring()

# ✅ CORS middleware APRÈS création de l'app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Chargement des modèles et preprocesseurs
model = load_model('model/best_lstm_model.keras')
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


# Modèles Pydantic
class TextInput(BaseModel):
    text: str


class FeedbackInput(BaseModel):
    email_text: str
    predicted_label: str
    user_satisfaction: str  # "yes" ou "no"
    confidence_score: float = None
    email_id: str = None  # Optionnel pour traçabilité


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: int = None


def preprocess_text(text):
    """Preprocessing spécialisé pour LSTM (même logique que durant l'entraînement)"""
    if pd.isna(text) or text is None:
        return ""

    text = str(text).lower()

    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                  ' URL_TOKEN ', text)

    # Remplacer les emails par un token spécial
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  ' EMAIL_TOKEN ', text)

    # Remplacer les numéros par un token
    text = re.sub(r'\b\d+\b', ' NUM_TOKEN ', text)

    # Nettoyer la ponctuation excessive mais garder la structure
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenisation simple pour LSTM (pas de stemming agressif)
    tokens = text.split()

    filtered_tokens = []
    for token in tokens:
        if len(token) > 1 and token not in ['the', 'a', 'an', 'and', 'or', 'but']:
            filtered_tokens.append(token)

    return ' '.join(filtered_tokens)


def extract_numerical_features(text):
    """Extraire des features numériques pour un seul texte"""
    if pd.isna(text) or text is None:
        text = ""

    text_str = str(text)

    char_count = len(text_str)
    word_count = len(text_str.split())

    exclamation_count = text_str.count('!')
    question_count = text_str.count('?')
    upper_count = sum(1 for c in text_str if c.isupper())
    upper_ratio = upper_count / max(char_count, 1)

    url_count = len(re.findall(r'http[s]?://', text_str))
    email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_str))

    suspicious_words = [
        'urgent', 'immediate', 'verify', 'confirm', 'suspended', 'expired',
        'winner', 'congratulations', 'free', 'click', 'now', 'limited'
    ]
    suspicious_count = sum(1 for word in suspicious_words if word in text_str.lower())

    digit_ratio = sum(1 for c in text_str if c.isdigit()) / max(char_count, 1)
    special_char_ratio = sum(1 for c in text_str if c in '!@#$%^&*()') / max(char_count, 1)

    features = [
        char_count, word_count, exclamation_count, question_count,
        upper_ratio, url_count, email_count, suspicious_count,
        digit_ratio, special_char_ratio
    ]

    return features


# Endpoints principaux
@app.get("/")
def read_root():
    return {
        "message": "API de détection de phishing avec LSTM et système de feedback",
        "status": "active",
        "tensorflow_version": tf.__version__,
        "feedback_enabled": True
    }


@app.get("/health")
def health_check():
    # ✅ Utiliser l'instance feedback_collector
    stats = feedback_collector.get_feedback_stats()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__,
        "message": "ML API is running",
        "feedback_stats": {
            "total_feedbacks": stats['total_feedbacks'],
            "accuracy_rate": round(stats['accuracy_rate'] * 100, 2)
        }
    }


@app.post("/predict")
def predict(text_input: TextInput):
    try:
        print(f"📧 Analyse d'un email de {len(text_input.text)} caractères")

        processed_text = preprocess_text(text_input.text)
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=150, padding='post', truncating='post')

        numerical_features = extract_numerical_features(text_input.text)
        scaled_features = scaler.transform([numerical_features])

        prediction = model.predict([padded_sequence, scaled_features])
        predicted_class = label_encoder.inverse_transform([int(prediction > 0.5)])

        confidence_level = "HIGH" if abs(prediction[0][0] - 0.5) > 0.3 else "MEDIUM" if abs(
            prediction[0][0] - 0.5) > 0.1 else "LOW"

        result = {
            "prediction": predicted_class[0],
            "probability": float(prediction[0][0]),
            "confidence": confidence_level,
            "timestamp": datetime.now().isoformat(),
            "model_version": "lstm_v1"
        }

        print(f"🎯 Prédiction: {predicted_class[0]} (confiance: {prediction[0][0]:.3f})")

        return result

    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        return {
            "error": str(e),
            "prediction": "error",
            "probability": 0.0
        }


# 🆕 ENDPOINTS FEEDBACK

@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackInput):
    """Soumettre un feedback utilisateur sur une prédiction"""
    try:
        print(f"📝 Nouveau feedback: {feedback.user_satisfaction} pour {feedback.predicted_label}")

        # ✅ Utiliser l'instance feedback_collector
        feedback_collector.save_feedback(
            email_text=feedback.email_text,
            predicted_label=feedback.predicted_label,
            user_satisfaction=feedback.user_satisfaction,
            confidence_score=feedback.confidence_score
        )

        stats = feedback_collector.get_feedback_stats()

        return FeedbackResponse(
            status="success",
            message=f"Feedback enregistré. Total: {stats['total_feedbacks']} feedbacks",
            feedback_id=stats['total_feedbacks']
        )

    except Exception as e:
        print(f"❌ Erreur feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement: {e}")


@app.get("/feedback/stats")
def get_feedback_stats():
    """Obtenir les statistiques des feedbacks"""
    try:
        stats = feedback_collector.get_feedback_stats()

        # Historique des réentraînements
        conn = sqlite3.connect(feedback_collector.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT timestamp, num_feedbacks, old_accuracy, new_accuracy, status 
            FROM retraining_logs 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''')

        retraining_history = [
            {
                "timestamp": row[0],
                "num_feedbacks": row[1],
                "old_accuracy": row[2],
                "new_accuracy": row[3],
                "status": row[4]
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            "feedback_statistics": {
                "total_feedbacks": stats['total_feedbacks'],
                "negative_feedbacks": stats['negative_feedbacks'],
                "unprocessed_feedbacks": stats['unprocessed_feedbacks'],
                "accuracy_rate": round(stats['accuracy_rate'] * 100, 2),
                "feedback_breakdown": stats['feedback_breakdown']
            },
            "retraining_history": retraining_history,
            "next_retraining_at": f"{50 - stats['unprocessed_feedbacks']} feedbacks restants",
            "last_update": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur stats: {e}")


@app.post("/feedback/trigger-retraining")
def trigger_manual_retraining():
    """Déclencher manuellement un réentraînement (pour les admins)"""
    try:
        feedback_collector.trigger_retraining()

        return {
            "status": "success",
            "message": "Réentraînement déclenché en arrière-plan",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur déclenchement: {e}")


@app.get("/feedback/export")
def export_feedback_data():
    """Exporter les données de feedback pour analyse (format JSON)"""
    try:
        # ✅ Utiliser l'instance feedback_collector
        conn = sqlite3.connect(feedback_collector.db_path)

        df = pd.read_sql_query('''
            SELECT 
                email_text, 
                predicted_label, 
                user_satisfaction, 
                correct_label,
                confidence_score,
                timestamp
            FROM feedbacks 
            ORDER BY timestamp DESC
        ''', conn)

        conn.close()

        return {
            "total_records": len(df),
            "export_timestamp": datetime.now().isoformat(),
            "data": df.to_dict('records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur export: {e}")


@app.post("/feedback/test")
def test_feedback_system():
    """Endpoint de test pour vérifier le système de feedback"""
    try:
        test_data = [
            {
                "email_text": "URGENT: Verify your account now or it will be suspended!",
                "predicted_label": "phishing",
                "user_satisfaction": "yes",
                "confidence_score": 0.85
            },
            {
                "email_text": "Meeting scheduled for tomorrow at 2 PM in conference room B",
                "predicted_label": "phishing",
                "user_satisfaction": "no",  # Faux positif
                "confidence_score": 0.65
            }
        ]

        results = []
        for data in test_data:
            # ✅ Utiliser l'instance feedback_collector
            feedback_collector.save_feedback(**data)
            results.append(f"Feedback créé: {data['user_satisfaction']} pour {data['predicted_label']}")

        stats = feedback_collector.get_feedback_stats()

        return {
            "status": "success",
            "message": "Données de test créées",
            "test_results": results,
            "updated_stats": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur test: {e}")


@app.post("/predict/batch")
def predict_batch(texts: list[str]):
    """Prédire sur plusieurs textes à la fois"""
    results = []
    for text in texts:
        try:
            result = predict(TextInput(text=text))
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "text": text[:50] + "..."})

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    print("🚀 Démarrage de l'API ML avec système de feedback sur le port 8000...")
    print("📊 Endpoints disponibles:")
    print("   POST /predict - Prédiction ML")
    print("   POST /feedback - Soumettre feedback")
    print("   GET  /feedback/stats - Statistiques feedback")
    print("   POST /feedback/trigger-retraining - Déclencher réentraînement")
    print("   GET  /feedback/export - Exporter données feedback")
    print("   POST /feedback/test - Tester le système")

    # ✅ Port corrigé à 800
    uvicorn.run(app, host="0.0.0.0", port=80)