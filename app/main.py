import json
import pickle
import re
from pathlib import Path
from typing import List

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from langdetect import detect
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration et Initialisation ---
app = FastAPI(
    title="API de Détection de Phishing Automatique (FR/EN)",
    description="Une API pour classifier des textes en détectant automatiquement la langue.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Chargement des Artefacts du Modèle ---
try:
    print("🚀 Démarrage de l'API et chargement des artefacts...")
    current_dir = Path(__file__).parent
    model_dir = current_dir / "model"

    model = load_model(model_dir / "best_lstm_model.keras")
    with open(model_dir / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open(model_dir / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    MAX_SEQUENCE_LENGTH = metadata['config']['max_sequence_length']

    with open(model_dir / "suspicious_words.json", 'r') as f:
        suspicious_words_data = json.load(f)
    SUSPICIOUS_WORDS_SET = set(suspicious_words_data.get('en', []) + suspicious_words_data.get('fr', []))

    # Préparation des stopwords NLTK pour les langues supportées
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("📥 Téléchargement des données NLTK (stopwords)...")
        nltk.download('stopwords', quiet=True)
    STOP_WORDS = {
        'en': set(nltk.corpus.stopwords.words('english')),
        'fr': set(nltk.corpus.stopwords.words('french'))
    }
    print(f"✅ Artefacts chargés (max_len: {MAX_SEQUENCE_LENGTH}, stopwords: fr/en).")
    print("\n🎉 API prête à recevoir des requêtes !")

except Exception as e:
    print(f"❌ ERREUR CRITIQUE AU DÉMARRAGE: {e}")
    model = None


# --- Modèles de Données Pydantic ---
# Le modèle d'entrée est simplifié : la langue n'est plus requise.
class TextInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "URGENT: Your account will be suspended in 24 hours. Click here to verify.",
            }
        }


class BatchInput(BaseModel):
    items: List[TextInput]

    class Config:
        json_schema_extra = {
            "example": {
                "items": [
                    {"text": "Bonjour, votre facture no. 8373 arrive à échéance."},
                    {"text": "Hi Sarah, thanks for sending the quarterly report."}
                ]
            }
        }


# --- Fonctions de Prétraitement ---
def preprocess_text(text: str, language: str):
    """Prétraitement de texte spécialisé et multilingue."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL_TOKEN ',
                  text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL_TOKEN ', text)
    text = re.sub(r'\b\d+\b', ' NUM_TOKEN ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    stop_words_lang = STOP_WORDS.get(language, set())

    filtered_tokens = [token for token in tokens if len(token) > 2 and token not in stop_words_lang]
    return ' '.join(filtered_tokens)


def extract_numerical_features(text: str):
    """Extraction des features numériques alignée sur l'entraînement."""
    if pd.isna(text):
        text = ""
    text_str = str(text)

    features = [
        len(text_str),
        len(text_str.split()),
        text_str.count('!'),
        text_str.count('?'),
        sum(1 for c in text_str if c.isupper()) / max(len(text_str), 1),
        len(re.findall(r'http[s]?://', text_str)),
        len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_str)),
        sum(1 for word in SUSPICIOUS_WORDS_SET if word in text_str.lower()),
        sum(1 for c in text_str if c.isdigit()) / max(len(text_str), 1),
        sum(1 for c in text_str if c in '!@#$%^&*()') / max(len(text_str), 1)
    ]
    return features


# --- Logique de prédiction principale ---
def perform_prediction(text: str):
    """Fonction cœur qui détecte la langue et effectue une prédiction."""
    # Détection automatique de la langue
    try:
        # On ne garde que les 1000 premiers caractères pour une détection rapide et fiable
        detected = detect(text[:1000])
        lang = detected.language if detected and detected.is_reliable else 'en'
        # Si la langue détectée n'est pas supportée par notre modèle, on utilise 'en' par défaut
        if lang not in ['fr', 'en']:
            lang = 'en'
    except Exception:
        lang = 'en'  # En cas d'erreur, fallback sur l'anglais

    # Séquence textuelle
    processed_text = preprocess_text(text, lang)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='post',
        truncating='post'
    )

    # Features numériques
    numerical_features = extract_numerical_features(text)
    scaled_features = scaler.transform([numerical_features])

    # Prédiction du modèle
    prediction_proba = model.predict([padded_sequence, scaled_features])[0][0]
    prediction_int = int(prediction_proba > 0.5)
    predicted_class = label_encoder.inverse_transform([prediction_int])[0]

    confidence = "HIGH" if abs(prediction_proba - 0.5) > 0.4 else "MEDIUM" if abs(
        prediction_proba - 0.5) > 0.2 else "LOW"

    return {
        "prediction": predicted_class,
        "probability": float(prediction_proba),
        "confidence": confidence,
        "language_detected": lang
    }


# --- Endpoints de l'API ---
@app.get("/", summary="Message de bienvenue")
def read_root():
    return {
        "message": "Bienvenue sur l'API de détection de phishing (LSTM Hybride FR/EN)",
        "version": app.version,
        "documentation": "/docs"
    }


@app.get("/health", summary="Vérification de l'état de l'API")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", summary="Prédire sur un seul texte")
def predict(item: TextInput):
    """
    Analyse un texte, détecte sa langue (fr/en) et prédit s'il s'agit d'un phishing.
    """
    try:
        print(f"📧 Analyse d'un texte de {len(item.text)} caractères...")
        result = perform_prediction(item.text)
        print(
            f"🎯 Résultat ({result['language_detected']}): {result['prediction']} (Proba: {result['probability']:.4f})")
        return result
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur: {e}")


@app.post("/predict/batch", summary="Prédire sur une liste de textes")
def predict_batch(batch: BatchInput):
    """
    Analyse une liste de textes en parallèle.
    """
    results = []
    print(f"📦 Traitement d'un batch de {len(batch.items)} textes...")
    for item in batch.items:
        try:
            result = perform_prediction(item.text)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "text": item.text[:50] + "..."})

    return {"results": results}


# --- Lancement de l'application ---
if __name__ == "__main__":
    import uvicorn

    print("🚀 Démarrage de l'API de test sur http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)