import json
import pickle
import re
from pathlib import Path
from typing import List
import csv
import os
from datetime import datetime
from typing import Optional
import json
import pickle
import re
import nltk
import tensorflow as tf
from langdetect import detect
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import threading
from fastapi import BackgroundTasks
from pathlib import Path
import pandas as pd

tf.config.set_visible_devices([], 'GPU')

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

# Variables globales pour les artefacts
model = None
tokenizer = None
scaler = None
label_encoder = None
MAX_SEQUENCE_LENGTH = 566
SUSPICIOUS_WORDS_SET = set()
STOP_WORDS = {}
NEGATIVE_FEEDBACK_THRESHOLD = 5
FEEDBACK_THRESHOLD = 5
FINETUNE_SAMPLE_SIZE = 200

IS_RETRAINING = False
RETRAIN_LOCK = threading.Lock()

FEEDBACK_CSV_PATH = Path("./data/user_feedbacks.csv")


# --- Chargement des Artefacts du Modèle (MÉTHODE QUI MARCHE) ---
def load_model_artifacts():
    """Charge tous les artefacts du modèle avec la méthode qui fonctionne"""
    global model, tokenizer, scaler, label_encoder, MAX_SEQUENCE_LENGTH, SUSPICIOUS_WORDS_SET, STOP_WORDS

    try:
        print("🚀 Démarrage de l'API et chargement des artefacts...")

        # ✅ UTILISATION DE LA MÉTHODE QUI MARCHE
        print("📦 Chargement des artefacts...")

        # Chargement du modèle
        model = load_model('model/best_lstm_model.keras')
        print("✅ Modèle LSTM chargé")

        # Chargement du tokenizer
        with open('model/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"✅ Tokenizer chargé (vocab: {len(tokenizer.word_index)} mots)")

        # Chargement du scaler
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler chargé")

        # Chargement du label encoder
        with open('model/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"✅ Label encoder chargé (classes: {label_encoder.classes_})")

        # Vérifier les dimensions du modèle
        print(f"🔍 Vérification des dimensions:")
        for i, input_layer in enumerate(model.inputs):
            print(f"  Input {i}: {input_layer.name} - Shape: {input_layer.shape}")

        # Charger les métadonnées (optionnel)
        metadata_file = Path("model/model_metadata.json")
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                config = metadata.get('config', {})
                MAX_SEQUENCE_LENGTH = config.get('max_sequence_length', 566)
                print(f"✅ Métadonnées chargées: max_sequence_length = {MAX_SEQUENCE_LENGTH}")
            except Exception as e:
                print(f"⚠️ Erreur chargement métadonnées: {e}")
                MAX_SEQUENCE_LENGTH = 566

        # Charger les mots suspects
        suspicious_words_file = Path("model/suspicious_words.json")
        if suspicious_words_file.exists():
            try:
                with open(suspicious_words_file, 'r') as f:
                    suspicious_words_data = json.load(f)
                SUSPICIOUS_WORDS_SET = set(suspicious_words_data.get('en', []) + suspicious_words_data.get('fr', []))
                print(f"✅ Mots suspects chargés ({len(SUSPICIOUS_WORDS_SET)} mots)")
            except Exception as e:
                print(f"⚠️ Erreur chargement mots suspects: {e}")
        else:
            print("⚠️ Fichier suspicious_words.json manquant, utilisation d'une liste vide")

        # Charger les stopwords NLTK avec gestion d'erreur
        try:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("📥 Téléchargement des données NLTK...")
                nltk.download('stopwords', quiet=True)

            STOP_WORDS = {
                'en': set(nltk.corpus.stopwords.words('english')),
                'fr': set(nltk.corpus.stopwords.words('french'))
            }
            print("✅ Stopwords chargés")
        except Exception as e:
            print(f"⚠️ Erreur chargement stopwords: {e}")
            # Fallback avec stopwords de base
            STOP_WORDS = {
                'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
                'fr': {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'dans', 'sur', 'avec', 'pour', 'de'}
            }
            print("✅ Stopwords de base chargés")

        print(f"\n🎉 API prête ! Longueur de séquence configurée: {MAX_SEQUENCE_LENGTH}")
        return True

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE AU DÉMARRAGE: {e}")
        print(f"❌ Type d'erreur: {type(e).__name__}")

        # Diagnostic détaillé
        print(f"\n🔍 DIAGNOSTIC:")
        current_dir = Path(".")
        print(f"📁 Répertoire courant: {current_dir.absolute()}")

        # Vérifier si le dossier model existe
        model_dir = Path("model")
        if model_dir.exists():
            print(f"📁 Contenu du dossier model:")
            for file in model_dir.iterdir():
                print(f"  - {file.name}")
        else:
            print("❌ Dossier 'model' n'existe pas")

        # Lister les fichiers dans le répertoire courant
        print(f"📄 Fichiers dans le répertoire courant:")
        for item in current_dir.iterdir():
            if item.is_file():
                print(f"  - {item.name}")

        return False


# Charger les artefacts au démarrage
model_loaded = load_model_artifacts()


# --- Modèles de Données Pydantic ---
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


class FeedbackInput(BaseModel):
    email_text: str
    predicted_class: str
    predicted_probability: float
    user_satisfaction: str  # "yes" ou "no"
    language_detected: str

    class Config:
        json_schema_extra = {
            "example": {
                "email_text": "URGENT: Click here to verify your account",
                "predicted_class": "phishing",
                "predicted_probability": 0.85,
                "user_satisfaction": "yes",
                "language_detected": "en"
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
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        # Détection automatique de la langue
        try:
            detected_lang = detect(text[:1000])
            lang = detected_lang if detected_lang in ['fr', 'en'] else 'en'
        except Exception:
            lang = 'en'  # Fallback sur l'anglais

        print(f"🌍 Langue détectée: {lang}")

        # Prétraitement du texte
        processed_text = preprocess_text(text, lang)
        print(f"📝 Texte prétraité: {processed_text[:100]}...")

        # Création des séquences
        sequence = tokenizer.texts_to_sequences([processed_text])

        # Filtrer les indices qui dépassent la taille du vocabulaire du modèle
        MAX_VOCAB_INDEX = 10000  # Limite du modèle (ajustez selon vos métadonnées)
        if sequence[0]:  # Si la séquence n'est pas vide
            # Remplacer les indices >= MAX_VOCAB_INDEX par 1 (token OOV)
            sequence[0] = [idx if idx < MAX_VOCAB_INDEX else 1 for idx in sequence[0]]

        print(f"🔢 Séquence filtrée: longueur = {len(sequence[0]) if sequence[0] else 0}")

        # Padding de la séquence
        padded_sequence = pad_sequences(
            sequence,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding='post',
            truncating='post'
        )
        print(f"📏 Séquence paddée: shape = {padded_sequence.shape}")

        # Features numériques
        numerical_features = extract_numerical_features(text)
        scaled_features = scaler.transform([numerical_features])
        print(f"🔢 Features numériques: shape = {scaled_features.shape}")

        # Vérification finale des dimensions
        expected_text_shape = (1, MAX_SEQUENCE_LENGTH)
        expected_num_shape = (1, 10)  # Nombre de features numériques

        if padded_sequence.shape != expected_text_shape:
            raise ValueError(f"Dimension texte incorrecte: {padded_sequence.shape} != {expected_text_shape}")
        if scaled_features.shape != expected_num_shape:
            raise ValueError(f"Dimension features incorrecte: {scaled_features.shape} != {expected_num_shape}")

        print(f"✅ Dimensions validées, prédiction en cours...")

        # Prédiction du modèle
        prediction_proba = model.predict([padded_sequence, scaled_features], verbose=0)[0][0]
        prediction_int = int(prediction_proba > 0.5)
        predicted_class = label_encoder.inverse_transform([prediction_int])[0]

        # Calcul de la confiance
        confidence_score = abs(prediction_proba - 0.5) * 2
        if confidence_score > 0.8:
            confidence = "HIGH"
        elif confidence_score > 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        print(f"✅ Prédiction réussie: {predicted_class} (prob: {prediction_proba:.4f}, conf: {confidence})")

        return {
            "prediction": predicted_class,
            "probability": float(prediction_proba),
            "confidence": confidence,
            "language_detected": lang
        }

    except Exception as e:
        print(f"❌ Erreur dans perform_prediction: {e}")
        print(f"❌ Type d'erreur: {type(e).__name__}")
        import traceback
        print(f"❌ Traceback complet:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


# --- Fonctions de Gestion des Feedbacks ---
def count_negative_feedbacks() -> int:
    """Compte uniquement les feedbacks négatifs non traités"""
    try:
        if not FEEDBACK_CSV_PATH.exists():
            return 0

        df = pd.read_csv(FEEDBACK_CSV_PATH)

        # Compter seulement les feedbacks négatifs ET non traités
        negative_unprocessed = len(df[
                                       (df['user_satisfaction'] == 'no') &
                                       (df['processed'] == False)
                                       ])

        print(f"📊 Feedbacks négatifs non traités: {negative_unprocessed}")
        return negative_unprocessed

    except Exception as e:
        print(f"❌ Erreur lors du comptage des feedbacks négatifs: {e}")
        return 0


def save_feedback_to_csv(feedback_data):
    """Sauvegarde le feedback dans un fichier CSV"""
    try:
        csv_headers = [
            'timestamp', 'email_text', 'predicted_class',
            'predicted_probability', 'user_satisfaction', 'language_detected', 'processed'
        ]

        # Ajouter la colonne processed (pour marquer les feedbacks traités)
        feedback_data['processed'] = False

        # Créer le dossier data s'il n'existe pas
        FEEDBACK_CSV_PATH.parent.mkdir(exist_ok=True)

        # Vérifier si le fichier existe
        file_exists = FEEDBACK_CSV_PATH.exists()

        with open(FEEDBACK_CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)

            # Écrire les en-têtes si le fichier est nouveau
            if not file_exists:
                writer.writeheader()

            writer.writerow(feedback_data)

        print(f"✅ Feedback sauvegardé dans {FEEDBACK_CSV_PATH}")
        return True

    except Exception as e:
        print(f"❌ Erreur sauvegarde feedback: {e}")
        return False


# --- Endpoints de l'API ---
@app.get("/", summary="Message de bienvenue")
def read_root():
    return {
        "message": "Bienvenue sur l'API de détection de phishing (LSTM Hybride FR/EN)",
        "version": app.version,
        "documentation": "/docs",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "model_loaded": model_loaded
    }


@app.get("/health", summary="Vérification de l'état de l'API")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded")

    negative_feedbacks = count_negative_feedbacks()

    return {
        "status": "healthy",
        "model_loaded": True,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "vocab_size": len(tokenizer.word_index) if tokenizer else 0,
        "is_retraining": IS_RETRAINING,
        "negative_feedbacks": negative_feedbacks,
        "negative_threshold": NEGATIVE_FEEDBACK_THRESHOLD,
        "finetune_sample_size": FINETUNE_SAMPLE_SIZE
    }


@app.post("/predict", summary="Prédire sur un seul texte")
def predict(item: TextInput):
    """
    Analyse un texte, détecte sa langue (fr/en) et prédit s'il s'agit d'un phishing.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        print(f"📧 Analyse d'un texte de {len(item.text)} caractères...")
        result = perform_prediction(item.text)
        print(
            f"🎯 Résultat ({result['language_detected']}): {result['prediction']} (Proba: {result['probability']:.4f})")
        return result
    except HTTPException:
        raise
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


@app.post("/feedback", summary="Enregistrer un feedback utilisateur")
async def save_feedback(feedback: FeedbackInput, background_tasks: BackgroundTasks):
    """
    Enregistre le feedback utilisateur
    """
    try:
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "email_text": feedback.email_text[:500],
            "predicted_class": feedback.predicted_class,
            "predicted_probability": feedback.predicted_probability,
            "user_satisfaction": feedback.user_satisfaction,
            "language_detected": feedback.language_detected
        }

        if not save_feedback_to_csv(feedback_data):
            raise HTTPException(status_code=500, detail="Erreur sauvegarde feedback")

        negative_count = count_negative_feedbacks()

        print(f"📝 Feedback enregistré: {feedback.user_satisfaction}")
        print(f"📊 Feedbacks négatifs: {negative_count}/{NEGATIVE_FEEDBACK_THRESHOLD}")

        response = {
            "status": "success",
            "message": "Feedback enregistré avec succès",
            "feedback_type": feedback.user_satisfaction,
            "negative_feedbacks": negative_count,
            "negative_threshold": NEGATIVE_FEEDBACK_THRESHOLD
        }

        return response

    except Exception as e:
        print(f"❌ Erreur feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {e}")


@app.get("/feedbacks", summary="Voir les feedbacks utilisateur")
def get_feedbacks():
    """
    Récupère tous les feedbacks enregistrés
    """
    try:
        if not FEEDBACK_CSV_PATH.exists():
            return {
                "message": "Aucun feedback enregistré pour le moment",
                "feedbacks": [],
                "count": 0
            }

        feedbacks = []
        with open(FEEDBACK_CSV_PATH, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                feedbacks.append(row)

        return {
            "message": "Feedbacks récupérés avec succès",
            "count": len(feedbacks),
            "feedbacks": feedbacks,
            "file_location": str(FEEDBACK_CSV_PATH)
        }

    except Exception as e:
        print(f"❌ Erreur lecture feedbacks: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {e}")


@app.get("/debug/model-info", summary="Informations de diagnostic du modèle")
def get_model_info():
    """Endpoint pour diagnostiquer les problèmes de modèle"""
    if not model_loaded:
        return {"error": "Modèle non chargé", "model_loaded": False}

    try:
        info = {
            "model_loaded": model_loaded,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "tokenizer_vocab_size": len(tokenizer.word_index) if tokenizer else 0,
            "model_inputs": [],
            "suspicious_words_count": len(SUSPICIOUS_WORDS_SET),
            "stopwords_languages": list(STOP_WORDS.keys()),
            "label_classes": label_encoder.classes_.tolist() if label_encoder else []
        }

        if model:
            for i, input_layer in enumerate(model.inputs):
                info["model_inputs"].append({
                    "index": i,
                    "name": input_layer.name,
                    "shape": input_layer.shape.as_list()
                })

        return info
    except Exception as e:
        return {"error": str(e), "model_loaded": model_loaded}


# --- Lancement de l'application ---
if __name__ == "__main__":
    import uvicorn

    print("🚀 Démarrage de l'API sur http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)