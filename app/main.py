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
import numpy as np
import pandas as pd
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
#Variables globales pour les artefacts
model = None
tokenizer = None
scaler = None
label_encoder = None
MAX_SEQUENCE_LENGTH = 566
SUSPICIOUS_WORDS_SET = set()
STOP_WORDS = {}
NEGATIVE_FEEDBACK_THRESHOLD = 5
FEEDBACK_THRESHOLD = 5

IS_RETRAINING = False
RETRAIN_LOCK = threading.Lock()

FEEDBACK_CSV_PATH = Path("./data/user_feedbacks.csv")


# --- Chargement des Artefacts du Modèle ---
def load_model_artifacts():
    """Charge tous les artefacts du modèle avec gestion d'erreurs"""
    global model, tokenizer, scaler, label_encoder, MAX_SEQUENCE_LENGTH, SUSPICIOUS_WORDS_SET, STOP_WORDS

    try:
        print("🚀 Démarrage de l'API et chargement des artefacts...")
        current_dir = Path(__file__).parent
        model_dir = current_dir / "model"

        # 1. CHARGER D'ABORD LES MÉTADONNÉES
        metadata_file = model_dir / "model_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError("Fichier model_metadata.json manquant")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Extraire la configuration critique
        config = metadata.get('config', {})
       # MAX_SEQUENCE_LENGTH = config.get('max_sequence_length')
        max_vocab_size = config.get('max_vocab_size')

      #  if MAX_SEQUENCE_LENGTH is None:
        #     raise ValueError("max_sequence_length non trouvé dans les métadonnées")

        print(f"✅ Configuration chargée:")
        print(f"  max_sequence_length: {MAX_SEQUENCE_LENGTH}")
        print(f"  max_vocab_size: {max_vocab_size}")

        # 2. CHARGER LES ARTEFACTS DANS LE BON ORDRE
        print("📦 Chargement des artefacts...")

        # Tokenizer
        with open(model_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        print(f"✅ Tokenizer chargé (vocab: {len(tokenizer.word_index)} mots)")

        # Scaler
        with open(model_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        print("✅ Scaler chargé")

        # Label encoder
        with open(model_dir / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        print(f"✅ Label encoder chargé (classes: {label_encoder.classes_})")

        # Modèle (en dernier pour vérifier la compatibilité)
        model_file = model_dir / "best_lstm_model.keras"
        if not model_file.exists():
            raise FileNotFoundError("Fichier best_lstm_model.keras manquant")

        model = load_model(model_file)
        print("✅ Modèle LSTM chargé")

        # Vérifier les dimensions du modèle
        print(f"🔍 Vérification des dimensions:")
        for i, input_layer in enumerate(model.inputs):
            print(f"  Input {i}: {input_layer.name} - Shape: {input_layer.shape}")

        # 3. CHARGER LES RESSOURCES SUPPLÉMENTAIRES
        # Mots suspects
        suspicious_words_file = model_dir / "suspicious_words.json"
        if suspicious_words_file.exists():
            with open(suspicious_words_file, 'r') as f:
                suspicious_words_data = json.load(f)
            SUSPICIOUS_WORDS_SET = set(suspicious_words_data.get('en', []) + suspicious_words_data.get('fr', []))
            print(f"✅ Mots suspects chargés ({len(SUSPICIOUS_WORDS_SET)} mots)")
        else:
            print("⚠️ Fichier suspicious_words.json manquant, utilisation d'une liste vide")

        # Stopwords NLTK
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

        print(f"\n🎉 API prête ! Longueur de séquence configurée: {MAX_SEQUENCE_LENGTH}")
        return True

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE AU DÉMARRAGE: {e}")
        print(f"❌ Type d'erreur: {type(e).__name__}")

        # Lister les fichiers présents pour debug
        if 'model_dir' in locals():
            print(f"📁 Fichiers dans {model_dir}:")
            try:
                for file in model_dir.iterdir():
                    print(f"  - {file.name}")
            except:
                print("  Impossible de lister les fichiers")

        return False


# Charger les artefacts au démarrage
model_loaded = load_model_artifacts()


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

        # Création des séquences avec la BONNE longueur
        sequence = tokenizer.texts_to_sequences([processed_text])
        # FILTRER les indices qui dépassent la taille du vocabulaire du modèle
        MAX_VOCAB_INDEX = 10000  # Limite du modèle
        if sequence[0]:  # Si la séquence n'est pas vide
            # Remplacer les indices >= 10000 par 1 (token OOV)
            sequence[0] = [idx if idx < MAX_VOCAB_INDEX else 1 for idx in sequence[0]]

        print(f"🔢 Séquence filtrée: longueur = {len(sequence[0]) if sequence[0] else 0}")
        print(f"🔢 Séquence brute: longueur = {len(sequence[0]) if sequence[0] else 0}")
        MAX_SEQUENCE_LENGTH = 566
        # CRITIQUE: Utiliser MAX_SEQUENCE_LENGTH du modèle
        padded_sequence = pad_sequences(
            sequence,
            maxlen=MAX_SEQUENCE_LENGTH,  # Utiliser la longueur correcte
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


# --- Fonctions de Gestion des Feedbacks Intelligents ---
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


def prepare_negative_feedback_dataset() -> Optional[pd.DataFrame]:
    """Prépare un dataset UNIQUEMENT avec les feedbacks négatifs"""
    try:
        if not FEEDBACK_CSV_PATH.exists():
            return None

        df = pd.read_csv(FEEDBACK_CSV_PATH)

        # Filtrer: feedbacks négatifs ET non traités
        negative_df = df[
            (df['user_satisfaction'] == 'no') &
            (df['processed'] == False)
            ].copy()

        if len(negative_df) == 0:
            return None

        # Pour les feedbacks négatifs, inverser la prédiction = label correct
        def get_correct_label(row):
            return 'legitimate' if row['predicted_class'] == 'phishing' else 'phishing'

        negative_df['correct_label'] = negative_df.apply(get_correct_label, axis=1)

        # Créer le dataset final
        retrain_df = pd.DataFrame({
            'text': negative_df['email_text'],
            'label': negative_df['correct_label'],
            'language': negative_df['language_detected']
        })

        print(f"📋 Dataset de feedbacks négatifs: {len(retrain_df)} échantillons")
        print(f"  Distribution: {retrain_df['label'].value_counts().to_dict()}")

        return retrain_df

    except Exception as e:
        print(f"❌ Erreur lors de la préparation du dataset négatif: {e}")
        return None


def mark_negative_feedbacks_processed():
    """Marque UNIQUEMENT les feedbacks négatifs comme traités"""
    try:
        if not FEEDBACK_CSV_PATH.exists():
            return

        df = pd.read_csv(FEEDBACK_CSV_PATH)

        # Marquer seulement les feedbacks négatifs ET non traités
        mask = (df['user_satisfaction'] == 'no') & (df['processed'] == False)
        df.loc[mask, 'processed'] = True

        processed_count = mask.sum()

        df.to_csv(FEEDBACK_CSV_PATH, index=False)

        print(f"✅ {processed_count} feedbacks négatifs marqués comme traités")
        print(f"📝 Feedbacks positifs conservés pour analyse future")

    except Exception as e:
        print(f"❌ Erreur lors du marquage des feedbacks négatifs: {e}")


def evaluate_and_compare_models(new_detector, X_text_test, X_num_test, y_test):
    """
    Compare les performances du nouveau modèle vs ancien modèle
    Retourne True si le nouveau modèle est meilleur
    """
    try:
        print("🏆 COMPARAISON DES PERFORMANCES")
        print("=" * 40)

        # 1. Évaluer l'ancien modèle (modèle actuel)
        global model
        if model is None:
            print("⚠️ Aucun modèle ancien à comparer, acceptation du nouveau")
            return True, {}

        y_test_encoded = new_detector.label_encoder.transform(y_test)

        # Prédictions ancien modèle
        old_pred_proba = model.predict([X_text_test, X_num_test], verbose=0)
        old_pred = (old_pred_proba > 0.5).astype(int)

        # Prédictions nouveau modèle
        new_pred_proba = new_detector.model.predict([X_text_test, X_num_test], verbose=0)
        new_pred = (new_pred_proba > 0.5).astype(int)

        # Calculer les métriques
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        old_accuracy = accuracy_score(y_test_encoded, old_pred)
        new_accuracy = accuracy_score(y_test_encoded, new_pred)

        old_f1 = f1_score(y_test_encoded, old_pred)
        new_f1 = f1_score(y_test_encoded, new_pred)

        old_precision = precision_score(y_test_encoded, old_pred)
        new_precision = precision_score(y_test_encoded, new_pred)

        old_recall = recall_score(y_test_encoded, old_pred)
        new_recall = recall_score(y_test_encoded, new_pred)

        # Affichage des résultats
        print(f"📊 ANCIEN MODÈLE:")
        print(f"  Accuracy:  {old_accuracy:.4f}")
        print(f"  F1-Score:  {old_f1:.4f}")
        print(f"  Precision: {old_precision:.4f}")
        print(f"  Recall:    {old_recall:.4f}")

        print(f"📊 NOUVEAU MODÈLE:")
        print(f"  Accuracy:  {new_accuracy:.4f}")
        print(f"  F1-Score:  {new_f1:.4f}")
        print(f"  Precision: {new_precision:.4f}")
        print(f"  Recall:    {new_recall:.4f}")

        # Calculer l'amélioration
        accuracy_improvement = new_accuracy - old_accuracy
        f1_improvement = new_f1 - old_f1

        print(f"📈 AMÉLIORATION:")
        print(f"  Accuracy:  {accuracy_improvement:+.4f}")
        print(f"  F1-Score:  {f1_improvement:+.4f}")

        # Critères de décision (vous pouvez ajuster ces seuils)
        MIN_ACCURACY_IMPROVEMENT = 0.01  # 1% d'amélioration minimum
        MIN_F1_IMPROVEMENT = 0.01

        is_better = (
                accuracy_improvement >= MIN_ACCURACY_IMPROVEMENT or
                f1_improvement >= MIN_F1_IMPROVEMENT
        )

        if is_better:
            print("✅ NOUVEAU MODÈLE ACCEPTÉ - Performances améliorées!")
        else:
            print("❌ NOUVEAU MODÈLE REJETÉ - Pas d'amélioration significative")

        return is_better, {
            'old_accuracy': old_accuracy,
            'new_accuracy': new_accuracy,
            'old_f1': old_f1,
            'new_f1': new_f1,
            'accuracy_improvement': accuracy_improvement,
            'f1_improvement': f1_improvement
        }

    except Exception as e:
        print(f"❌ Erreur lors de la comparaison: {e}")
        # En cas d'erreur, accepter le nouveau modèle par sécurité
        return True, {}


async def trigger_intelligent_retraining():
    """Réentraînement intelligent basé sur feedbacks négatifs"""
    global IS_RETRAINING, model, tokenizer, scaler, label_encoder

    with RETRAIN_LOCK:
        if IS_RETRAINING:
            return {"status": "already_running"}
        IS_RETRAINING = True

    try:
        print("🧠 RÉENTRAÎNEMENT INTELLIGENT")

        # 1. Vérifier les feedbacks négatifs
        negative_count = count_negative_feedbacks()
        if negative_count < NEGATIVE_FEEDBACK_THRESHOLD:
            return {"status": "insufficient_negative_feedback"}

        # 2. Préparer les feedbacks négatifs
        negative_df = prepare_negative_feedback_dataset()
        if negative_df is None:
            return {"status": "no_negative_data"}

        # 3. Créer nouveau détecteur
        from train_model import LSTMPhishingDetector

        config = {
            'epochs': 10,
            'batch_size': 64,
            'learning_rate': 0.0005,
            'patience': 3
        }

        new_detector = LSTMPhishingDetector(config)

        # 4. RÉENTRAÎNER avec la nouvelle fonction
        history, metrics, X_text_test, X_num_test, y_test = new_detector.retrain_from_feedback(
            negative_df,
            main_dataset_path='full_merged_dataset_fr_en_spam.csv',
            sample_size=2000
        )

        # 5. Comparer les modèles
        is_better, comparison = evaluate_and_compare_models(
            new_detector, X_text_test, X_num_test, y_test
        )

        if not is_better:
            print("❌ NOUVEAU MODÈLE REJETÉ")
            return {"status": "model_rejected"}

        # 6. Remplacer le modèle
        print("✅ NOUVEAU MODÈLE ACCEPTÉ")

        # Sauvegarder les nouveaux artefacts
        new_detector.save_model_artifacts("best_lstm_model")

        # Recharger dans l'API
        success = load_model_artifacts()
        if not success:
            return {"status": "reload_failed"}

        # Marquer les feedbacks comme traités
        mark_negative_feedbacks_processed()

        return {"status": "success", "metrics": comparison}

    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return {"status": "error", "message": str(e)}

    finally:
        with RETRAIN_LOCK:
            IS_RETRAINING = False


def save_feedback_to_csv(feedback_data):
    """
    Sauvegarde un feedback dans le fichier CSV
    """
    try:
        # Créer le dossier data s'il n'existe pas
        FEEDBACK_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Ajouter la colonne 'processed' si elle manque
        feedback_data['processed'] = False

        # Si le fichier n'existe pas, créer avec en-têtes
        if not FEEDBACK_CSV_PATH.exists():
            df = pd.DataFrame([feedback_data])
            df.to_csv(FEEDBACK_CSV_PATH, index=False)
            print(f"✅ Fichier feedback créé: {FEEDBACK_CSV_PATH}")
        else:
            # Ajouter au fichier existant
            df = pd.DataFrame([feedback_data])
            df.to_csv(FEEDBACK_CSV_PATH, mode='a', header=False, index=False)
            print(f"✅ Feedback ajouté au fichier: {FEEDBACK_CSV_PATH}")

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
        "documentation": "/docs"
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
        "negative_threshold": NEGATIVE_FEEDBACK_THRESHOLD
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


@app.get("/feedbacks", summary="Voir les feedbacks utilisateur")
def get_feedbacks():
    """
    Récupère tous les feedbacks enregistrés
    """
    try:
        csv_filename = Path("./data/user_feedbacks.csv")

        if not csv_filename.exists():
            return {
                "message": "Aucun feedback enregistré pour le moment",
                "feedbacks": [],
                "count": 0
            }

        feedbacks = []
        with open(csv_filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                feedbacks.append(row)

        return {
            "message": "Feedbacks récupérés avec succès",
            "count": len(feedbacks),
            "feedbacks": feedbacks,
            "file_location": str(csv_filename)
        }

    except Exception as e:
        print(f"❌ Erreur lecture feedbacks: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {e}")

@app.post("/feedback", summary="Enregistrer un feedback avec réentraînement intelligent")
async def save_feedback(feedback: FeedbackInput, background_tasks: BackgroundTasks):
    """
    Enregistre le feedback et déclenche un réentraînement intelligent
    après 10 feedbacks NÉGATIFS
    """
    try:
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "email_text": feedback.email_text[:500],
            "predicted_class": feedback.predicted_class,
            "predicted_probability": feedback.predicted_probability,
            "user_satisfaction": feedback.user_satisfaction,
            "language_detected": feedback.language_detected,
            "processed": False  # Ajouter explicitement cette colonne
        }

        # Sauvegarder le feedback
        if not save_feedback_to_csv(feedback_data):
            raise HTTPException(status_code=500, detail="Erreur sauvegarde feedback")

        # Compter UNIQUEMENT les feedbacks négatifs
        negative_count = count_negative_feedbacks()

        print(f"📝 Feedback enregistré: {feedback.user_satisfaction}")
        print(f"📊 Feedbacks négatifs: {negative_count}/{NEGATIVE_FEEDBACK_THRESHOLD}")

        # Déclencher le réentraînement SEULEMENT pour les feedbacks négatifs
        should_retrain = (
                feedback.user_satisfaction == "no" and
                negative_count >= NEGATIVE_FEEDBACK_THRESHOLD and
                not IS_RETRAINING
        )

        response = {
            "status": "success",
            "message": "Feedback enregistré avec succès",
            "feedback_type": feedback.user_satisfaction,
            "negative_feedbacks": negative_count,
            "negative_threshold": NEGATIVE_FEEDBACK_THRESHOLD,
            "will_retrain": should_retrain
        }

        if should_retrain:
            print(f"🚀 Seuil de {NEGATIVE_FEEDBACK_THRESHOLD} feedbacks NÉGATIFS atteint!")
            background_tasks.add_task(trigger_intelligent_retraining)
            response["message"] += " - Réentraînement intelligent déclenché!"

        return response

    except Exception as e:
        print(f"❌ Erreur feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {e}")

# --- Endpoint de diagnostic ---
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

    print("🚀 Démarrage de l'API de test sur http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)