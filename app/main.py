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
import subprocess
import threading
import numpy as np
from collections import deque
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
import io
import html

# Ajouter ces lignes après vos imports existants
from fastapi import Request
from fastapi.responses import JSONResponse
import time



# --- Configuration et Initialisation ---
app = FastAPI(
    title="API de Détection de Phishing Automatique (FR/EN) - Smart Percentile",
    description="Une API adaptative pour classifier des textes avec optimisation automatique des longueurs.",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
scaler = None
label_encoder = None
MAX_SEQUENCE_LENGTH = 566
SUSPICIOUS_WORDS_SET = set()
STOP_WORDS = {}

MODEL_SUPPORTS_VARIABLE_LENGTH = False
SMART_PERCENTILE_ENABLED = False
length_history = deque(maxlen=500)
current_production_length = None
adaptation_stats = {
    'total_predictions': 0,
    'adaptations_triggered': 0,
    'avg_efficiency': 0.0,
    'last_adaptation': None
}

# Variables existantes pour fine-tuning
AUTO_FINETUNING_ENABLED = True
IS_FINETUNING_RUNNING = False
FINETUNING_LOCK = threading.Lock()
NEGATIVE_FEEDBACK_THRESHOLD = 1
tf.config.set_visible_devices([], 'GPU')

FEEDBACK_CSV_PATH = Path("./data/user_feedbacks.csv")


def clean_input_field(text: str) -> str:
    """Nettoie un champ texte comme dans le frontend JS (HTML stripping, normalisation, entités, caractères spéciaux)"""
    if not text:
        return ""

    cleaned = text

    # 1. Supprimer les balises HTML
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # 2. Décoder les entités HTML (&nbsp;, &amp;, etc.)
    cleaned = html.unescape(cleaned)

    # 3. Supprimer les caractères de contrôle ASCII (non imprimables)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)

    # 4. Supprimer les caractères non souhaités (emojis cassés, symboles, etc.)
    cleaned = re.sub(r"[^\w\s\-.,!?@()[\]{}:;'\"€/àâçéèêëîïôöùûü%&*=+\\/|~`<>]", "", cleaned)

    # 5. Normaliser les espaces multiples
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def get_raw_email_from_csv(email_id: str) -> dict:
    """Récupère les données brutes d'un email depuis le fichier CSV"""
    print(f"🔍 Recherche du fichier CSV pour email ID: {email_id}")
    
    # Essayer plusieurs emplacements possibles (volume partagé en priorité)
    possible_paths = [
        Path("/shared/data/emails_live.csv"),  # Volume partagé Docker
        Path("./emails_live.csv"),             # Répertoire courant
        Path("./src/pages/emails_live.csv"),   # Dossier src/pages
        Path("/app/emails_live.csv"),          # Dans le container
    ]
    
    csv_path = None
    for path in possible_paths:
        print(f"🔍 Vérification: {path} - Existe: {path.exists()}")
        if path.exists():
            csv_path = path
            print(f"✅ Fichier trouvé: {csv_path}")
            break
    
    if csv_path is None:
        print(f"❌ Aucun fichier CSV trouvé")
        # Lister le contenu des dossiers pour debug
        for check_dir in ["/shared", "/shared/data", ".", "./src", "./src/pages"]:
            try:
                check_path = Path(check_dir)
                if check_path.exists():
                    files = list(check_path.iterdir())
                    print(f"📁 Contenu de {check_dir}: {[f.name for f in files]}")
            except:
                print(f"📁 Impossible de lire {check_dir}")
        
        raise HTTPException(status_code=404, detail=f"Fichier emails_live.csv non trouvé dans: {[str(p) for p in possible_paths]}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Chercher l'email par ID
        email_row = df[df['id'] == email_id]
        
        if email_row.empty:
            raise HTTPException(status_code=404, detail=f"Email avec ID {email_id} non trouvé")
        
        row = email_row.iloc[0]
        
        # Retourner les données brutes SANS nettoyage
        return {
            'id': row['id'],
            'from': str(row['from']) if not pd.isna(row['from']) else "",
            'subject': str(row['subject']) if not pd.isna(row['subject']) else "",
            'body': str(row['body']) if not pd.isna(row['body']) else "",
            'type': row['type'] if 'type' in row else 'unknown'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture CSV: {str(e)}")


def analyze_input_lengths(texts):
    """Quick analysis of input text lengths"""
    lengths = []
    for text in texts:
        if pd.isna(text):
            lengths.append(0)
        else:
            clean_text = str(text).lower()
            clean_text = re.sub(r'http[s]?://\S+', ' URL_TOKEN ', clean_text)
            clean_text = re.sub(r'\S+@\S+', ' EMAIL_TOKEN ', clean_text)
            clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
            words = clean_text.split()
            lengths.append(len(words))
    return lengths


def calculate_production_length(current_batch_lengths):
    """Calculate optimal length for this batch in production using smart_percentile"""
    global current_production_length

    all_lengths = list(current_batch_lengths)

    # Ajouter l'historique récent si disponible
    if len(length_history) > 0:
        recent_history = list(length_history)[-100:]  # 100 derniers échantillons
        all_lengths.extend(recent_history)

    # Vérifier si on a assez de données (minimum 10 échantillons)
    if len(all_lengths) < 10:
        # Pas assez de données, utiliser la longueur d'entraînement
        fallback = current_production_length or MAX_SEQUENCE_LENGTH
        print(f"  📊 Pas assez d'échantillons ({len(all_lengths)}<10), utilisation de {fallback}")
        return fallback

    # Appliquer smart_percentile sur les données combinées
    mean_length = np.mean(all_lengths)
    std_length = np.std(all_lengths)
    p95_length = int(np.percentile(all_lengths, 95))

    # Logique smart_percentile adaptée pour la production
    if std_length > mean_length * 0.8:  # Grande variabilité
        optimal = min(p95_length, int(mean_length + 1.5 * std_length))
        print(f"  🧠 Production: variabilité élevée (std={std_length:.1f}) → ajustement conservateur")
    else:
        optimal = p95_length
        print(f"  🧠 Production: distribution stable → 95e percentile")

    # Appliquer les limites (entre 30 et 800)
    optimal = max(optimal, 30)
    optimal = min(optimal, 800)

    # Calculer l'efficacité
    current_avg = np.mean(current_batch_lengths)
    efficiency = current_avg / optimal if optimal > 0 else 0
    print(f"  🎯 Longueur adaptée: {optimal} (efficacité: {efficiency:.2f})")

    return optimal


def prepare_sequences_adaptive(texts, languages, target_length):
    """Prepare sequences with a specific target length"""
    processed_texts = []
    for text, lang in zip(texts, languages):
        processed = preprocess_text(text, lang)
        processed_texts.append(processed)

    # Tokenization
    sequences = tokenizer.texts_to_sequences(processed_texts)

    # Filtrer les tokens qui dépassent la taille du vocabulaire
    max_vocab_id = 10000
    for i, sequence in enumerate(sequences):
        if sequence:
            sequences[i] = [token_id for token_id in sequence if token_id <= max_vocab_id]

    # Padding avec la longueur adaptée
    padded_sequences = pad_sequences(
        sequences,
        maxlen=target_length,
        padding='post',
        truncating='post'
    )

    return padded_sequences, processed_texts


# --- Chargement des Artefacts du Modèle (VERSION AMÉLIORÉE) ---
def load_model_artifacts():
    """Load all model artifacts with smart_percentile capability detection"""
    global model, tokenizer, scaler, label_encoder, MAX_SEQUENCE_LENGTH
    global SUSPICIOUS_WORDS_SET, STOP_WORDS
    global MODEL_SUPPORTS_VARIABLE_LENGTH, SMART_PERCENTILE_ENABLED, current_production_length

    try:
        print("🚀 Démarrage de l'API et chargement des artefacts...")

        # ÉTAPE 1: Charger les métadonnées en premier pour obtenir la bonne longueur
        metadata_file = Path("model/model_prod/model_metadata.json")
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                config = metadata.get('config', {})
                MAX_SEQUENCE_LENGTH = config.get('max_sequence_length', 566)

                # NOUVEAU: Détecter les capacités smart_percentile
                SMART_PERCENTILE_ENABLED = config.get('smart_percentile_enabled', False)
                MODEL_SUPPORTS_VARIABLE_LENGTH = config.get('enable_variable_length', False)

                print(f"✅ Métadonnées chargées:")
                print(f"  max_sequence_length: {MAX_SEQUENCE_LENGTH}")
                print(f"  🧠 Smart_percentile: {'✅ ACTIVÉ' if SMART_PERCENTILE_ENABLED else '❌ DÉSACTIVÉ'}")
                print(f"  🔄 Longueurs variables: {'✅ SUPPORTÉ' if MODEL_SUPPORTS_VARIABLE_LENGTH else '❌ FIXE'}")

                # Initialiser la longueur de production
                current_production_length = MAX_SEQUENCE_LENGTH

            except Exception as e:
                print(f"⚠️ Erreur chargement métadonnées: {e}")
                print(f"⚠️ Utilisation des valeurs par défaut")
        else:
            print(f"⚠️ Fichier model_metadata.json non trouvé")

        # ÉTAPE 2-8: Chargement standard des autres artefacts (identique à votre code)
        # [Garder tout votre code existant pour le chargement du modèle, tokenizer, etc.]

        # ÉTAPE 2: Chargement du modèle
        model_path = Path("model/model_prod/best_lstm_model.keras")
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

        model = load_model(str(model_path))
        print("✅ Modèle LSTM chargé")

        # ÉTAPE 3: Vérifier les dimensions du modèle
        print(f"🔍 Vérification des dimensions du modèle:")
        for i, input_layer in enumerate(model.inputs):
            print(f"  Input {i}: {input_layer.name} - Shape: {input_layer.shape}")

            # Vérifier si la forme correspond à notre MAX_SEQUENCE_LENGTH
            if i == 0 and len(input_layer.shape) >= 2:  # Premier input (texte)
                expected_seq_length = input_layer.shape[1]
                if expected_seq_length is None:
                    # Le modèle accepte vraiment des longueurs variables
                    MODEL_SUPPORTS_VARIABLE_LENGTH = True
                    print(f"  ✅ Modèle à longueurs variables confirmé")
                elif expected_seq_length != MAX_SEQUENCE_LENGTH:
                    print(f"  🔧 Correction longueur: {expected_seq_length}")
                    MAX_SEQUENCE_LENGTH = expected_seq_length
                    current_production_length = expected_seq_length

        # ÉTAPE 4: Chargement du tokenizer
        tokenizer_path = Path("model/model_prod/tokenizer.pkl")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer non trouvé: {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"✅ Tokenizer chargé (vocab: {len(tokenizer.word_index)} mots)")

        # ÉTAPE 5: Chargement du scaler
        scaler_path = Path("model/model_prod/scaler.pkl")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler non trouvé: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler chargé")

        # ÉTAPE 6: Chargement du label encoder
        label_encoder_path = Path("model/model_prod/label_encoder.pkl")
        if not label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder non trouvé: {label_encoder_path}")
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"✅ Label encoder chargé (classes: {label_encoder.classes_})")

        # ÉTAPE 7: Charger les mots suspects
        suspicious_words_file = Path("model/model_prod/suspicious_words.json")
        if suspicious_words_file.exists():
            try:
                with open(suspicious_words_file, 'r') as f:
                    suspicious_words_data = json.load(f)
                SUSPICIOUS_WORDS_SET = set(suspicious_words_data.get('en', []) + suspicious_words_data.get('fr', []))
                print(f"✅ Mots suspects chargés ({len(SUSPICIOUS_WORDS_SET)} mots)")
            except Exception as e:
                print(f"⚠️ Erreur chargement mots suspects: {e}")

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
            STOP_WORDS = {
                'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'},
                'fr': {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'dans', 'sur', 'avec', 'pour', 'de'}
            }

        # ÉTAPE 9: Test de prédiction pour vérifier le fonctionnement
        print(f"\n🧪 Test de validation du modèle...")
        try:
            test_text = "Test email content"
            test_processed = preprocess_text(test_text, 'en')
            test_sequence = tokenizer.texts_to_sequences([test_processed])
            test_padded = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
            test_features = extract_numerical_features(test_text)
            test_scaled = scaler.transform([test_features])
            test_pred = model.predict([test_padded, test_scaled], verbose=0)
            print(f"✅ Test de prédiction réussi: {test_pred[0][0]:.4f}")
        except Exception as e:
            print(f"❌ Échec du test de validation: {e}")
            return False

        # NOUVEAU: Affichage du résumé des capacités
        print(f"\n🎉 API prête ! Capacités activées:")
        print(f"  📏 Longueur de séquence: {MAX_SEQUENCE_LENGTH}")
        print(f"  🧠 Smart_percentile: {'✅ OUI' if SMART_PERCENTILE_ENABLED else '❌ NON'}")
        print(f"  🔄 Longueurs variables: {'✅ OUI' if MODEL_SUPPORTS_VARIABLE_LENGTH else '❌ NON'}")
        print(
            f"  📊 Adaptation en production: {'✅ DISPONIBLE' if (SMART_PERCENTILE_ENABLED and MODEL_SUPPORTS_VARIABLE_LENGTH) else '❌ INDISPONIBLE'}")

        return True

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE AU DÉMARRAGE: {e}")
        return False


# --- Modèles de Données Pydantic (NOUVEAUX) ---
class TextInput(BaseModel):
    text: str

class EmailIDInput(BaseModel):
    email_id: str


class TextInputAdaptive(BaseModel):
    text: str
    enable_adaptation: bool = True  # NOUVEAU: Permettre de désactiver l'adaptation


class BatchInput(BaseModel):
    items: List[TextInput]


class BatchInputAdaptive(BaseModel):  # NOUVEAU
    items: List[TextInput]
    enable_adaptation: bool = True


class FeedbackInput(BaseModel):
    email_text: str
    predicted_class: str
    predicted_probability: float
    user_satisfaction: str
    language_detected: str


# --- Fonctions de Prétraitement (IDENTIQUES) ---
def preprocess_text(text: str, language: str):
    """Specialized and multilingual text preprocessing."""
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
    """Extract numerical features aligned with training."""
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


# --- NOUVELLE FONCTION: Prédiction avec Smart Percentile ---
def perform_prediction_adaptive(texts, enable_adaptation=True):
    """
    Adaptive prediction with smart_percentile in production

    Args:
        texts: List of texts to analyze or single text
        enable_adaptation: If True, adapt length according to smart_percentile

    Returns:
        dict: Results with adaptation information
    """
    global adaptation_stats, length_history, current_production_length

    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Normaliser l'entrée en liste
    if isinstance(texts, str):
        texts = [texts]

    try:
        adaptation_stats['total_predictions'] += len(texts)

        # Détection des langues
        languages = []
        for text in texts:
            try:
                detected_lang = detect(text[:1000])
                lang = detected_lang if detected_lang in ['fr', 'en'] else 'en'
            except Exception:
                lang = 'en'
            languages.append(lang)

        print(f"🔮 PRÉDICTION SMART_PERCENTILE")
        print(f"   Nombre de textes: {len(texts)}")
        print(f"   Adaptation activée: {enable_adaptation}")
        print(f"   Modèle supporte variables: {MODEL_SUPPORTS_VARIABLE_LENGTH}")
        print(f"   Smart_percentile activé: {SMART_PERCENTILE_ENABLED}")

        # Décider si on doit adapter
        should_adapt = (enable_adaptation and
                        SMART_PERCENTILE_ENABLED and
                        MODEL_SUPPORTS_VARIABLE_LENGTH)

        if should_adapt:
            # 1. Analyser les longueurs du batch actuel
            current_lengths = analyze_input_lengths(texts)
            print(
                f"   Longueurs actuelles: {min(current_lengths)}-{max(current_lengths)} (moy: {np.mean(current_lengths):.1f})")

            # 2. Calculer la longueur optimale avec smart_percentile
            optimal_length = calculate_production_length(current_lengths)

            # 3. Vérifier si on doit vraiment adapter
            training_length = MAX_SEQUENCE_LENGTH
            if abs(optimal_length - training_length) > 10:  # Seuil de changement
                print(f"   🔄 Adaptation: {training_length} → {optimal_length}")
                # Utiliser la longueur adaptée
                padded_sequences, processed_texts = prepare_sequences_adaptive(texts, languages, optimal_length)
                current_production_length = optimal_length
                adaptation_stats['adaptations_triggered'] += 1
                adaptation_stats['last_adaptation'] = datetime.now().isoformat()
                adaptation_triggered = True
                actual_length_used = optimal_length
            else:
                print(f"   ⚪ Pas d'adaptation nécessaire (écart < 10)")
                # Utiliser la méthode standard
                padded_sequences, processed_texts = prepare_sequences_adaptive(texts, languages, training_length)
                adaptation_triggered = False
                actual_length_used = training_length

            # 4. Mettre à jour l'historique pour les futures prédictions
            length_history.extend(current_lengths)

            # 5. Calculer l'efficacité
            efficiency = np.mean(current_lengths) / actual_length_used if actual_length_used > 0 else 1.0
            adaptation_stats['avg_efficiency'] = (adaptation_stats['avg_efficiency'] * 0.9 + efficiency * 0.1)

        else:
            # Mode standard sans adaptation
            print(f"   📋 Mode standard (adaptation désactivée)")
            padded_sequences, processed_texts = prepare_sequences_adaptive(texts, languages, MAX_SEQUENCE_LENGTH)
            adaptation_triggered = False
            actual_length_used = MAX_SEQUENCE_LENGTH
            efficiency = 1.0

        # 6. Extraction des features numériques
        numerical_features = []
        for text in texts:
            features = extract_numerical_features(text)
            numerical_features.append(features)
        scaled_features = scaler.transform(numerical_features)

        # 7. Prédiction
        probabilities = model.predict([padded_sequences, scaled_features], verbose=0)

        # 8. Préparer les résultats
        results = []
        for i, (text, lang, prob) in enumerate(zip(texts, languages, probabilities)):
            prediction_int = int(prob[0] > 0.5)
            predicted_class = label_encoder.inverse_transform([prediction_int])[0]

            confidence_score = abs(prob[0] - 0.5) * 2
            if confidence_score > 0.8:
                confidence = "HIGH"
            elif confidence_score > 0.4:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            result = {
                "prediction": predicted_class,
                "probability": float(prob[0]),
                "confidence": confidence,
                "language_detected": lang,
                "sequence_length_used": actual_length_used,
                "adaptation_info": {
                    "adaptation_enabled": should_adapt,
                    "adaptation_triggered": adaptation_triggered,
                    "efficiency": efficiency if should_adapt else None,
                    "original_length": len(processed_texts[i].split()) if processed_texts else None
                } if should_adapt else None
            }
            results.append(result)

        # Retourner un seul résultat si un seul texte en entrée
        if len(results) == 1:
            return results[0]
        else:
            return {"results": results}

    except Exception as e:
        print(f"❌ Erreur dans perform_prediction_adaptive: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


# --- Logique de prédiction principale (VERSION ORIGINALE CONSERVÉE) ---
def perform_prediction(text: str):
    """Original function kept for compatibility"""
    # Votre code existant identique
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        try:
            detected_lang = detect(text[:1000])
            lang = detected_lang if detected_lang in ['fr', 'en'] else 'en'
        except Exception:
            lang = 'en'

        processed_text = preprocess_text(text, lang)
        sequence = tokenizer.texts_to_sequences([processed_text])

        if sequence[0]:
            max_vocab_id = 10000
            sequence[0] = [token_id for token_id in sequence[0] if token_id <= max_vocab_id]

        padded_sequence = pad_sequences(
            sequence,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding='post',
            truncating='post'
        )

        numerical_features = extract_numerical_features(text)
        scaled_features = scaler.transform([numerical_features])

        prediction_proba = model.predict([padded_sequence, scaled_features], verbose=0)[0][0]
        prediction_int = int(prediction_proba > 0.5)
        predicted_class = label_encoder.inverse_transform([prediction_int])[0]

        confidence_score = abs(prediction_proba - 0.5) * 2
        if confidence_score > 0.8:
            confidence = "HIGH"
        elif confidence_score > 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        print(f"✅ PREDICTION: {predicted_class} (prob: {prediction_proba:.4f}, confidence: {confidence})")

        return {
            "prediction": predicted_class,
            "probability": float(confidence_score),
            "confidence": confidence,
            "language_detected": lang,
            "sequence_length_used": MAX_SEQUENCE_LENGTH,
            "debug_info": {
                "processed_text_length": len(processed_text),
                "original_sequence_length": len(sequence[0]) if sequence[0] else 0,
                "padded_sequence_shape": list(padded_sequence.shape),
                "features_shape": list(scaled_features.shape)
            }
        }

    except Exception as e:
        print(f"❌ Erreur dans perform_prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")



def count_negative_feedbacks() -> int:
    """Count only unprocessed negative feedbacks"""
    try:
        if not FEEDBACK_CSV_PATH.exists():
            return 0
        df = pd.read_csv(FEEDBACK_CSV_PATH)
        negative_unprocessed = len(df[
                                       (df['user_satisfaction'] == 'no') &
                                       (df['processed'] == False)
                                       ])
        return negative_unprocessed
    except Exception as e:
        print(f"❌ Erreur lors du comptage des feedbacks négatifs: {e}")
        return 0


def save_feedback_to_csv(feedback_data):
    """Save feedback to CSV file"""
    try:
        csv_headers = [
            'timestamp', 'email_text', 'predicted_class',
            'predicted_probability', 'user_satisfaction', 'language_detected', 'processed'
        ]
        feedback_data['processed'] = False
        FEEDBACK_CSV_PATH.parent.mkdir(exist_ok=True)
        file_exists = FEEDBACK_CSV_PATH.exists()
        with open(FEEDBACK_CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(feedback_data)
        return True
    except Exception as e:
        print(f"❌ Erreur sauvegarde feedback: {e}")
        return False

def run_finetuning_script():
    """
    Lance le script de fine-tuning en arrière-plan et AFFICHE LES LOGS EN TEMPS RÉEL.
    """
    global IS_FINETUNING_RUNNING

    print("🚀 DÉMARRAGE DU FINE-TUNING AUTOMATIQUE (avec logs en temps réel)")
    print("=" * 60)

    try:
        # Marquer que le fine-tuning est en cours
        IS_FINETUNING_RUNNING = True

        script_path = Path("traitement.py")
        if not script_path.exists():
            print(f"❌ Script de fine-tuning non trouvé: {script_path}")
            IS_FINETUNING_RUNNING = False
            return False

        # Lancer le processus avec Popen pour capturer la sortie en temps réel
        process = subprocess.Popen(
            ["python", "-u", "traitement.py"],  # Le flag -u est CRUCIAL pour désactiver le buffering
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirige stderr vers stdout pour tout voir
            text=True,
            encoding='utf-8',
            errors='replace' # Gère les erreurs de décodage
        )

        print("🎯 Processus de fine-tuning lancé. Affichage des logs...")
        print("-" * 60)

        # Lire la sortie ligne par ligne et l'afficher
        while True:
            output_line = process.stdout.readline()
            if output_line == '' and process.poll() is not None:
                break
            if output_line:
                # Affiche la ligne dans la console de l'API
                print(f"FT-LOG | {output_line.strip()}", flush=True)

        # Attendre la fin du processus et récupérer le code de retour
        return_code = process.poll()
        print("-" * 60)

        if return_code == 0:
            print("✅ FINE-TUNING TERMINÉ AVEC SUCCÈS!")
            print("💡 Pour utiliser le nouveau modèle, redémarrez l'API si le déploiement a eu lieu:")
            print("   docker-compose restart fastapi")
            return True
        else:
            print(f"❌ FINE-TUNING ÉCHOUÉ! (Code de retour: {return_code})")
            return False

    except Exception as e:
        print(f"❌ Erreur critique lors du lancement du fine-tuning: {e}")
        return False
    finally:
        # S'assurer que le statut est bien réinitialisé
        IS_FINETUNING_RUNNING = False
        print("=" * 60)
        print("🚀 Processus de fine-tuning terminé.")

def trigger_automatic_finetuning():
    """
    Déclenche le fine-tuning automatique en arrière-plan
    """
    global IS_FINETUNING_RUNNING

    # Utiliser un verrou pour éviter les conditions de course
    with FINETUNING_LOCK:
        if IS_FINETUNING_RUNNING:
            print("⚠️ Fine-tuning déjà en cours, ignorer le déclenchement")
            return False

        if not AUTO_FINETUNING_ENABLED:
            print("⚠️ Fine-tuning automatique désactivé")
            return False

        # Lancer le fine-tuning dans un thread séparé
        finetuning_thread = threading.Thread(
            target=run_finetuning_script,
            name="AutoFineTuning"
        )
        finetuning_thread.daemon = True
        finetuning_thread.start()

        # ---- LA MODIFICATION EST ICI ----
        # Ajout d'un \n pour éviter le collage des logs
        print("\n🚀 Fine-tuning automatique lancé en arrière-plan.")
        # -------------------------------
        return True


def check_and_trigger_finetuning():
    """
    Vérifie si les conditions pour déclencher le fine-tuning sont remplies et lance si nécessaire
    """
    try:
        negative_count = count_negative_feedbacks()

        if negative_count >= NEGATIVE_FEEDBACK_THRESHOLD:
            print(f"🚨 Seuil de fine-tuning atteint: {negative_count}/{NEGATIVE_FEEDBACK_THRESHOLD} feedbacks négatifs")

            if AUTO_FINETUNING_ENABLED and not IS_FINETUNING_RUNNING:
                print("🚀 Déclenchement automatique du fine-tuning...")
                return trigger_automatic_finetuning()
            elif IS_FINETUNING_RUNNING:
                print("⚠️ Fine-tuning déjà en cours")
                return False
            else:
                print("💡 Fine-tuning automatique désactivé, déclenchement manuel requis")
                return False
        else:
            print(f"📊 Feedbacks négatifs: {negative_count}/{NEGATIVE_FEEDBACK_THRESHOLD}")
            return False

    except Exception as e:
        print(f"❌ Erreur vérification fine-tuning: {e}")
        return False


def reload_model_artifacts():
    """
    Recharge le modèle et tous les artefacts depuis le disque
    """
    global model, tokenizer, scaler, label_encoder, MAX_SEQUENCE_LENGTH, SUSPICIOUS_WORDS_SET, STOP_WORDS

    print("🔄 RECHARGEMENT DU MODÈLE EN COURS...")
    print("=" * 50)

    try:
        # Sauvegarder l'ancien modèle au cas où
        old_model = model

        # Recharger tous les artefacts
        success = load_model_artifacts()

        if success:
            print("✅ MODÈLE RECHARGÉ AVEC SUCCÈS!")
            print(f"   Nouvelle longueur de séquence: {MAX_SEQUENCE_LENGTH}")
            print(f"   Vocabulaire: {len(tokenizer.word_index)} mots")

            # Test rapide du nouveau modèle
            try:
                test_text = "Test de validation du nouveau modèle"
                test_result = perform_prediction(test_text)
                print(f"   Test de validation réussi: {test_result['prediction']}")

                return True
            except Exception as e:
                print(f"❌ Test de validation échoué: {e}")
                # Restaurer l'ancien modèle si possible
                if old_model is not None:
                    model = old_model
                    print("🔄 Ancien modèle restauré")
                return False
        else:
            print("❌ Échec du rechargement")
            return False

    except Exception as e:
        print(f"❌ Erreur critique lors du rechargement: {e}")
        return False


# [Garder toutes vos autres fonctions de fine-tuning existantes]

# --- NOUVEAUX ENDPOINTS SMART_PERCENTILE ---

@app.get("/", summary="Welcome message with capabilities")
def read_root():
    return {
        "message": "Bienvenue sur l'API de détection de phishing (LSTM Hybride FR/EN)",
        "version": app.version,
        "documentation": "/docs",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "model_loaded": model is not None,
        "smart_percentile_enabled": SMART_PERCENTILE_ENABLED,
        "variable_length_supported": MODEL_SUPPORTS_VARIABLE_LENGTH,
        "adaptive_prediction_available": SMART_PERCENTILE_ENABLED and MODEL_SUPPORTS_VARIABLE_LENGTH
    }


@app.get("/health", summary="Health check with adaptation info")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded")

    negative_feedbacks = count_negative_feedbacks()
    finetuning_ready = negative_feedbacks >= NEGATIVE_FEEDBACK_THRESHOLD

    return {
        "status": "healthy",
        "model_loaded": True,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "vocab_size": len(tokenizer.word_index) if tokenizer else 0,
        "negative_feedbacks": negative_feedbacks,
        "finetuning_ready": finetuning_ready,
        "model_classes": list(label_encoder.classes_) if label_encoder else [],

        # NOUVELLES INFORMATIONS
        "smart_percentile_capabilities": {
            "enabled": SMART_PERCENTILE_ENABLED,
            "variable_length_supported": MODEL_SUPPORTS_VARIABLE_LENGTH,
            "adaptive_prediction_available": SMART_PERCENTILE_ENABLED and MODEL_SUPPORTS_VARIABLE_LENGTH,
            "current_production_length": current_production_length,
            "adaptation_stats": adaptation_stats
        }
    }


@app.post("/predict", summary="Predict on single text (standard mode)")
def predict(item: TextInput):
    """Analyze text with standard method (compatibility mode)"""
    print(f"⚠️ ANCIEN ENDPOINT /predict APPELÉ avec texte de {len(item.text)} chars")
    
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        result = perform_prediction(item.text)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {e}")


@app.post("/predict/email-id", summary="Predict using raw data from CSV by email ID")
def predict_by_email_id(item: EmailIDInput):
    """Analyze email using raw data directly from CSV file"""
    print(f"🎯 ENDPOINT /predict/email-id APPELÉ avec ID: {item.email_id}")
    
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        # Récupérer les données brutes depuis le CSV
        email_data = get_raw_email_from_csv(item.email_id)
        
        # Créer le texte combiné avec les données BRUTES
        raw_text = f"From: {email_data['from']}\nSubject: {email_data['subject']}\nBody: {email_data['body']}"
        
        print(f"🔍 RAW EMAIL ID {item.email_id} - INPUT (1000 chars):")
        print(f"'{raw_text[:1000]}'")
        print(f"Total length: {len(raw_text)}")
        
        # Prédiction avec les données brutes
        result = perform_prediction(raw_text)
        
        # Ajouter l'ID de l'email dans la réponse
        result['email_id'] = item.email_id
        result['data_source'] = 'raw_csv'
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {e}")


@app.post("/predict/adaptive", summary="Predict with smart_percentile adaptation")
def predict_adaptive(item: TextInputAdaptive):
    """
    Analyze text with automatic smart_percentile adaptation
    Optimize sequence length according to text characteristics
    """
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    if not (SMART_PERCENTILE_ENABLED and MODEL_SUPPORTS_VARIABLE_LENGTH):
        # Fallback sur la méthode standard si pas de support
        result = perform_prediction(item.text)
        result["adaptation_info"] = {
            "adaptation_available": False,
            "reason": "Modèle ne supporte pas l'adaptation"
        }
        return result

    try:
        result = perform_prediction_adaptive([item.text], item.enable_adaptation)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {e}")


@app.post("/predict/batch", summary="Predict on text list (standard mode)")
def predict_batch(batch: BatchInput):
    """Analyze a list of texts in standard mode"""
    results = []
    for item in batch.items:
        try:
            result = perform_prediction(item.text)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "text": item.text[:50] + "..."})
    return {"results": results}


@app.post("/predict/batch/adaptive", summary="Predict on list with smart_percentile adaptation")
def predict_batch_adaptive(batch: BatchInputAdaptive):
    """
    Analyze a list of texts with automatic smart_percentile adaptation
    Optimize globally for the entire batch
    """
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    if not (SMART_PERCENTILE_ENABLED and MODEL_SUPPORTS_VARIABLE_LENGTH):
        # Fallback sur la méthode standard
        results = []
        for item in batch.items:
            try:
                result = perform_prediction(item.text)
                result["adaptation_info"] = {"adaptation_available": False}
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "text": item.text[:50] + "..."})
        return {"results": results}

    try:
        texts = [item.text for item in batch.items]
        result = perform_prediction_adaptive(texts, batch.enable_adaptation)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {e}")


@app.get("/adaptation/stats", summary="Smart_percentile adaptation statistics")
def get_adaptation_stats():
    """Return smart_percentile adaptation statistics"""
    if not SMART_PERCENTILE_ENABLED:
        return {
            "smart_percentile_enabled": False,
            "message": "Smart_percentile non activé sur ce modèle"
        }

    recent_lengths = list(length_history)[-50:] if len(length_history) > 0 else []

    return {
        "smart_percentile_enabled": True,
        "variable_length_supported": MODEL_SUPPORTS_VARIABLE_LENGTH,
        "current_production_length": current_production_length,
        "training_length": MAX_SEQUENCE_LENGTH,
        "adaptation_stats": adaptation_stats,
        "length_history_size": len(length_history),
        "recent_analysis": {
            "sample_size": len(recent_lengths),
            "avg_length": np.mean(recent_lengths) if recent_lengths else 0,
            "min_max": (min(recent_lengths), max(recent_lengths)) if recent_lengths else (0, 0),
            "efficiency": adaptation_stats['avg_efficiency']
        } if recent_lengths else None
    }


@app.post("/adaptation/reset", summary="Reset adaptation history")
def reset_adaptation_history():
    """Reset adaptation history (useful for testing)"""
    global length_history, adaptation_stats, current_production_length

    length_history.clear()
    adaptation_stats = {
        'total_predictions': 0,
        'adaptations_triggered': 0,
        'avg_efficiency': 0.0,
        'last_adaptation': None
    }
    current_production_length = MAX_SEQUENCE_LENGTH

    return {
        "status": "success",
        "message": "Historique d'adaptation réinitialisé",
        "reset_to_length": MAX_SEQUENCE_LENGTH
    }


# --- ENDPOINTS EXISTANTS (GARDER IDENTIQUES) ---
# [Garder tous vos endpoints existants : feedback, feedbacks, debug/model-info, etc.]

@app.post("/feedback", summary="Save user feedback")
async def save_feedback(feedback: FeedbackInput, background_tasks: BackgroundTasks):
    """Save user feedback and automatically trigger fine-tuning if necessary"""
    # [Garder votre code existant identique]
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

        print(f"📝 Feedback enregistré: {feedback.user_satisfaction}")

        # ✨ NOUVEAU: Vérifier et déclencher automatiquement le fine-tuning
        auto_triggered = False
        if feedback.user_satisfaction == "no":  # Seulement pour les feedbacks négatifs
            auto_triggered = check_and_trigger_finetuning()

        negative_count = count_negative_feedbacks()
        finetuning_ready = negative_count >= NEGATIVE_FEEDBACK_THRESHOLD

        print(f"📊 Feedbacks négatifs: {negative_count}")

        response = {
            "status": "success",
            "message": "Feedback enregistré avec succès",
            "feedback_type": feedback.user_satisfaction,
            "negative_feedbacks": negative_count,
            "finetuning_ready": finetuning_ready,
            "auto_finetuning_triggered": auto_triggered,  # ✨ NOUVEAU
            "is_finetuning_running": IS_FINETUNING_RUNNING  # ✨ NOUVEAU
        }

        if auto_triggered:
            response["finetuning_message"] = "🚀 Fine-tuning automatique lancé en arrière-plan!"
        elif finetuning_ready and not IS_FINETUNING_RUNNING:
            response["finetuning_message"] = "Seuil atteint! Fine-tuning prêt à être lancé"
        elif IS_FINETUNING_RUNNING:
            response["finetuning_message"] = "Fine-tuning en cours d'exécution..."

        return response

    except Exception as e:
        print(f"❌ Erreur feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {e}")

@app.get("/feedbacks", summary="View user feedbacks")
def get_feedbacks():
    """
    Retrieve all recorded feedbacks
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


@app.get("/feedbacks/stats", summary="Feedback statistics")
def get_feedback_stats():
    """
    Retrieve feedback statistics for monitoring
    """
    try:
        if not FEEDBACK_CSV_PATH.exists():
            return {
                "total_feedbacks": 0,
                "positive_feedbacks": 0,
                "negative_feedbacks": 0,
                "negative_unprocessed": 0,
                "finetuning_ready": False
            }

        df = pd.read_csv(FEEDBACK_CSV_PATH)

        total = len(df)
        positive = len(df[df['user_satisfaction'] == 'yes'])
        negative = len(df[df['user_satisfaction'] == 'no'])
        negative_unprocessed = len(df[
                                       (df['user_satisfaction'] == 'no') &
                                       (df['processed'] == False)
                                       ])

        finetuning_ready = negative_unprocessed >= 1

        return {
            "total_feedbacks": total,
            "positive_feedbacks": positive,
            "negative_feedbacks": negative,
            "negative_unprocessed": negative_unprocessed,
            "finetuning_ready": finetuning_ready,
            "success_rate": round((positive / total * 100), 2) if total > 0 else 0
        }

    except Exception as e:
        print(f"❌ Erreur stats feedbacks: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {e}")


@app.get("/debug/model-info", summary="Model diagnostic information")
def get_model_info():
    """Endpoint to diagnose model issues"""
    if not model:
        return {"error": "Modèle non chargé", "model_loaded": False}

    try:
        info = {
            "model_loaded": True,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "tokenizer_vocab_size": len(tokenizer.word_index) if tokenizer else 0,
            "model_inputs": [],
            "suspicious_words_count": len(SUSPICIOUS_WORDS_SET),
            "stopwords_languages": list(STOP_WORDS.keys()),
            "label_classes": label_encoder.classes_.tolist() if label_encoder else [],
            "model_summary": None
        }

        if model:
            for i, input_layer in enumerate(model.inputs):
                info["model_inputs"].append({
                    "index": i,
                    "name": input_layer.name,
                    "shape": input_layer.shape.as_list(),
                    "dtype": str(input_layer.dtype)
                })

            # Ajouter un résumé du modèle
            try:
                import io
                import sys
                from contextlib import redirect_stdout

                f = io.StringIO()
                with redirect_stdout(f):
                    model.summary()
                info["model_summary"] = f.getvalue()
            except Exception:
                info["model_summary"] = "Impossible de générer le résumé du modèle"

        return info
    except Exception as e:
        return {"error": str(e), "model_loaded": model is not None}


# --- Nouveau endpoint pour déclencher le fine-tuning ---
@app.post("/trigger-finetuning", summary="Trigger fine-tuning (development only)")
def trigger_finetuning():
    """
    Endpoint to check if fine-tuning can be triggered
    Note: Actual fine-tuning must be executed via 'python traitement.py'
    """
    try:
        negative_count = count_negative_feedbacks()

        if negative_count >= 1:
            return {
                "status": "ready",
                "message": "Fine-tuning peut être déclenché",
                "negative_feedbacks": negative_count,
                "command": "python traitement.py",
                "note": "Exécutez cette commande dans le terminal pour démarrer le fine-tuning"
            }
        else:
            return {
                "status": "not_ready",
                "message": f"Pas assez de feedbacks négatifs ({negative_count}/5)",
                "negative_feedbacks": negative_count,
                "needed": 1 - negative_count
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {e}")


@app.api_route("/reload-model", methods=["GET", "POST"], summary="Recharger le modèle depuis le disque")
async def reload_model_endpoint():
    """
    Force le rechargement du modèle depuis le disque
    """
    try:
        print("🔄 RECHARGEMENT MANUEL DU MODÈLE DEMANDÉ...")

        success = reload_model_artifacts()

        if success:
            # CORRECTION: Récupérer les métadonnées correctement
            try:
                metadata_file = Path("model/model_prod/model_metadata.json")
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        current_metadata = json.load(f)
                else:
                    current_metadata = {}
            except:
                current_metadata = {}

            return {
                "status": "success",
                "message": "Modèle rechargé avec succès",
                "model_version": current_metadata.get('model_version', 'unknown'),
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "vocab_size": len(tokenizer.word_index) if tokenizer else 0,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Échec du rechargement du modèle"
            )

    except Exception as e:
        print(f"❌ Erreur rechargement manuel: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur rechargement: {str(e)}"
        )


# CORRECTION 2: Améliorer l'endpoint /model-status

@app.get("/model-status", summary="Statut détaillé du modèle")
def get_model_status():
    """
    Retourne des informations détaillées sur le modèle actuel
    """
    try:
        metadata_file = Path("model/model_prod/model_metadata.json")
        current_metadata = {}

        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                current_metadata = json.load(f)

        model_file_path = Path("model/model_prod/best_lstm_model.keras")

        return {
            "model_loaded": model is not None,
            "model_version": current_metadata.get('model_version', 'unknown'),
            "last_retraining": current_metadata.get('last_individual_retraining', 'never'),
            "last_feedback_processed": current_metadata.get('last_feedback_processed', 'none'),
            "total_retrainings": current_metadata.get('total_individual_retrainings', 0),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "vocab_size": len(tokenizer.word_index) if tokenizer else 0,
            "suspicious_words_count": len(SUSPICIOUS_WORDS_SET),
            "deployment_method": current_metadata.get('deployment_method', 'initial_load'),
            "model_file_exists": model_file_path.exists(),
            "model_file_modified": datetime.fromtimestamp(
                model_file_path.stat().st_mtime
            ).isoformat() if model_file_path.exists() else None,
            "check_timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "error": str(e),
            "model_loaded": model is not None,
            "check_timestamp": datetime.now().isoformat()
        }

@app.post("/process-csv", summary="Process CSV file with ML predictions")
async def process_csv_file(file: UploadFile = File(...)):
    """
    Process email CSV file with ML model and return predictions
    """
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        print(f"\n🔥 === DÉBUT TRAITEMENT CSV ===")
        print(f"📁 Fichier reçu: {file.filename}")
        print(f"📁 Content-Type: {file.content_type}")

        # Vérifier le type de fichier
        if not file.filename.endswith('.csv'):
            print(f"❌ Type de fichier invalide: {file.filename}")
            raise HTTPException(status_code=400, detail="Le fichier doit être un CSV")

        # Lire le fichier CSV uploadé
        print(f"📖 Lecture du fichier...")
        content = await file.read()
        print(f"📖 Taille du fichier: {len(content)} bytes")

        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        print(f"📊 CSV parsé avec succès")
        print(f"📊 Colonnes détectées: {list(df.columns)}")
        print(f"📊 Nombre de lignes: {len(df)}")

        # Vérifier les colonnes requises - From, Subject, Body comme sur l'interface web
        required_columns = ['from', 'subject', 'body', 'type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Colonnes manquantes: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes: {missing_columns}"
            )

        print(f"✅ Validation des colonnes OK")
        print(f"📊 Distribution AVANT traitement:")
        distribution_avant = df['type'].value_counts()
        for type_email, count in distribution_avant.items():
            print(f"   - {type_email}: {count}")

        # Traitement par batch pour éviter les timeouts
        batch_size = 50
        all_results = []
        total_batches = (len(df) + batch_size - 1) // batch_size

        print(f"🔄 Début du traitement par batch ({total_batches} batches de {batch_size})")

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_results = []
            batch_num = i // batch_size + 1

            print(f"🚀 === BATCH {batch_num}/{total_batches} ===")
            print(f"   📊 Emails dans ce batch: {len(batch_df)}")

            # Traiter chaque email du batch
            for j, row in batch_df.iterrows():
                email_index = i + (j - batch_df.index[0]) + 1
                try:
                    # Utiliser les données BRUTES comme predict_by_email_id
                    from_field = str(row['from']) if not pd.isna(row['from']) else ""
                    subject_field = str(row['subject']) if not pd.isna(row['subject']) else ""
                    body_field = str(row['body']) if not pd.isna(row['body']) else ""
                    
                    # Créer le texte combiné pour l'analyse (même format que l'interface web)
                    combined_text = f"From: {from_field}\nSubject: {subject_field}\nBody: {body_field}".strip()

                    if not combined_text or combined_text == "From: \nSubject: \nBody: ":
                        print(f"   ⚪ Email {email_index}: Vide - type conservé")
                        # Email vide - garder le type original
                        original_type = row['type']
                        batch_results.append({
                            'new_type': original_type,
                            'prediction': 'unknown',
                            'probability': 0.0,
                            'confidence': 'LOW'
                        })
                        continue

                    # Prédiction avec le modèle en utilisant le texte combiné
                    print(f"   🧠 Email {email_index}: Analyse ML en cours...")
                    result = perform_prediction(combined_text)

                    # Mapper la prédiction vers IMPORTANT/SPAM
                    new_type = 'SPAM' if result['prediction'] == 'phishing' else 'IMPORTANT'
                    old_type = row['type']

                    if new_type != old_type:
                        print(f"   🔄 Email {email_index}: {old_type} → {new_type} (conf: {result['confidence']})")
                    else:
                        print(f"   ✅ Email {email_index}: {new_type} confirmé (conf: {result['confidence']})")

                    batch_results.append({
                        'new_type': new_type,
                        'prediction': result['prediction'],
                        'probability': result['probability'],
                        'confidence': result['confidence']
                    })

                except Exception as e:
                    print(f"   ❌ Email {email_index}: ERREUR - {str(e)}")
                    # En cas d'erreur, garder le type original
                    original_type = row['type']
                    batch_results.append({
                        'new_type': original_type,
                        'prediction': 'error',
                        'probability': 0.0,
                        'confidence': 'LOW'
                    })

            all_results.extend(batch_results)
            print(f"✅ Batch {batch_num} terminé ({len(batch_results)} emails traités)")

        print(f"🎯 === TRAITEMENT TERMINÉ ===")

        # Mettre à jour le DataFrame avec les nouvelles prédictions
        results_df = pd.DataFrame(all_results)
        df['type'] = results_df['new_type']

        # Statistiques détaillées
        print(f"📈 === STATISTIQUES FINALES ===")
        distribution_apres = df['type'].value_counts()
        for type_email, count in distribution_apres.items():
            print(f"   - {type_email}: {count}")

        # Calculer les changements
        if len(distribution_avant) > 0 and len(distribution_apres) > 0:
            spam_avant = distribution_avant.get('SPAM', 0)
            spam_apres = distribution_apres.get('SPAM', 0)
            important_avant = distribution_avant.get('IMPORTANT', 0)
            important_apres = distribution_apres.get('IMPORTANT', 0)

            print(f"📊 ÉVOLUTION:")
            print(f"   - SPAM: {spam_avant} → {spam_apres} ({spam_apres - spam_avant:+d})")
            print(f"   - IMPORTANT: {important_avant} → {important_apres} ({important_apres - important_avant:+d})")

        # Préparer les colonnes de sortie
        output_columns = ["id", "type", "from", "to", "date", "subject", "body", "message_id", "processed_at"]
        available_columns = [col for col in output_columns if col in df.columns]
        df_output = df[available_columns].copy()

        # Ajouter timestamp de traitement
        from datetime import datetime
        df_output['processed_at'] = datetime.now().isoformat()

        print(f"📝 Préparation du CSV de sortie...")
        print(f"📝 Colonnes incluses: {available_columns}")

        # Convertir en CSV
        output = io.StringIO()
        df_output.to_csv(output, index=False)
        csv_content = output.getvalue()

        print(f"✅ CSV de sortie généré: {len(csv_content)} caractères")
        print(f"📤 Envoi de la réponse au client...")

        # Préparer la réponse en streaming
        headers = {
            'Content-Disposition': 'attachment; filename="emails_live_processed.csv"',
            'Content-Type': 'text/csv'
        }

        print(f"🎉 === TRAITEMENT CSV TERMINÉ AVEC SUCCÈS ===")
        print(f"   📊 Total traité: {len(df)} emails")
        print(f"   📤 Fichier renvoyé au client\n")

        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers=headers
        )

    except pd.errors.EmptyDataError:
        print(f"❌ ERREUR: Fichier CSV vide")
        raise HTTPException(status_code=400, detail="Fichier CSV vide")
    except pd.errors.ParserError as e:
        print(f"❌ ERREUR: Parsing CSV échoué - {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erreur parsing CSV: {str(e)}")
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE: {str(e)}")
        import traceback
        print(f"❌ Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erreur traitement: {str(e)}")


# --- Charger les artefacts au démarrage ---
print("🚀 Initialisation de l'API de détection de phishing avec Smart Percentile...")
model_loaded = load_model_artifacts()

if not model_loaded:
    print("❌ ÉCHEC DU CHARGEMENT DES ARTEFACTS")
else:
    print("✅ API prête avec capacités d'adaptation smart_percentile")

if __name__ == "__main__":
    import uvicorn

    print("🚀 Démarrage de l'API sur http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)