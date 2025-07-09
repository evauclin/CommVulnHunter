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


# --- ENDPOINTS ---

@app.get("/health", summary="Health check")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded")

    negative_feedbacks = 0
    finetuning_ready = False

    return {
        "status": "healthy",
        "model_loaded": True,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "vocab_size": len(tokenizer.word_index) if tokenizer else 0,
        "negative_feedbacks": negative_feedbacks,
        "finetuning_ready": finetuning_ready,
        "model_classes": list(label_encoder.classes_) if label_encoder else []
    }


@app.post("/predict", summary="Predict on single text (standard mode)")
def predict(item: TextInput):
    """Analyze text with standard method (compatibility mode)"""
    print(f"🔮 ENDPOINT /predict APPELÉ avec texte de {len(item.text)} chars")
    
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
        
        print(f"🔍 EMAIL ID {item.email_id} - Longueur: {len(raw_text)} chars")
        
        # Prédiction avec les données brutes
        result = perform_prediction(raw_text)
        
        # Ajouter l'ID de l'email dans la réponse
        result['email_id'] = item.email_id
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {e}")




@app.post("/predict/batch", summary="Predict on text list")
def predict_batch(batch: BatchInput):
    """Analyze a list of texts"""
    results = []
    for item in batch.items:
        try:
            result = perform_prediction(item.text)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "text": item.text[:50] + "..."})
    return {"results": results}








@app.post("/feedback", summary="Save user feedback")
async def save_feedback(feedback: FeedbackInput, background_tasks: BackgroundTasks):
    """Save user feedback"""
    try:
        print(f"📝 Feedback reçu: {feedback.user_satisfaction}")
        
        return {
            "status": "success",
            "message": "Feedback enregistré avec succès",
            "feedback_type": feedback.user_satisfaction
        }

    except Exception as e:
        print(f"❌ Erreur feedback: {e}")
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


@app.post("/process-csv", summary="Process CSV file with ML predictions")
async def process_csv_file(file: UploadFile = File(...)):
    """
    Process email CSV file with ML model and return predictions
    """
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    try:
        print(f"\n🔥 TRAITEMENT CSV: {file.filename}")

        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Le fichier doit être un CSV")

        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        print(f"📊 CSV parsé: {len(df)} lignes")

        required_columns = ['from', 'subject', 'body', 'type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes: {missing_columns}"
            )

        distribution_avant = df['type'].value_counts()
        print(f"📊 Distribution initiale: {dict(distribution_avant)}")

        batch_size = 50
        all_results = []
        total_batches = (len(df) + batch_size - 1) // batch_size

        print(f"🔄 Traitement par batch: {total_batches} batches")

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_results = []
            batch_num = i // batch_size + 1

            print(f"🚀 BATCH {batch_num}/{total_batches} ({len(batch_df)} emails)")

            # Traiter chaque email du batch
            for j, row in batch_df.iterrows():
                email_index = i + (j - batch_df.index[0]) + 1
                try:
                    # Utiliser les données BRUTES comme predict_by_email_id
                    from_field = str(row['from']) if not pd.isna(row['from']) else ""
                    subject_field = str(row['subject']) if not pd.isna(row['subject']) else ""
                    body_field = str(row['body']) if not pd.isna(row['body']) else ""
                    
                    combined_text = f"From: {from_field}\nSubject: {subject_field}\nBody: {body_field}".strip()

                    if not combined_text or combined_text == "From: \nSubject: \nBody: ":
                        original_type = row['type']
                        batch_results.append({
                            'new_type': original_type,
                            'prediction': 'unknown',
                            'probability': 0.0,
                            'confidence': 'LOW'
                        })
                        continue

                    result = perform_prediction(combined_text)
                    new_type = 'SPAM' if result['prediction'] == 'phishing' else 'IMPORTANT'
                    old_type = row['type']

                    if new_type != old_type:
                        print(f"   🔄 Email {email_index}: {old_type} → {new_type}")

                    batch_results.append({
                        'new_type': new_type,
                        'prediction': result['prediction'],
                        'probability': result['probability'],
                        'confidence': result['confidence']
                    })

                except Exception as e:
                    print(f"   ❌ Email {email_index}: ERREUR - {str(e)}")
                    original_type = row['type']
                    batch_results.append({
                        'new_type': original_type,
                        'prediction': 'error',
                        'probability': 0.0,
                        'confidence': 'LOW'
                    })

            all_results.extend(batch_results)
            print(f"✅ Batch {batch_num} terminé")

        results_df = pd.DataFrame(all_results)
        df['type'] = results_df['new_type']

        distribution_apres = df['type'].value_counts()
        print(f"📈 Distribution finale: {dict(distribution_apres)}")

        if len(distribution_avant) > 0 and len(distribution_apres) > 0:
            spam_avant = distribution_avant.get('SPAM', 0)
            spam_apres = distribution_apres.get('SPAM', 0)
            important_avant = distribution_avant.get('IMPORTANT', 0)
            important_apres = distribution_apres.get('IMPORTANT', 0)

            print(f"📊 SPAM: {spam_avant} → {spam_apres} ({spam_apres - spam_avant:+d})")
            print(f"📊 IMPORTANT: {important_avant} → {important_apres} ({important_apres - important_avant:+d})")

        output_columns = ["id", "type", "from", "to", "date", "subject", "body", "message_id", "processed_at"]
        available_columns = [col for col in output_columns if col in df.columns]
        df_output = df[available_columns].copy()

        from datetime import datetime
        df_output['processed_at'] = datetime.now().isoformat()

        output = io.StringIO()
        df_output.to_csv(output, index=False)
        csv_content = output.getvalue()

        headers = {
            'Content-Disposition': 'attachment; filename="emails_live_processed.csv"',
            'Content-Type': 'text/csv'
        }

        print(f"🎉 TRAITEMENT TERMINÉ: {len(df)} emails traités")

        return StreamingResponse(
            io.BytesIO(csv_content.encode('utf-8')),
            media_type="text/csv",
            headers=headers
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Fichier CSV vide")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Erreur parsing CSV: {str(e)}")
    except Exception as e:
        print(f"❌ ERREUR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur traitement: {str(e)}")


# --- Charger les artefacts au démarrage ---
print("🚀 Initialisation de l'API de détection de phishing...")
model_loaded = load_model_artifacts()

if not model_loaded:
    print("❌ ÉCHEC DU CHARGEMENT DES ARTEFACTS")
else:
    print("✅ API prête")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)