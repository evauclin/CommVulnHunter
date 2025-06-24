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



# --- Configuration et Initialisation ---
app = FastAPI(
    title="API de Détection de Phishing Automatique (FR/EN)",
    description="Une API pour classifier des textes en détectant automatiquement la langue.",
    version="3.0.1"
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
MAX_SEQUENCE_LENGTH = 566  # Valeur par défaut - sera remplacée par les métadonnées
SUSPICIOUS_WORDS_SET = set()
STOP_WORDS = {}
# Ajouter ces variables après les autres variables globales
AUTO_FINETUNING_ENABLED = True
IS_FINETUNING_RUNNING = False
FINETUNING_LOCK = threading.Lock()
NEGATIVE_FEEDBACK_THRESHOLD = 5
tf.config.set_visible_devices([], 'GPU')

FEEDBACK_CSV_PATH = Path("./data/user_feedbacks.csv")


# --- Chargement des Artefacts du Modèle ---
def load_model_artifacts():
    """Charge tous les artefacts du modèle avec la correction de la longueur de séquence"""
    global model, tokenizer, scaler, label_encoder, MAX_SEQUENCE_LENGTH, SUSPICIOUS_WORDS_SET, STOP_WORDS

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
                print(f"✅ Métadonnées chargées: max_sequence_length = {MAX_SEQUENCE_LENGTH}")

                # Afficher la configuration complète du modèle
                print(f"📋 Configuration du modèle:")
                print(f"  max_vocab_size: {config.get('max_vocab_size', 'Non défini')}")
                print(f"  max_sequence_length: {MAX_SEQUENCE_LENGTH}")
                print(f"  embedding_dim: {config.get('embedding_dim', 'Non défini')}")
                print(f"  lstm_units: {config.get('lstm_units', 'Non défini')}")

            except Exception as e:
                print(f"⚠️ Erreur chargement métadonnées: {e}")
                print(f"⚠️ Utilisation de la valeur par défaut: MAX_SEQUENCE_LENGTH = {MAX_SEQUENCE_LENGTH}")
        else:
            print(f"⚠️ Fichier model_metadata.json non trouvé")
            print(f"⚠️ Utilisation de la valeur par défaut: MAX_SEQUENCE_LENGTH = {MAX_SEQUENCE_LENGTH}")

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
                if expected_seq_length is not None and expected_seq_length != MAX_SEQUENCE_LENGTH:
                    print(f"⚠️ ATTENTION: Incohérence détectée!")
                    print(f"  Modèle attend: {expected_seq_length}")
                    print(f"  Métadonnées indiquent: {MAX_SEQUENCE_LENGTH}")
                    print(f"  🔧 Correction: utilisation de {expected_seq_length}")
                    MAX_SEQUENCE_LENGTH = expected_seq_length

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
        else:
            print("⚠️ Fichier suspicious_words.json manquant, utilisation d'une liste vide")

        # ÉTAPE 8: Charger les stopwords NLTK
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

        # ÉTAPE 9: Test de prédiction pour vérifier le fonctionnement
        print(f"\n🧪 Test de validation du modèle...")
        try:
            test_text = "Test email content"
            test_processed = preprocess_text(test_text, 'en')
            test_sequence = tokenizer.texts_to_sequences([test_processed])
            test_padded = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
            test_features = extract_numerical_features(test_text)
            test_scaled = scaler.transform([test_features])

            # Test de prédiction
            test_pred = model.predict([test_padded, test_scaled], verbose=0)
            print(f"✅ Test de prédiction réussi: {test_pred[0][0]:.4f}")

        except Exception as e:
            print(f"❌ Échec du test de validation: {e}")
            return False

        print(f"\n🎉 API prête ! Configuration finale:")
        print(f"  Longueur de séquence: {MAX_SEQUENCE_LENGTH}")
        print(f"  Taille du vocabulaire: {len(tokenizer.word_index)}")
        print(f"  Classes: {list(label_encoder.classes_)}")
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
                print(f"  - {file.name} ({file.stat().st_size} bytes)")
        else:
            print("❌ Dossier 'model' n'existe pas")

        return False


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
    """Fonction cœur qui détecte la langue et effectue une prédiction avec la bonne longueur de séquence."""
    if not model:
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
        # Filtrer les tokens qui dépassent la taille du vocabulaire du modèle
        if sequence[0]:  # Si la séquence n'est pas vide
            max_vocab_id = 10000  # indices valides: 0 à 10000
            sequence[0] = [token_id for token_id in sequence[0] if token_id <= max_vocab_id]
            print(
                f"🔧 Tokens filtrés: {len([t for t in tokenizer.texts_to_sequences([processed_text])[0] if t > max_vocab_id])} tokens supprimés")
        print(f"🔢 Séquence créée: longueur = {len(sequence[0]) if sequence[0] else 0}")

        # Utiliser la bonne longueur de séquence
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


def check_finetuning_trigger():
    """
    Vérifie si les conditions pour déclencher le fine-tuning sont remplies
    """
    try:
        negative_count = count_negative_feedbacks()

        # Seuil pour déclencher le fine-tuning (5 feedbacks négatifs)
        NEGATIVE_FEEDBACK_THRESHOLD = 5

        if negative_count >= NEGATIVE_FEEDBACK_THRESHOLD:
            print(f"🚨 Seuil de fine-tuning atteint: {negative_count}/{NEGATIVE_FEEDBACK_THRESHOLD} feedbacks négatifs")
            print("💡 Vous pouvez maintenant exécuter le fine-tuning avec: python traitement.py")
            return True
        else:
            print(f"📊 Feedbacks négatifs: {negative_count}/{NEGATIVE_FEEDBACK_THRESHOLD}")
            return False

    except Exception as e:
        print(f"❌ Erreur vérification fine-tuning: {e}")
        return False


# --- FONCTIONS DE FINE-TUNING AUTOMATIQUE (À AJOUTER DANS VOTRE MAIN.PY) ---

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

# Dans main.py, modifiez cette fonction
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

# --- Endpoints de l'API ---
@app.get("/", summary="Message de bienvenue")
def read_root():
    return {
        "message": "Bienvenue sur l'API de détection de phishing (LSTM Hybride FR/EN)",
        "version": app.version,
        "documentation": "/docs",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "model_loaded": model is not None
    }


@app.get("/health", summary="Vérification de l'état de l'API")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded")

    negative_feedbacks = count_negative_feedbacks()
    finetuning_ready = check_finetuning_trigger()

    return {
        "status": "healthy",
        "model_loaded": True,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "vocab_size": len(tokenizer.word_index) if tokenizer else 0,
        "negative_feedbacks": negative_feedbacks,
        "finetuning_ready": finetuning_ready,
        "model_classes": list(label_encoder.classes_) if label_encoder else []
    }


@app.post("/predict", summary="Prédire sur un seul texte")
def predict(item: TextInput):
    """
    Analyse un texte, détecte sa langue (fr/en) et prédit s'il s'agit d'un phishing.
    """
    if not model:
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


# --- REMPLACEZ VOTRE FONCTION save_feedback PAR CELLE-CI ---

@app.post("/feedback", summary="Enregistrer un feedback utilisateur")
async def save_feedback(feedback: FeedbackInput, background_tasks: BackgroundTasks):
    """
    Enregistre le feedback utilisateur et déclenche automatiquement le fine-tuning si nécessaire
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


@app.get("/feedbacks/stats", summary="Statistiques des feedbacks")
def get_feedback_stats():
    """
    Récupère les statistiques des feedbacks pour le monitoring
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

        finetuning_ready = negative_unprocessed >= 5

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


@app.get("/debug/model-info", summary="Informations de diagnostic du modèle")
def get_model_info():
    """Endpoint pour diagnostiquer les problèmes de modèle"""
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
@app.post("/trigger-finetuning", summary="Déclenche le fine-tuning (développement uniquement)")
def trigger_finetuning():
    """
    Endpoint pour vérifier si le fine-tuning peut être déclenché
    Note: Le fine-tuning réel doit être exécuté via 'python traitement.py'
    """
    try:
        negative_count = count_negative_feedbacks()

        if negative_count >= 5:
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
                "needed": 5 - negative_count
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {e}")


# --- Charger les artefacts au démarrage ---
print("🚀 Initialisation de l'API de détection de phishing...")
model_loaded = load_model_artifacts()

if not model_loaded:
    print("❌ ÉCHEC DU CHARGEMENT DES ARTEFACTS")
    print("❌ L'API ne pourra pas traiter les prédictions")
else:
    print("✅ API prête à traiter les requêtes de prédiction")

# --- Lancement de l'application ---
if __name__ == "__main__":
    import uvicorn

    print("🚀 Démarrage de l'API sur http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)