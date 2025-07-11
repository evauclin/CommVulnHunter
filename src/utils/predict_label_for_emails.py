import pandas as pd
import numpy as np
import pickle
import json
import re
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langdetect import detect
import nltk
import sys
import os

# Ensure NLTK stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Constants
DEFAULT_MODEL_PATH = Path('../../app/model')
MAX_SEQUENCE_LENGTH = 566


def load_artifacts(model_path=DEFAULT_MODEL_PATH):
    print("Chargement des artefacts du modèle...")

    # Load metadata
    metadata_file = model_path / "model_metadata.json"
    max_seq_len = MAX_SEQUENCE_LENGTH
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        max_seq_len = metadata.get('config', {}).get('max_sequence_length', max_seq_len)

    # Load model and artifacts
    model = load_model(model_path / 'best_lstm_model.keras')
    tokenizer = pickle.load(open(model_path / 'tokenizer.pkl', 'rb'))
    scaler = pickle.load(open(model_path / 'scaler.pkl', 'rb'))
    label_encoder = pickle.load(open(model_path / 'label_encoder.pkl', 'rb'))

    suspicious_words_set = set()
    suspicious_words_file = model_path / "suspicious_words.json"
    if suspicious_words_file.exists():
        with open(suspicious_words_file, 'r') as f:
            data = json.load(f)
        suspicious_words_set = set(data.get('en', []) + data.get('fr', []))

    print("Artefacts chargés.")
    return model, tokenizer, scaler, label_encoder, max_seq_len, suspicious_words_set


STOP_WORDS = {
    'en': set(nltk.corpus.stopwords.words('english')),
    'fr': set(nltk.corpus.stopwords.words('french'))
}


def preprocess_text(text, language='en'):
    if pd.isna(text) or not text:
        return ""

    text = str(text).lower()
    text = re.sub(r'http[s]?://\S+', ' URL_TOKEN ', text)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' EMAIL_TOKEN ', text)
    text = re.sub(r'\b\d+\b', ' NUM_TOKEN ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    stop_words = STOP_WORDS.get(language, set())
    filtered = [t for t in tokens if len(t) > 2 and t not in stop_words]
    return ' '.join(filtered)


def extract_numerical_features(text, suspicious_words):
    if pd.isna(text) or not text:
        return [0.0] * 10

    text = str(text)
    length = len(text)
    features = [
        length,
        len(text.split()),
        text.count('!'),
        text.count('?'),
        sum(1 for c in text if c.isupper()) / max(length, 1),
        len(re.findall(r'http[s]?://', text)),
        len(re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)),
        sum(1 for word in suspicious_words if word in text.lower()),
        sum(1 for c in text if c.isdigit()) / max(length, 1),
        sum(1 for c in text if c in '!@#$%^&*()') / max(length, 1),
    ]
    return features


def detect_language(text):
    try:
        return detect(text[:1000]) if detect(text[:1000]) in ['fr', 'en'] else 'en'
    except:
        return 'en'


def map_prediction_to_label(pred, prob, confidence):
    return 'SPAM' if pred == 'phishing' else 'IMPORTANT'


def predict_batch(texts, original_types, model, tokenizer, scaler, label_encoder, max_seq_len, suspicious_words):
    results = []

    for i, text in enumerate(texts):
        original_type = original_types[i] if original_types else None

        if pd.isna(text) or not str(text).strip():
            results.append({
                'new_type': original_type or 'IMPORTANT',
                'prediction': 'unknown',
                'probability': 0.0,
                'confidence': 'LOW',
                'language': 'en',
                'original_type': original_type
            })
            continue

        lang = detect_language(text)
        cleaned = preprocess_text(text, lang)

        sequence = tokenizer.texts_to_sequences([cleaned])
        sequence[0] = [token for token in sequence[0] if token <= len(tokenizer.word_index)]
        padded = pad_sequences(sequence, maxlen=max_seq_len, padding='post', truncating='post')

        features = extract_numerical_features(text, suspicious_words)
        scaled = scaler.transform([features])

        try:
            proba = model.predict([padded, scaled], verbose=0)[0][0]
            pred_class = label_encoder.inverse_transform([int(proba > 0.5)])[0]
            confidence_score = abs(proba - 0.5) * 2
            confidence = "HIGH" if confidence_score > 0.8 else "MEDIUM" if confidence_score > 0.4 else "LOW"
            mapped_label = map_prediction_to_label(pred_class, proba, confidence)

            results.append({
                'new_type': mapped_label,
                'prediction': pred_class,
                'probability': float(proba),
                'confidence': confidence,
                'language': lang,
                'original_type': original_type
            })
        except Exception as e:
            print(f"Erreur de prédiction pour l'email {i + 1}: {e}")
            results.append({
                'new_type': original_type or 'IMPORTANT',
                'prediction': 'error',
                'probability': 0.0,
                'confidence': 'LOW',
                'language': lang,
                'original_type': original_type
            })

    return results


def main(csv_path='../pages/emails_live.csv', model_path=DEFAULT_MODEL_PATH):
    print("Chargement du CSV...")
    if not Path(csv_path).exists():
        print(f"Erreur: Fichier introuvable: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if 'body' not in df.columns or 'type' not in df.columns:
        print("Erreur: Le fichier CSV doit contenir les colonnes 'body' et 'type'.")
        return

    original_types_backup = df['type'].copy()
    print(df['type'].value_counts())

    model, tokenizer, scaler, label_encoder, max_seq_len, suspicious_words = load_artifacts(model_path)

    texts = df['body'].tolist()
    original_types = df['type'].tolist()

    batch_size = 100
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch_results = predict_batch(
            texts[i:i + batch_size],
            original_types[i:i + batch_size],
            model, tokenizer, scaler, label_encoder, max_seq_len, suspicious_words
        )
        all_results.extend(batch_results)
        print(f"Traité: {min(i + batch_size, len(texts))}/{len(texts)}")

    df_results = pd.DataFrame(all_results)
    df['type'] = df_results['new_type']

    output_file = csv_path
    columns_to_keep = ["id", "type", "from", "to", "date", "subject", "body", "message_id", "processed_at"]
    df[columns_to_keep].to_csv(output_file, index=False)

    print(f"\nRésultats sauvegardés dans: {output_file}")
    print(f"Résumé: {(original_types_backup != df['type']).sum()}/{len(df)} labels modifiés")


if __name__ == "__main__":
    main()
