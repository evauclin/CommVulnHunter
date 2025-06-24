# Évaluateur Simple de Modèle
import pandas as pd
import numpy as np
import pickle
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def simple_evaluate_model(model_path, metadata_path, test_dataset_path):


    print("=== ÉVALUATION SIMPLE DU MODÈLE ===")

    print(f"Chargement des métadonnées: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    config = metadata['config']

    print(f"Chargement du modèle: {model_path}")
    model = load_model(model_path)

    print(f"Chargement du tokenizer: {metadata['tokenizer_file']}")
    with open("model/tokenizer.pkl", 'rb') as f:
        tokenizer = pickle.load(f)

    print(f"Chargement du scaler: {metadata['scaler_file']}")
    with open("model/scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)

    print(f"Chargement du label encoder: {metadata['label_encoder_file']}")
    with open("model/label_encoder.pkl", 'rb') as f:
        label_encoder = pickle.load(f)

    print(f"Chargement du dataset de test: {test_dataset_path}")
    test_df = pd.read_csv(test_dataset_path)
    print(f"   {len(test_df)} échantillons")

    print("Préprocessing...")

    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http[s]?://\S+', ' URL_TOKEN ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL_TOKEN ', text)
        text = re.sub(r'\b\d+\b', ' NUM_TOKEN ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = [token for token in text.split() if len(token) > 2]
        return ' '.join(tokens)

    def extract_numerical_features(df):
        features = []
        for text in df['text']:
            if pd.isna(text):
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
            digit_ratio = sum(1 for c in text_str if c.isdigit()) / max(char_count, 1)
            special_char_ratio = sum(1 for c in text_str if c in '!@#$%^&*()') / max(char_count, 1)

            features.append([
                char_count, word_count, exclamation_count, question_count,
                upper_ratio, url_count, email_count, 0,
                digit_ratio, special_char_ratio
            ])
        return np.array(features)

    test_df['processed_text'] = test_df['text'].apply(preprocess_text)

    sequences = tokenizer.texts_to_sequences(test_df['processed_text'])
    X_text = pad_sequences(sequences, maxlen=config['max_sequence_length'], padding='post')

    X_num = extract_numerical_features(test_df)
    X_num = scaler.transform(X_num)

    y_true = label_encoder.transform(test_df['label'])

    print("Génération des prédictions...")
    y_pred_proba = model.predict([X_text, X_num], verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_pred_proba = y_pred_proba.flatten()

    print("Calcul des métriques...")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    results = {
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1-Score': float(f1),
        'AUC': float(auc)
    }

    for key in results:
        results[key] = round(results[key], 4)

    output_file = 'evaluation_results.py'
    with open(f"model/{output_file}", 'w') as f:
        f.write("evaluation_results = {\n")
        for key, value in results.items():
            f.write(f"    '{key}': {value},\n")
        f.write("}\n")
    print(f"💾 Résultats sauvegardés: {output_file}")

    print("\n✅ RÉSULTATS D'ÉVALUATION:")
    print("=" * 30)
    for key, value in results.items():
        print(f"{key}: {value}")

    return results


if __name__ == "__main__":
    results = simple_evaluate_model(
        model_path='model/best_lstm_model.keras',
        metadata_path='model/model_metadata.json',
        test_dataset_path='test_dataset.csv',
    )