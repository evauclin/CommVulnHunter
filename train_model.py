# Modèle LSTM/RNN pour la Détection de Phishing
# Entraînement deep learning avec TensorFlow/Keras

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Embedding, Dropout, Bidirectional,
    Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten,
    Input, Concatenate, BatchNormalization, Attention
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Sklearn pour preprocessing et métriques
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import warnings
warnings.filterwarnings('ignore')

# Configuration
tf.random.set_seed(42)
np.random.seed(42)

print("=== MODÈLE LSTM/RNN POUR DÉTECTION DE PHISHING ===\n")

class LSTMPhishingDetector:
    def __init__(self, config=None):
        self.config = config or {
            'max_vocab_size': 10000,
            'max_sequence_length': 200,
            'embedding_dim': 128,
            'lstm_units': 64,
            'dense_units': 32,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'patience': 10
        }

        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

        # NLTK setup
        try:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))

        print("Configuration du modèle:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

    def load_data(self, filepath, sample_size=None):
        """Charger et examiner les données"""
        print(f"\nChargement des données depuis {filepath}...")

        try:
            df = pd.read_csv(filepath)

            if sample_size and len(df) > sample_size:
                print(f"🔄 Échantillonnage: {sample_size} sur {len(df)} échantillons")
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

            print(f"Dataset chargé: {len(df)} échantillons")
            print(f"Colonnes: {list(df.columns)}")

            # Distribution des labels
            label_counts = df['label'].value_counts()
            print(f"\nDistribution des labels:")
            for label, count in label_counts.items():
                print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

            # Statistiques des textes
            df['text_length'] = df['text'].str.len()
            print(f"\nStatistiques des textes:")
            print(f"  Longueur moyenne: {df['text_length'].mean():.0f} caractères")
            print(f"  Longueur médiane: {df['text_length'].median():.0f} caractères")
            print(f"  Min: {df['text_length'].min()}, Max: {df['text_length'].max()}")

            return df

        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return None

    def preprocess_text(self, text):
        """Preprocessing spécialisé pour LSTM"""
        if pd.isna(text):
            return ""

        text = str(text).lower()

        # Nettoyer mais préserver la structure pour le LSTM
        # Remplacer les URLs par un token spécial
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

        # Filtrer les tokens très courts et les stop words les plus fréquents
        filtered_tokens = []
        for token in tokens:
            if len(token) > 1 and token not in ['the', 'a', 'an', 'and', 'or', 'but']:
                filtered_tokens.append(token)

        return ' '.join(filtered_tokens)

    def extract_numerical_features(self, df):
        """Extraire des features numériques pour le modèle hybride"""
        features = []

        for text in df['text']:
            if pd.isna(text):
                text = ""

            text_str = str(text)

            # Features de base
            char_count = len(text_str)
            word_count = len(text_str.split())

            # Features de ponctuation et style
            exclamation_count = text_str.count('!')
            question_count = text_str.count('?')
            upper_count = sum(1 for c in text_str if c.isupper())
            upper_ratio = upper_count / max(char_count, 1)

            # Features spécifiques phishing
            url_count = len(re.findall(r'http[s]?://', text_str))
            email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_str))

            # Mots suspects
            suspicious_words = [
                'urgent', 'immediate', 'verify', 'confirm', 'suspended', 'expired',
                'winner', 'congratulations', 'free', 'click', 'now', 'limited'
            ]
            suspicious_count = sum(1 for word in suspicious_words if word in text_str.lower())

            # Ratios
            digit_ratio = sum(1 for c in text_str if c.isdigit()) / max(char_count, 1)
            special_char_ratio = sum(1 for c in text_str if c in '!@#$%^&*()') / max(char_count, 1)

            features.append([
                char_count, word_count, exclamation_count, question_count,
                upper_ratio, url_count, email_count, suspicious_count,
                digit_ratio, special_char_ratio
            ])

        return np.array(features)

    def prepare_sequences(self, df, is_training=True):
        """Préparer les séquences pour LSTM"""
        print("Préparation des séquences...")

        # Preprocessing du texte
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        if is_training:
            # Créer et entraîner le tokenizer
            self.tokenizer = Tokenizer(
                num_words=self.config['max_vocab_size'],
                oov_token='<OOV>',
                filters='',  # Pas de filtrage supplémentaire
                lower=True
            )

            self.tokenizer.fit_on_texts(df['processed_text'])

            print(f"  Vocabulaire créé: {len(self.tokenizer.word_index)} mots uniques")
            print(f"  Vocabulaire utilisé: {min(len(self.tokenizer.word_index), self.config['max_vocab_size'])} mots")

        # Convertir en séquences
        sequences = self.tokenizer.texts_to_sequences(df['processed_text'])

        # Padding
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.config['max_sequence_length'],
            padding='post',
            truncating='post'
        )

        print(f"  Séquences créées: {padded_sequences.shape}")

        # Features numériques
        numerical_features = self.extract_numerical_features(df)

        if is_training:
            numerical_features = self.scaler.fit_transform(numerical_features)
        else:
            numerical_features = self.scaler.transform(numerical_features)

        print(f"  Features numériques: {numerical_features.shape}")

        return padded_sequences, numerical_features

    def build_lstm_model(self):
        """Construire le modèle LSTM"""
        print("Construction du modèle LSTM...")

        # Input pour les séquences de texte
        text_input = Input(shape=(self.config['max_sequence_length'],), name='text_input')

        # Embedding layer
        embedding = Embedding(
            input_dim=min(len(self.tokenizer.word_index) + 1, self.config['max_vocab_size'] + 1),
            output_dim=self.config['embedding_dim'],
            input_length=self.config['max_sequence_length'],
            mask_zero=True
        )(text_input)

        # LSTM bidirectionnel
        lstm_out = Bidirectional(LSTM(
            self.config['lstm_units'],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate']
        ))(embedding)

        # Couche d'attention simple (Global Max Pooling)
        lstm_features = GlobalMaxPooling1D()(lstm_out)

        # Input pour les features numériques
        numerical_input = Input(shape=(10,), name='numerical_input')
        numerical_dense = Dense(16, activation='relu')(numerical_input)

        # Combiner les features
        combined = Concatenate()([lstm_features, numerical_dense])

        # Couches denses
        dense1 = Dense(self.config['dense_units'], activation='relu')(combined)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.config['dropout_rate'])(dense1)

        dense2 = Dense(16, activation='relu')(dense1)
        dense2 = Dropout(self.config['dropout_rate'])(dense2)

        # Sortie
        output = Dense(1, activation='sigmoid', name='output')(dense2)

        # Créer le modèle
        model = Model(inputs=[text_input, numerical_input], outputs=output)

        # Compiler
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        print("Architecture du modèle:")
        model.summary()

        self.model = model
        return model

    def save_model_artifacts(self, model_name="best_lstm_model"):
        """Sauvegarder le modèle et tous les artefacts pour l'API"""
        print(f"\nSAUVEGARDE DES ARTEFACTS POUR L'API")
        print("=" * 50)

        import pickle

        # Sauvegarder le modèle
        model_path = f"{model_name}.keras"
        self.model.save(model_path)
        print(f"✅ Modèle sauvegardé: {model_path}")

        # Sauvegarder le tokenizer
        tokenizer_path = "tokenizer.pkl"
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"✅ Tokenizer sauvegardé: {tokenizer_path}")

        # Sauvegarder le scaler
        scaler_path = "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler sauvegardé: {scaler_path}")

        # Sauvegarder le label encoder
        label_encoder_path = "label_encoder.pkl"
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✅ Label encoder sauvegardé: {label_encoder_path}")

        # Créer un fichier de métadonnées
        metadata = {
            'model_type': 'LSTM',
            'model_file': model_path,
            'tokenizer_file': tokenizer_path,
            'scaler_file': scaler_path,
            'label_encoder_file': label_encoder_path,
            'config': self.config,
            'vocabulary_size': len(self.tokenizer.word_index),
            'classes': list(self.label_encoder.classes_),
            'creation_date': pd.Timestamp.now().isoformat()
        }

        metadata_path = "model_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f" Métadonnées sauvegardées: {metadata_path}")

        print(f"\n ARTEFACTS PRÊTS POUR L'API:")
        print(f"   {model_path}")
        print(f"   {tokenizer_path}")
        print(f"   {scaler_path}")
        print(f"   {label_encoder_path}")
        print(f"   {metadata_path}")
        print(f"\n️  Copiez ces fichiers dans votre dossier Docker!")

        return {
            'model_path': model_path,
            'tokenizer_path': tokenizer_path,
            'scaler_path': scaler_path,
            'label_encoder_path': label_encoder_path,
            'metadata_path': metadata_path
        }

    def build_gru_model(self):
        """Alternative avec GRU (plus rapide que LSTM)"""
        print(" Construction du modèle GRU...")

        text_input = Input(shape=(self.config['max_sequence_length'],), name='text_input')

        embedding = Embedding(
            input_dim=min(len(self.tokenizer.word_index) + 1, self.config['max_vocab_size'] + 1),
            output_dim=self.config['embedding_dim'],
            input_length=self.config['max_sequence_length'],
            mask_zero=True
        )(text_input)

        # GRU bidirectionnel
        gru_out = Bidirectional(GRU(
            self.config['lstm_units'],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate']
        ))(embedding)

        gru_features = GlobalMaxPooling1D()(gru_out)

        # Features numériques
        numerical_input = Input(shape=(10,), name='numerical_input')
        numerical_dense = Dense(16, activation='relu')(numerical_input)

        # Combinaison
        combined = Concatenate()([gru_features, numerical_dense])

        dense1 = Dense(self.config['dense_units'], activation='relu')(combined)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.config['dropout_rate'])(dense1)

        output = Dense(1, activation='sigmoid')(dense1)

        model = Model(inputs=[text_input, numerical_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        self.model = model
        return model

    def build_cnn_lstm_model(self):
        """Modèle hybride CNN-LSTM"""
        print("Construction du modèle CNN-LSTM...")

        text_input = Input(shape=(self.config['max_sequence_length'],), name='text_input')

        embedding = Embedding(
            input_dim=min(len(self.tokenizer.word_index) + 1, self.config['max_vocab_size'] + 1),
            output_dim=self.config['embedding_dim'],
            input_length=self.config['max_sequence_length']
        )(text_input)

        # Couches CNN pour capturer les patterns locaux
        conv1 = Conv1D(64, 3, activation='relu')(embedding)
        conv1 = MaxPooling1D(2)(conv1)

        conv2 = Conv1D(32, 3, activation='relu')(conv1)
        conv2 = MaxPooling1D(2)(conv2)

        # LSTM pour les dépendances temporelles
        lstm_out = LSTM(
            self.config['lstm_units'],
            return_sequences=False,
            dropout=self.config['dropout_rate']
        )(conv2)

        # Features numériques
        numerical_input = Input(shape=(10,), name='numerical_input')
        numerical_dense = Dense(16, activation='relu')(numerical_input)

        # Combinaison
        combined = Concatenate()([lstm_out, numerical_dense])

        dense1 = Dense(self.config['dense_units'], activation='relu')(combined)
        dense1 = Dropout(self.config['dropout_rate'])(dense1)

        output = Dense(1, activation='sigmoid')(dense1)

        model = Model(inputs=[text_input, numerical_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        self.model = model
        return model

    def train_model(self, X_text_train, X_num_train, y_train,
                   X_text_val, X_num_val, y_val):
        """Entraîner le modèle LSTM uniquement"""
        print("Entraînement du modèle LSTM...")

        # Encoder les labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)

        # Construire le modèle LSTM
        self.build_lstm_model()

        # Callbacks
        callbacks = [
            EarlyStopping(
                patience=self.config['patience'],
                restore_best_weights=True,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                monitor='val_loss'
            ),
            ModelCheckpoint(
                'best_lstm_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        # Entraînement
        print("Début de l'entraînement...")
        self.history = self.model.fit(
            [X_text_train, X_num_train], y_train_encoded,
            validation_data=([X_text_val, X_num_val], y_val_encoded),
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            callbacks=callbacks,
            verbose=1
        )

        print(" Entraînement terminé!")

        # Sauvegarder automatiquement tous les artefacts
        self.save_model_artifacts("best_lstm_model")

        return self.history

    def evaluate_model(self, X_text_test, X_num_test, y_test):
        """Évaluer le modèle"""
        print("\nÉVALUATION DU MODÈLE")
        print("=" * 40)

        # Prédictions
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_proba = self.model.predict([X_text_test, X_num_test])
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Métriques
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred)
        recall = recall_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred)
        auc = roc_auc_score(y_test_encoded, y_pred_proba)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")

        # Matrice de confusion
        cm = confusion_matrix(y_test_encoded, y_pred)
        labels = self.label_encoder.classes_

        print(f"\nMatrice de confusion:")
        print(f"    Predicted:  {labels[0]:<10} {labels[1]:<10}")
        print(f"Actual:")
        for i, label in enumerate(labels):
            print(f"{label:<10}  {cm[i,0]:<10} {cm[i,1]:<10}")

        # Rapport détaillé
        print(f"\n Rapport de classification:")
        report = classification_report(y_test_encoded, y_pred, target_names=labels, digits=4)
        print(report)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def plot_training_history(self):
        """Visualiser l'historique d'entraînement"""
        if not self.history:
            print("Pas d'historique d'entraînement disponible")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0,0].plot(self.history.history['loss'], label='Train Loss')
        axes[0,0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0,0].set_title('Model Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()

        # Accuracy
        axes[0,1].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0,1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0,1].set_title('Model Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()

        # Precision
        axes[1,0].plot(self.history.history['precision'], label='Train Precision')
        axes[1,0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1,0].set_title('Model Precision')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].legend()

        # Recall
        axes[1,1].plot(self.history.history['recall'], label='Train Recall')
        axes[1,1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1,1].set_title('Model Recall')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].legend()

        plt.tight_layout()
        plt.show()

    def predict_new_texts(self, texts):
        """Prédire sur de nouveaux textes"""
        if not self.model:
            print("Modèle non entraîné")
            return None

        # Créer DataFrame temporaire
        temp_df = pd.DataFrame({'text': texts})

        # Préparer les données
        X_text, X_num = self.prepare_sequences(temp_df, is_training=False)

        # Prédictions
        probabilities = self.model.predict([X_text, X_num])
        predictions = (probabilities > 0.5).astype(int)

        # Décoder
        predictions_decoded = self.label_encoder.inverse_transform(predictions.flatten())

        return predictions_decoded, probabilities.flatten()

# Fonction principale simplifiée pour LSTM uniquement
def main():
    # Configuration personnalisée
    config = {
        'max_vocab_size': 10000,
        'max_sequence_length': 150,
        'embedding_dim': 100,
        'lstm_units': 64,
        'dense_units': 32,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 2046,
        'epochs': 1,
        'patience': 8
    }

    # Initialiser le détecteur
    detector = LSTMPhishingDetector(config)

    # Charger les données (échantillon pour la démo)
    df = detector.load_data('full_merged_dataset_benign_phishing.csv', sample_size=15000)

    if df is None:
        print("Impossible de charger les données")
        return

    # Diviser les données
    print("\n📋 Division des données...")
    X = df['text']
    y = df['label']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train)} échantillons")
    print(f"  Validation: {len(X_val)} échantillons")
    print(f"  Test: {len(X_test)} échantillons")

    # Préparer les données
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    val_df = pd.DataFrame({'text': X_val, 'label': y_val})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})

    X_text_train, X_num_train = detector.prepare_sequences(train_df, is_training=True)
    X_text_val, X_num_val = detector.prepare_sequences(val_df, is_training=False)
    X_text_test, X_num_test = detector.prepare_sequences(test_df, is_training=False)

    # Entraîner le modèle LSTM
    print(f"\n{'='*60}")
    print("ENTRAÎNEMENT DU MODÈLE LSTM")
    print(f"{'='*60}")

    # Entraîner
    history = detector.train_model(
        X_text_train, X_num_train, y_train,
        X_text_val, X_num_val, y_val
    )

    # Évaluer
    metrics = detector.evaluate_model(X_text_test, X_num_test, y_test)

    # Visualiser l'entraînement
    print("\nVisualisation de l'entraînement:")
    detector.plot_training_history()

    # Résultats finaux
    print(f"\n{'='*60}")
    print("RÉSULTATS FINAUX")
    print(f"{'='*60}")

    print(f"Métriques de performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")

    # Test sur de nouveaux exemples
    print(f"\n EXEMPLES DE PRÉDICTIONS")
    print("=" * 50)

    sample_texts = [
        "URGENT: Your account will be suspended in 24 hours. Click here to verify your identity immediately and avoid account closure.",
        "Hi Sarah, thanks for sending the quarterly report. Could we schedule a meeting next week to discuss the budget allocation?",
        "Congratulations! You've won $10,000 in our lottery. Send your bank details to claim your prize now!",
        "Reminder: Your meeting with the development team is scheduled for tomorrow at 2 PM in conference room B."
    ]

    predictions, probabilities = detector.predict_new_texts(sample_texts)

    for i, (text, pred, prob) in enumerate(zip(sample_texts, predictions, probabilities)):
        print(f"\nTexte {i+1}: {text[:80]}...")
        print(f"Prédiction: {pred}")
        print(f"Probabilité phishing: {prob:.4f}")
        print(f"Confiance: {'PHISHING' if prob > 0.7 else 'SUSPECT' if prob > 0.3 else 'LÉGITIME'}")

    print(f"\n ENTRAÎNEMENT TERMINÉ!")
    print("Fichiers créés pour l'API Docker:")
    print("  - best_lstm_model.h5")
    print("  - tokenizer.pkl")
    print("  - scaler.pkl")
    print("  - label_encoder.pkl")
    print("  - model_metadata.json")

    return detector

if __name__ == "__main__":
    main()