import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle

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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer

import warnings
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)

print("=== MODÈLE LSTM/RNN POUR DÉTECTION DE PHISHING (FR/EN) ===\n")

class LSTMPhishingDetector:
    def __init__(self, config=None):
        self.config = config or {}

        default_config = {
            'embedding_dim': 128,
            'lstm_units': 64,
            'dense_units': 32,
            'dropout_rate': 0.4,
            'learning_rate': 0.001,
            'batch_size': 128,
            'epochs': 30,
            'patience': 5,
            'suspicious_words_path': 'suspicious_words.json',
            'vocab_coverage': 0.95,
            'sequence_percentile': 95,
            'min_word_frequency': 2
        }

        self.config = {**default_config, **(config or {})}
        self.config['max_vocab_size'] = None
        self.config['max_sequence_length'] = None
        self.num_numerical_features = None

        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

        try:
            stopwords.words('english')
            stopwords.words('french')
        except LookupError:
            print("Downloading NLTK data (stopwords for English and French)...")
            nltk.download('punkt')
            nltk.download('stopwords')

        self.stop_words = {
            'en': set(stopwords.words('english')),
            'fr': set(stopwords.words('french'))
        }
        print("Stopwords pour 'en' et 'fr' chargés.")

        print("\nConfiguration du modèle:")
        for key, value in self.config.items():
            if value is not None:
                print(f"  {key}: {value}")

    def load_data(self, filepath, sample_size=None):
        print(f"\nChargement des données depuis {filepath}...")

        try:
            df = pd.read_csv(filepath)

            if sample_size and len(df) > sample_size:
                print(f"🔄 Échantillonnage: {sample_size} sur {len(df)} échantillons")
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

            print(f"Dataset chargé: {len(df)} échantillons")
            print(f"Colonnes: {list(df.columns)}")

            if 'language' in df.columns:
                lang_counts = df['language'].value_counts()
                print(f"\nDistribution des langues:")
                for lang, count in lang_counts.items():
                    print(f"  {lang}: {count} ({count/len(df)*100:.1f}%)")
            else:
                print("\n⚠️ Colonne 'language' non trouvée. Le prétraitement utilisera l'anglais par défaut.")
                df['language'] = 'en'

            label_counts = df['label'].value_counts()
            print(f"\nDistribution des labels:")
            for label, count in label_counts.items():
                print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

            df['text_length'] = df['text'].str.len()
            print(f"\nStatistiques des textes:")
            print(f"  Longueur moyenne: {df['text_length'].mean():.0f} caractères")
            print(f"  Longueur médiane: {df['text_length'].median():.0f} caractères")
            print(f"  Min: {df['text_length'].min()}, Max: {df['text_length'].max()}")

            return df

        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return None

    def preprocess_text(self, row):
        text = row['text']
        language = row.get('language', 'en')

        if pd.isna(text):
            return ""

        text = str(text).lower()

        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL_TOKEN ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL_TOKEN ', text)
        text = re.sub(r'\b\d+\b', ' NUM_TOKEN ', text)

        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()

        stop_words_lang = self.stop_words.get(language, self.stop_words['en'])
        filtered_tokens = [token for token in tokens if len(token) > 2 and token not in stop_words_lang]

        return ' '.join(filtered_tokens)

    def calculate_vocab_size(self, texts):
        print("\n📊 Calcul automatique de la taille du vocabulaire...")

        word_freq = Counter()
        for text in texts:
            words = text.split()
            word_freq.update(words)

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        min_freq = self.config.get('min_word_frequency', 2)
        filtered_words = [(word, freq) for word, freq in sorted_words if freq >= min_freq]

        total_word_occurrences = sum(freq for _, freq in filtered_words)
        cumulative_coverage = 0
        vocab_size = 0

        target_coverage = self.config.get('vocab_coverage', 0.95)

        for i, (word, freq) in enumerate(filtered_words):
            cumulative_coverage += freq / total_word_occurrences
            if cumulative_coverage >= target_coverage:
                vocab_size = i + 1
                break

        vocab_size = max(vocab_size, 1000)
        vocab_size += 100

        print(f"  Nombre total de mots uniques: {len(word_freq)}")
        print(f"  Mots avec fréquence >= {min_freq}: {len(filtered_words)}")
        print(f"  Taille du vocabulaire pour {target_coverage*100}% de couverture: {vocab_size}")

        return vocab_size

    def calculate_sequence_length(self, texts):
        print("\n📏 Calcul automatique de la longueur des séquences...")

        sequence_lengths = [len(text.split()) for text in texts]

        percentile = self.config.get('sequence_percentile', 95)
        max_length = int(np.percentile(sequence_lengths, percentile))

        max_length = max(max_length, 50)
        max_length = min(max_length, 700)

        print(f"  Longueur moyenne: {np.mean(sequence_lengths):.1f} mots")
        print(f"  Longueur médiane: {np.median(sequence_lengths):.1f} mots")
        print(f"  {percentile}ème percentile: {max_length} mots")
        print(f"  Min: {min(sequence_lengths)}, Max: {max(sequence_lengths)}")

        return max_length

    def extract_numerical_features(self, df):
        features = []

        try:
            with open(self.config['suspicious_words_path'], 'r') as f:
                suspicious_words_data = json.load(f)
                suspicious_words_set = set(suspicious_words_data.get('en', []) + suspicious_words_data.get('fr', []))
            print(f"✅ Mots suspects chargés depuis {self.config['suspicious_words_path']}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"⚠️ Erreur chargement mots suspects: {e}. Utilisation d'une liste vide.")
            suspicious_words_set = set()

        for text in df['text']:
            if pd.isna(text):
                text = ""
            text_str = str(text)
            text_lower = text_str.lower()

            char_count = len(text_str)
            word_count = len(text_str.split())
            exclamation_count = text_str.count('!')
            question_count = text_str.count('?')
            upper_count = sum(1 for c in text_str if c.isupper())
            upper_ratio = upper_count / max(char_count, 1)
            url_count = len(re.findall(r'http[s]?://', text_str))
            email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_str))

            suspicious_count = sum(1 for word in suspicious_words_set if word in text_lower)

            digit_ratio = sum(1 for c in text_str if c.isdigit()) / max(char_count, 1)
            special_char_ratio = sum(1 for c in text_str if c in '!@#$%^&*()') / max(char_count, 1)

            features.append([
                char_count, word_count, exclamation_count, question_count,
                upper_ratio, url_count, email_count, suspicious_count,
                digit_ratio, special_char_ratio
            ])

        features_np = np.array(features)

        if self.num_numerical_features is None:
            self.num_numerical_features = features_np.shape[1]
            print(f"📊 Nombre de features numériques détectées : {self.num_numerical_features}")

        return features_np

    def prepare_sequences(self, df, is_training=True):
        print("\nPréparation des séquences...")
        df['processed_text'] = df.apply(self.preprocess_text, axis=1)

        if is_training:
            if self.config['max_vocab_size'] is None:
                self.config['max_vocab_size'] = self.calculate_vocab_size(df['processed_text'].values)
                print(f"\n✅ max_vocab_size calculé automatiquement: {self.config['max_vocab_size']}")

            if self.config['max_sequence_length'] is None:
                self.config['max_sequence_length'] = self.calculate_sequence_length(df['processed_text'].values)
                print(f"✅ max_sequence_length calculé automatiquement: {self.config['max_sequence_length']}")

            self.tokenizer = Tokenizer(
                num_words=self.config['max_vocab_size'],
                oov_token='<OOV>',
                filters='',
                lower=False
            )
            self.tokenizer.fit_on_texts(df['processed_text'])

            actual_vocab_size = min(len(self.tokenizer.word_index) + 1, self.config['max_vocab_size'])
            print(f"\n📖 Vocabulaire final:")
            print(f"  Taille totale du vocabulaire tokenizer: {len(self.tokenizer.word_index)}")
            print(f"  Taille effective utilisée: {actual_vocab_size}")

        sequences = self.tokenizer.texts_to_sequences(df['processed_text'])

        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.config['max_sequence_length'],
            padding='post',
            truncating='post'
        )
        print(f"  Séquences créées: {padded_sequences.shape}")

        numerical_features = self.extract_numerical_features(df)
        if is_training:
            numerical_features = self.scaler.fit_transform(numerical_features)
        else:
            numerical_features = self.scaler.transform(numerical_features)
        print(f"  Features numériques: {numerical_features.shape}")

        return padded_sequences, numerical_features

    def build_lstm_model(self):
        print("\nConstruction du modèle LSTM...")

        text_input = Input(shape=(self.config['max_sequence_length'],), name='text_input')

        vocab_size = min(len(self.tokenizer.word_index) + 1, self.config['max_vocab_size'])
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=self.config['embedding_dim'],
            input_length=self.config['max_sequence_length'],
            mask_zero=True
        )(text_input)

        lstm_out = Bidirectional(LSTM(
            self.config['lstm_units'],
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate']
        ))(embedding)

        lstm_features = GlobalMaxPooling1D()(lstm_out)

        numerical_input = Input(shape=(self.num_numerical_features,), name='numerical_input')
        numerical_dense = Dense(16, activation='relu')(numerical_input)

        combined = Concatenate()([lstm_features, numerical_dense])

        dense1 = Dense(self.config['dense_units'], activation='relu')(combined)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.config['dropout_rate'])(dense1)

        dense2 = Dense(16, activation='relu')(dense1)
        dense2 = Dropout(self.config['dropout_rate'])(dense2)

        output = Dense(1, activation='sigmoid', name='output')(dense2)

        model = Model(inputs=[text_input, numerical_input], outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )

        print("\nArchitecture du modèle:")
        model.summary()

        self.model = model
        return model

    def save_model_artifacts(self, model_name="best_lstm_model"):
        print(f"\nSAUVEGARDE DES ARTEFACTS POUR L'API")
        print("=" * 50)

        model_path = f"{model_name}.keras"
        self.model.save(model_path)
        print(f"✅ Modèle sauvegardé: {model_path}")

        tokenizer_path = "tokenizer.pkl"
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"✅ Tokenizer sauvegardé: {tokenizer_path}")

        scaler_path = "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler sauvegardé: {scaler_path}")

        label_encoder_path = "label_encoder.pkl"
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✅ Label encoder sauvegardé: {label_encoder_path}")

        metadata = {
            'model_type': 'LSTM_Hybrid_FR_EN',
            'model_file': model_path,
            'tokenizer_file': tokenizer_path,
            'scaler_file': scaler_path,
            'label_encoder_file': label_encoder_path,
            'config': self.config,
            'vocabulary_size': len(self.tokenizer.word_index),
            'actual_vocab_size': min(len(self.tokenizer.word_index) + 1, self.config['max_vocab_size']),
            'classes': list(self.label_encoder.classes_),
            'creation_date': pd.Timestamp.now().isoformat()
        }

        metadata_path = "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Métadonnées sauvegardées: {metadata_path}")

        print(f"\n📦 ARTEFACTS PRÊTS POUR L'API:")
        print(f"  {model_path}")
        print(f"  {tokenizer_path}")
        print(f"  {scaler_path}")
        print(f"  {label_encoder_path}")
        print(f"  {metadata_path}")
        print(f"\n⚡️ Copiez ces fichiers dans votre dossier Docker!")

        return {
            'model_path': model_path,
            'tokenizer_path': tokenizer_path,
            'scaler_path': scaler_path,
            'label_encoder_path': label_encoder_path,
            'metadata_path': metadata_path
        }

    def train_model(self, X_text_train, X_num_train, y_train,
                    X_text_val, X_num_val, y_val):
        print("\nEntraînement du modèle LSTM...")

        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)

        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_encoded),
            y=y_train_encoded
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"⚖️ Poids de classe calculés pour gérer le déséquilibre : {class_weight_dict}")

        self.build_lstm_model()

        callbacks = [
            EarlyStopping(
                patience=self.config['patience'],
                restore_best_weights=True,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                monitor='val_loss'
            ),
            ModelCheckpoint(
                'best_lstm_model.keras',
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        print("\nDébut de l'entraînement...")
        self.history = self.model.fit(
            [X_text_train, X_num_train], y_train_encoded,
            validation_data=([X_text_val, X_num_val], y_val_encoded),
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✅ Entraînement terminé!")
        self.save_model_artifacts("best_lstm_model")

        return self.history

    def evaluate_model(self, X_text_test, X_num_test, y_test):
        print("\nÉVALUATION DU MODÈLE")
        print("=" * 40)

        y_test_encoded = self.label_encoder.transform(y_test)

        y_pred_proba = self.model.predict([X_text_test, X_num_test])
        y_pred = (y_pred_proba > 0.5).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred)
        recall = recall_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred)
        auc = roc_auc_score(y_test_encoded, y_pred_proba)

        print(f"\n📊 Métriques de performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")

        cm = confusion_matrix(y_test_encoded, y_pred)
        labels = self.label_encoder.classes_

        print(f"\n🔢 Matrice de confusion:")
        print(f"              Predicted: {labels[0]:<10} {labels[1]:<10}")
        print(f"Actual:")
        for i, label in enumerate(labels):
            print(f"{label:<10}              {cm[i,0]:<10} {cm[i,1]:<10}")

        print(f"\n📝 Rapport de classification:")
        print(classification_report(y_test_encoded, y_pred, target_names=labels, digits=4))

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
        if not self.history:
            print("Pas d'historique d'entraînement disponible")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0,0].plot(self.history.history['loss'], label='Train Loss')
        axes[0,0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0,0].set_title('Model Loss')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()

        axes[0,1].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0,1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0,1].set_title('Model Accuracy')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()

        axes[1,0].plot(self.history.history['precision'], label='Train Precision')
        axes[1,0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1,0].set_title('Model Precision')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].legend()

        axes[1,1].plot(self.history.history['recall'], label='Train Recall')
        axes[1,1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1,1].set_title('Model Recall')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].legend()

        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def predict_new_texts(self, texts, languages):
        if not self.model:
            print("Modèle non entraîné")
            return None

        temp_df = pd.DataFrame({'text': texts, 'language': languages})
        X_text, X_num = self.prepare_sequences(temp_df, is_training=False)

        probabilities = self.model.predict([X_text, X_num])
        predictions = (probabilities > 0.5).astype(int)

        predictions_decoded = self.label_encoder.inverse_transform(predictions.flatten())

        return predictions_decoded, probabilities.flatten()

    def retrain_from_feedback(self, feedback_df, main_dataset_path=None, sample_size=2000):
        """
        Réentraîne le modèle en utilisant les feedbacks négatifs + échantillon du dataset principal
        Suit exactement la même méthode de traitement que l'entraînement initial

        Args:
            feedback_df: DataFrame avec colonnes ['text', 'label', 'language']
            main_dataset_path: Chemin vers le dataset principal (optionnel)
            sample_size: Taille de l'échantillon du dataset principal

        Returns:
            history: Historique d'entraînement
            metrics: Métriques de performance
            X_text_test, X_num_test, y_test: Données de test pour comparaison
        """
        print(f"\n🔄 RÉENTRAÎNEMENT À PARTIR DES FEEDBACKS")
        print("=" * 60)

        # 1. Préparer le dataset combiné
        combined_df = None

        if main_dataset_path:
            try:
                print(f"📂 Chargement du dataset principal depuis {main_dataset_path}...")
                main_df = self.load_data(main_dataset_path, sample_size=sample_size)

                if main_df is not None:
                    # Combiner avec les feedbacks
                    combined_df = pd.concat([main_df, feedback_df], ignore_index=True)
                    print(
                        f"🔗 Dataset combiné: {len(main_df)} (principal) + {len(feedback_df)} (feedbacks) = {len(combined_df)}")
                else:
                    combined_df = feedback_df
                    print("⚠️ Dataset principal non trouvé, utilisation uniquement des feedbacks")

            except Exception as e:
                print(f"⚠️ Erreur lors du chargement du dataset principal: {e}")
                combined_df = feedback_df
                print("📝 Utilisation uniquement des feedbacks pour le réentraînement")
        else:
            combined_df = feedback_df
            print("📝 Réentraînement uniquement sur les feedbacks (aucun dataset principal spécifié)")

        if len(combined_df) == 0:
            raise ValueError("Aucune donnée disponible pour le réentraînement")

        # 2. Vérifier les colonnes requises
        required_columns = ['text', 'label', 'language']
        missing_columns = [col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes dans le dataset: {missing_columns}")

        # 3. Afficher les statistiques du dataset combiné
        print(f"\n📊 STATISTIQUES DU DATASET DE RÉENTRAÎNEMENT:")
        print(f"  Nombre total d'échantillons: {len(combined_df)}")

        # Distribution des labels
        label_counts = combined_df['label'].value_counts()
        print(f"\n📋 Distribution des labels:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count / len(combined_df) * 100:.1f}%)")

        # Distribution des langues
        if 'language' in combined_df.columns:
            lang_counts = combined_df['language'].value_counts()
            print(f"\n🌍 Distribution des langues:")
            for lang, count in lang_counts.items():
                print(f"  {lang}: {count} ({count / len(combined_df) * 100:.1f}%)")

        # 4. Division des données (EXACTEMENT comme dans main())
        print(f"\n📋 Division des données pour le réentraînement...")

        # Créer la colonne de stratification (label + langue)
        combined_df['stratify_col'] = combined_df['label'].astype(str) + '_' + combined_df['language'].astype(str)

        X = combined_df[['text', 'language', 'stratify_col']]
        y = combined_df['label']

        # Division train/temp avec stratification
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=combined_df['stratify_col']
        )

        # Division validation/test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=X_temp['stratify_col']
        )

        print(f"  Train: {len(X_train)} échantillons")
        print(f"  Validation: {len(X_val)} échantillons")
        print(f"  Test: {len(X_test)} échantillons")

        # 5. Préparation des séquences (EXACTEMENT comme dans main())
        print(f"\n🔧 Préparation des séquences...")

        # IMPORTANT: is_training=True pour recalculer vocab et séquences
        X_text_train, X_num_train = self.prepare_sequences(X_train, is_training=True)
        X_text_val, X_num_val = self.prepare_sequences(X_val, is_training=False)
        X_text_test, X_num_test = self.prepare_sequences(X_test, is_training=False)

        print(f"  Séquences texte - Train: {X_text_train.shape}, Val: {X_text_val.shape}, Test: {X_text_test.shape}")
        print(f"  Features numériques - Train: {X_num_train.shape}, Val: {X_num_val.shape}, Test: {X_num_test.shape}")

        # 6. Entraînement du modèle (EXACTEMENT comme dans train_model())
        print(f"\n🚀 DÉBUT DU RÉENTRAÎNEMENT...")
        print("=" * 50)

        # Utiliser la méthode train_model existante
        history = self.train_model(
            X_text_train, X_num_train, y_train,
            X_text_val, X_num_val, y_val
        )

        # 7. Évaluation du modèle réentraîné
        print(f"\n📊 ÉVALUATION DU MODÈLE RÉENTRAÎNÉ")
        print("=" * 50)

        metrics = self.evaluate_model(X_text_test, X_num_test, y_test)

        print(f"\n✅ RÉENTRAÎNEMENT TERMINÉ!")
        print(f"📈 Performances du modèle réentraîné:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")

        # 8. Sauvegarder les nouveaux artefacts avec un nom spécifique
        print(f"\n💾 Sauvegarde des artefacts du modèle réentraîné...")
        artifacts = self.save_model_artifacts("./data/retrained_lstm_model")

        # 9. Retourner les données nécessaires pour la comparaison
        return history, metrics, X_text_test, X_num_test, y_test
def main():
    config = {
        'embedding_dim': 128,
        'lstm_units': 64,
        'dense_units': 32,
        'dropout_rate': 0.4,
        'learning_rate': 0.001,
        'batch_size': 128,
        'epochs': 30,
        'patience': 2,
        'vocab_coverage': 0.95,
        'sequence_percentile': 95,
        'min_word_frequency': 2
    }

    detector = LSTMPhishingDetector(config)

    df = detector.load_data('full_merged_dataset_fr_en_spam.csv')
    if df is None:
        print("Impossible de charger les données")
        return

    print("\n📋 Division des données...")
    df['stratify_col'] = df['label'].astype(str) + '_' + df['language'].astype(str)

    X = df[['text', 'language', 'stratify_col']]
    y = df['label']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=df['stratify_col']
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=X_temp['stratify_col']
    )

    print(f"  Train: {len(X_train)} échantillons")
    print(f"  Validation: {len(X_val)} échantillons")
    print(f"  Test: {len(X_test)} échantillons")

    X_text_train, X_num_train = detector.prepare_sequences(X_train, is_training=True)
    X_text_val, X_num_val = detector.prepare_sequences(X_val, is_training=False)
    X_text_test, X_num_test = detector.prepare_sequences(X_test, is_training=False)

    print(f"\n{'='*60}")
    print("ENTRAÎNEMENT DU MODÈLE LSTM")
    print(f"{'='*60}")

    history = detector.train_model(
        X_text_train, X_num_train, y_train,
        X_text_val, X_num_val, y_val
    )

    metrics = detector.evaluate_model(X_text_test, X_num_test, y_test)

    print("\nVisualisation de l'entraînement:")
    detector.plot_training_history()

    print(f"\n{'='*60}")
    print("RÉSULTATS FINAUX")
    print(f"{'='*60}")
    print(f"\nParamètres calculés automatiquement:")
    print(f"  max_vocab_size: {detector.config['max_vocab_size']}")
    print(f"  max_sequence_length: {detector.config['max_sequence_length']}")
    print(f"\nMétriques de performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")

    print(f"\n🔮 EXEMPLES DE PRÉDICTIONS (FR & EN)")
    print("=" * 50)
    sample_texts = [
        "URGENT: Your account will be suspended in 24 hours. Click here to verify your identity.",
        "Bonjour, votre facture no. 8373 arrive à échéance. Veuillez confirmer votre paiement immédiatement pour éviter la suspension.",
        "Hi Sarah, thanks for sending the quarterly report. Could we schedule a meeting next week?",
        "Félicitations ! Vous avez gagné un prix de 10.000€. Envoyez vos coordonnées bancaires pour réclamer votre gain."
    ]
    sample_langs = ['en', 'fr', 'en', 'fr']

    predictions, probabilities = detector.predict_new_texts(sample_texts, sample_langs)

    for i, (text, pred, prob) in enumerate(zip(sample_texts, predictions, probabilities)):
        print(f"\nTexte {i+1} ({sample_langs[i]}): {text[:80]}...")
        print(f"  Prédiction: {pred}")
        print(f"  Probabilité (phishing): {prob:.4f}")
        print(f"  Confiance: {'PHISHING' if prob > 0.8 else 'SUSPECT' if prob > 0.5 else 'LÉGITIME'}")

    print(f"\n✅ ENTRAÎNEMENT TERMINÉ!")
    print("\nFichiers créés pour l'API Docker:")
    print("  - best_lstm_model.keras")
    print("  - tokenizer.pkl")
    print("  - scaler.pkl")
    print("  - label_encoder.pkl")
    print("  - model_metadata.json")

    return detector

if __name__ == "__main__":
    main()