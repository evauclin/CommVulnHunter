import pandas as pd
import numpy as np
import json
import pickle
import re
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.corpus import stopwords

# Configuration pour la reproductibilité
tf.random.set_seed(42)
np.random.seed(42)


class FineTuningManager:
    """
    Gestionnaire pour le fine-tuning du modèle LSTM basé sur les feedbacks négatifs
    """

    def __init__(self, model_dir="./model", data_dir="./data"):
        """
        Initialise le gestionnaire de fine-tuning

        Args:
            model_dir: Répertoire contenant les artefacts du modèle
            data_dir: Répertoire contenant les données et feedbacks
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)

        # Chemins des artefacts
        self.model_path = self.model_dir / "best_lstm_model.keras"
        self.tokenizer_path = self.model_dir / "tokenizer.pkl"
        self.scaler_path = self.model_dir / "scaler.pkl"
        self.label_encoder_path = self.model_dir / "label_encoder.pkl"
        self.metadata_path = self.model_dir / "model_metadata.json"
        self.suspicious_words_path = self.model_dir / "suspicious_words.json"

        # Chemin du fichier de feedbacks
        self.feedback_csv_path = self.data_dir / "user_feedbacks.csv"

        # Artefacts du modèle
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.suspicious_words_set = set()

        # Configuration fine-tuning
        self.finetune_config = {
            'learning_rate': 0.00005,  # Learning rate très faible
            'epochs': 10,  # Peu d'époques pour éviter l'overfitting
            'batch_size': 32,  # Petit batch size
            'patience': 3,  # Patience réduite
            'validation_split': 0.2,  # Split pour validation
            'sample_size': 200  # Taille échantillon dataset principal
        }

        # Charger stopwords
        self._setup_stopwords()

        print("🎯 FineTuningManager initialisé")
        print(f"   Model dir: {self.model_dir}")
        print(f"   Data dir: {self.data_dir}")

    def _setup_stopwords(self):
        """Configure les stopwords NLTK"""
        try:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("📥 Téléchargement stopwords NLTK...")
                nltk.download('stopwords', quiet=True)

            self.stop_words = {
                'en': set(stopwords.words('english')),
                'fr': set(stopwords.words('french'))
            }
            print("✅ Stopwords configurés")
        except Exception as e:
            print(f"⚠️ Erreur stopwords: {e}")
            # Fallback basique
            self.stop_words = {
                'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'},
                'fr': {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais'}
            }

    # CORRECTION À APPLIQUER DANS traitement.py

    def load_model_artifacts(self):
        """
        ÉTAPE 1: Charge tous les artefacts du modèle existant
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 1: CHARGEMENT DES ARTEFACTS DU MODÈLE")
        print("=" * 60)

        try:
            # Charger le modèle EN PREMIER pour obtenir la vraie longueur de séquence
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")

            self.model = load_model(str(self.model_path))
            print(f"✅ Modèle chargé: {self.model_path}")

            # CORRECTION CRITIQUE: Détecter la vraie longueur de séquence depuis le modèle
            model_input_shape = self.model.inputs[0].shape
            actual_sequence_length = model_input_shape[1]
            print(f"🔍 Longueur de séquence détectée depuis le modèle: {actual_sequence_length}")

            # Charger le tokenizer
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Tokenizer chargé (vocab: {len(self.tokenizer.word_index)})")

            # Charger le scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Scaler chargé")

            # Charger le label encoder
            with open(self.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Label encoder chargé: {self.label_encoder.classes_}")

            # Charger les métadonnées
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print("✅ Métadonnées chargées")

            # CORRECTION CRITIQUE: Utiliser la longueur détectée du modèle
            if 'config' not in self.metadata:
                self.metadata['config'] = {}

            # Forcer l'utilisation de la longueur réelle du modèle
            self.metadata['config']['max_sequence_length'] = actual_sequence_length
            print(f"🔧 Longueur de séquence corrigée: {actual_sequence_length}")

            # Charger les mots suspects
            if self.suspicious_words_path.exists():
                with open(self.suspicious_words_path, 'r') as f:
                    suspicious_data = json.load(f)
                self.suspicious_words_set = set(
                    suspicious_data.get('en', []) + suspicious_data.get('fr', [])
                )
                print(f"✅ Mots suspects chargés: {len(self.suspicious_words_set)}")
            else:
                print("⚠️ Fichier mots suspects non trouvé")

            print(f"\n📋 Configuration du modèle:")
            print(f"   max_sequence_length: {self.metadata['config']['max_sequence_length']}")
            print(f"   max_vocab_size: {self.metadata['config'].get('max_vocab_size', 'Non défini')}")
            print(f"   Classes: {self.metadata.get('classes', self.label_encoder.classes_)}")

            return True

        except Exception as e:
            print(f"❌ Erreur chargement artefacts: {e}")
            import traceback
            traceback.print_exc()
            return False
    def extract_negative_feedbacks(self):
        """
        ÉTAPE 2: Extrait les feedbacks négatifs non traités du fichier CSV
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 2: EXTRACTION DES FEEDBACKS NÉGATIFS")
        print("=" * 60)

        if not self.feedback_csv_path.exists():
            print(f"❌ Fichier feedback non trouvé: {self.feedback_csv_path}")
            return pd.DataFrame()

        try:
            # Charger tous les feedbacks
            df = pd.read_csv(self.feedback_csv_path)
            print(f"📊 Total feedbacks: {len(df)}")

            # Filtrer les feedbacks négatifs non traités
            negative_feedbacks = df[
                (df['user_satisfaction'] == 'no') &
                (df['processed'] == False)
                ].copy()

            print(f"🔍 Feedbacks négatifs non traités: {len(negative_feedbacks)}")

            if len(negative_feedbacks) == 0:
                print("ℹ️ Aucun feedback négatif à traiter")
                return pd.DataFrame()

            # Afficher les détails
            print(f"\n📋 Distribution des feedbacks négatifs:")
            if 'language_detected' in negative_feedbacks.columns:
                lang_dist = negative_feedbacks['language_detected'].value_counts()
                for lang, count in lang_dist.items():
                    print(f"   {lang}: {count}")

            # Préparer les données pour l'entraînement
            feedback_data = []
            for _, row in negative_feedbacks.iterrows():
                # Déterminer le vrai label (inverse de la prédiction)
                predicted_class = row['predicted_class']
                if predicted_class in ['phishing', 'spam']:
                    true_label = 'benign'
                else:
                    true_label = 'phishing'

                feedback_data.append({
                    'text': row['email_text'],
                    'label': true_label,
                    'language': row['language_detected'],
                    'feedback_id': row.name,
                    'original_prediction': predicted_class,
                    'confidence': row['predicted_probability']
                })

            feedback_df = pd.DataFrame(feedback_data)
            print(f"✅ Données de feedback préparées: {len(feedback_df)} échantillons")

            return feedback_df

        except Exception as e:
            print(f"❌ Erreur extraction feedbacks: {e}")
            return pd.DataFrame()

    def load_dataset_sample(self, dataset_path, sample_size=200):
        """
        ÉTAPE 3: Charge un échantillon du dataset principal
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 3: CHARGEMENT ÉCHANTILLON DATASET PRINCIPAL")
        print("=" * 60)

        try:
            if not Path(dataset_path).exists():
                print(f"⚠️ Dataset principal non trouvé: {dataset_path}")
                return pd.DataFrame()

            # Charger le dataset
            df = pd.read_csv(dataset_path)
            print(f"📊 Dataset principal: {len(df)} échantillons")

            # Échantillonnage stratifié
            if len(df) > sample_size:
                # Créer une colonne de stratification
                df['stratify_col'] = df['label'].astype(str) + '_' + df['language'].astype(str)

                # Échantillonnage stratifié
                sample_df = df.groupby('stratify_col', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), sample_size // len(df['stratify_col'].unique()) + 1),
                                       random_state=42)
                ).reset_index(drop=True)

                # Si on a encore trop d'échantillons, faire un échantillonnage final
                if len(sample_df) > sample_size:
                    sample_df = sample_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

                print(f"📋 Échantillon sélectionné: {len(sample_df)}")
            else:
                sample_df = df.copy()
                print(f"📋 Dataset complet utilisé: {len(sample_df)}")

            # Afficher la distribution
            print(f"\n📊 Distribution de l'échantillon:")
            label_dist = sample_df['label'].value_counts()
            for label, count in label_dist.items():
                print(f"   {label}: {count}")

            if 'language' in sample_df.columns:
                lang_dist = sample_df['language'].value_counts()
                print(f"\n🌍 Distribution des langues:")
                for lang, count in lang_dist.items():
                    print(f"   {lang}: {count}")

            return sample_df[['text', 'label', 'language']].copy()

        except Exception as e:
            print(f"❌ Erreur chargement dataset: {e}")
            return pd.DataFrame()

    def combine_datasets(self, feedback_df, sample_df):
        """
        ÉTAPE 4: Combine les feedbacks négatifs avec l'échantillon du dataset
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 4: COMBINAISON DES DATASETS")
        print("=" * 60)

        if feedback_df.empty and sample_df.empty:
            print("❌ Aucune donnée disponible pour le fine-tuning")
            return pd.DataFrame()

        # Préparer les DataFrames
        datasets_to_combine = []

        if not feedback_df.empty:
            feedback_clean = feedback_df[['text', 'label', 'language']].copy()
            feedback_clean['source'] = 'feedback'
            datasets_to_combine.append(feedback_clean)
            print(f"📝 Feedbacks: {len(feedback_clean)} échantillons")

        if not sample_df.empty:
            sample_clean = sample_df[['text', 'label', 'language']].copy()
            sample_clean['source'] = 'dataset'
            datasets_to_combine.append(sample_clean)
            print(f"📊 Dataset principal: {len(sample_clean)} échantillons")

        if not datasets_to_combine:
            return pd.DataFrame()

        # Combiner
        combined_df = pd.concat(datasets_to_combine, ignore_index=True)
        print(f"🔗 Dataset combiné: {len(combined_df)} échantillons")

        # Afficher les statistiques finales
        print(f"\n📋 Distribution finale:")
        label_dist = combined_df['label'].value_counts()
        for label, count in label_dist.items():
            print(f"   {label}: {count} ({count / len(combined_df) * 100:.1f}%)")

        print(f"\n📊 Sources:")
        source_dist = combined_df['source'].value_counts()
        for source, count in source_dist.items():
            print(f"   {source}: {count}")

        return combined_df

    def preprocess_text(self, text, language='en'):
        """
        Préprocesse le texte avec la même méthode que le modèle original
        """
        if pd.isna(text):
            return ""

        text = str(text).lower()

        # Remplacements de tokens
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                      ' URL_TOKEN ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL_TOKEN ', text)
        text = re.sub(r'\b\d+\b', ' NUM_TOKEN ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Filtrage des mots
        tokens = text.split()
        stop_words_lang = self.stop_words.get(language, self.stop_words['en'])
        filtered_tokens = [token for token in tokens if len(token) > 2 and token not in stop_words_lang]

        return ' '.join(filtered_tokens)

    def extract_numerical_features(self, texts):
        """
        Extrait les features numériques avec la même méthode que le modèle original
        """
        features = []

        for text in texts:
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
            suspicious_count = sum(1 for word in self.suspicious_words_set if word in text_str.lower())
            digit_ratio = sum(1 for c in text_str if c.isdigit()) / max(char_count, 1)
            special_char_ratio = sum(1 for c in text_str if c in '!@#$%^&*()') / max(char_count, 1)

            features.append([
                char_count, word_count, exclamation_count, question_count,
                upper_ratio, url_count, email_count, suspicious_count,
                digit_ratio, special_char_ratio
            ])

        return np.array(features)

    # CORRECTION COMPLÈTE À APPLIQUER DANS prepare_fine_tuning_data dans traitement.py

    def prepare_fine_tuning_data(self, combined_df):
        """
        ÉTAPE 5: Prépare les données pour le fine-tuning
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 5: PRÉPARATION DES DONNÉES FINE-TUNING")
        print("=" * 60)

        if combined_df.empty:
            print("❌ Aucune donnée à préparer")
            return None, None, None, None, None, None

        try:
            # CORRECTION 1: Détecter la vraie longueur de séquence depuis le modèle
            model_input_shape = self.model.inputs[0].shape
            actual_sequence_length = model_input_shape[1]

            # Mettre à jour les métadonnées avec la vraie longueur
            self.metadata['config']['max_sequence_length'] = actual_sequence_length
            print(f"🔧 Longueur de séquence corrigée: {actual_sequence_length}")

            # CORRECTION 2: Détecter la taille max du vocabulaire depuis le modèle
            # La couche embedding nous donne la vraie taille du vocabulaire
            embedding_layer = None
            for layer in self.model.layers:
                if 'embedding' in layer.name.lower():
                    embedding_layer = layer
                    break

            if embedding_layer is not None:
                actual_vocab_size = embedding_layer.input_dim
                print(f"🔧 Taille du vocabulaire détectée: {actual_vocab_size}")
            else:
                # Fallback : utiliser 10001 (0 à 10000)
                actual_vocab_size = 10001
                print(f"⚠️ Couche embedding non trouvée, utilisation de la taille par défaut: {actual_vocab_size}")

            # Prétraitement des textes
            print("🔧 Prétraitement des textes...")
            processed_texts = []
            for _, row in combined_df.iterrows():
                processed_text = self.preprocess_text(row['text'], row.get('language', 'en'))
                processed_texts.append(processed_text)

            # CORRECTION 3: Création des séquences avec filtrage des tokens invalides
            print("📝 Création des séquences...")
            sequences = self.tokenizer.texts_to_sequences(processed_texts)

            # Filtrer les tokens qui dépassent la taille du vocabulaire
            max_vocab_id = actual_vocab_size - 1  # Index maximum valide
            filtered_sequences = []
            total_tokens_removed = 0

            for sequence in sequences:
                filtered_sequence = [token_id for token_id in sequence if token_id <= max_vocab_id]
                tokens_removed = len(sequence) - len(filtered_sequence)
                total_tokens_removed += tokens_removed
                filtered_sequences.append(filtered_sequence)

            print(f"🔧 Tokens invalides supprimés: {total_tokens_removed}")
            print(f"🔧 Plage valide des tokens: 0 à {max_vocab_id}")

            # UTILISER LA LONGUEUR CORRIGÉE ET LES SÉQUENCES FILTRÉES
            max_seq_length = actual_sequence_length
            X_text = pad_sequences(filtered_sequences, maxlen=max_seq_length, padding='post', truncating='post')

            # Features numériques (UTILISER le scaler existant)
            print("🔢 Extraction des features numériques...")
            X_num = self.extract_numerical_features(combined_df['text'])
            X_num = self.scaler.transform(X_num)  # Transform seulement, pas fit_transform

            # Labels (UTILISER le label encoder existant)
            print("🏷️ Préparation des labels...")
            y = self.label_encoder.transform(combined_df['label'])

            # Division train/validation
            print("📋 Division train/validation...")
            X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
                X_text, X_num, y,
                test_size=self.finetune_config['validation_split'],
                random_state=42,
                stratify=y
            )

            print(f"✅ Données préparées:")
            print(f"   Train: {len(X_text_train)} échantillons")
            print(f"   Validation: {len(X_text_val)} échantillons")
            print(f"   Séquences: {X_text_train.shape} (longueur: {max_seq_length})")
            print(f"   Features: {X_num_train.shape}")
            print(f"   Vocabulaire: 0 à {max_vocab_id}")
            print(f"   Distribution train: {np.bincount(y_train)}")
            print(f"   Distribution val: {np.bincount(y_val)}")

            return X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val

        except Exception as e:
            print(f"❌ Erreur préparation données: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None, None
    def perform_fine_tuning(self, X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val):
        """
        ÉTAPE 6: Effectue le fine-tuning du modèle
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 6: FINE-TUNING DU MODÈLE")
        print("=" * 60)

        try:
            # Configurer l'optimiseur avec un learning rate très faible
            print(f"🎯 Configuration fine-tuning:")
            for key, value in self.finetune_config.items():
                print(f"   {key}: {value}")

            new_optimizer = Adam(learning_rate=self.finetune_config['learning_rate'])
            self.model.compile(
                optimizer=new_optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall')]
            )

            # Callbacks pour le fine-tuning
            callbacks = [
                EarlyStopping(
                    patience=self.finetune_config['patience'],
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                ReduceLROnPlateau(
                    factor=0.5,
                    patience=2,
                    min_lr=1e-8,
                    monitor='val_loss'
                )
            ]

            # Calcul des poids de classe pour gérer le déséquilibre
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            print(f"⚖️ Poids de classe: {class_weight_dict}")

            # Fine-tuning
            print(f"\n🚀 Début du fine-tuning...")
            history = self.model.fit(
                [X_text_train, X_num_train], y_train,
                validation_data=([X_text_val, X_num_val], y_val),
                batch_size=self.finetune_config['batch_size'],
                epochs=self.finetune_config['epochs'],
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )

            print("✅ Fine-tuning terminé!")
            return history

        except Exception as e:
            print(f"❌ Erreur fine-tuning: {e}")
            return None

    def evaluate_finetuned_model(self, X_text_val, X_num_val, y_val):
        """
        ÉTAPE 7: Évalue le modèle fine-tuné
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 7: ÉVALUATION DU MODÈLE FINE-TUNÉ")
        print("=" * 60)

        try:
            # Prédictions
            y_pred_proba = self.model.predict([X_text_val, X_num_val], verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Métriques
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            print(f"📊 Métriques du modèle fine-tuné:")
            for metric, value in metrics.items():
                print(f"   {metric.capitalize()}: {value:.4f}")

            # Rapport détaillé
            print(f"\n📋 Rapport de classification:")
            report = classification_report(
                y_val, y_pred,
                target_names=self.label_encoder.classes_,
                digits=4
            )
            print(report)

            return metrics

        except Exception as e:
            print(f"❌ Erreur évaluation: {e}")
            return {}

    def save_finetuned_model(self):
        """
        ÉTAPE 8: Sauvegarde le modèle fine-tuné
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 8: SAUVEGARDE DU MODÈLE FINE-TUNÉ")
        print("=" * 60)

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Créer le dossier de sauvegarde
            save_dir = self.data_dir / f"finetuned_model_{timestamp}"
            save_dir.mkdir(exist_ok=True)

            # Sauvegarder le modèle
            model_path = save_dir / "best_lstm_model.keras"
            self.model.save(str(model_path))
            print(f"✅ Modèle sauvegardé: {model_path}")

            # Copier les artefacts inchangés
            import shutil

            artifacts_to_copy = [
                ('tokenizer.pkl', self.tokenizer_path),
                ('scaler.pkl', self.scaler_path),
                ('label_encoder.pkl', self.label_encoder_path),
                ('suspicious_words.json', self.suspicious_words_path)
            ]

            for filename, source_path in artifacts_to_copy:
                if source_path.exists():
                    dest_path = save_dir / filename
                    shutil.copy2(source_path, dest_path)
                    print(f"✅ {filename} copié")

            # Mettre à jour les métadonnées
            updated_metadata = self.metadata.copy()
            updated_metadata.update({
                'finetune_timestamp': timestamp,
                'finetune_config': self.finetune_config,
                'model_type': 'LSTM_Hybrid_FR_EN_Finetuned',
                'original_model': str(self.model_path),
                'finetuned_model': str(model_path)
            })

            metadata_path = save_dir / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(updated_metadata, f, indent=2)
            print(f"✅ Métadonnées mises à jour: {metadata_path}")

            # Instructions pour remplacer le modèle
            print(f"\n📋 INSTRUCTIONS POUR UTILISER LE MODÈLE FINE-TUNÉ:")
            print(f"   1. Arrêter l'API Docker")
            print(f"   2. Remplacer les fichiers dans ./model/ par ceux de {save_dir}")
            print(f"   3. Redémarrer l'API Docker")
            print(f"\n💡 Ou utiliser le script de remplacement automatique")

            return {
                'save_dir': save_dir,
                'model_path': model_path,
                'metadata_path': metadata_path,
                'timestamp': timestamp
            }

        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return None

    def mark_feedbacks_as_processed(self, feedback_df):
        """
        ÉTAPE 9: Marque les feedbacks comme traités
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 9: MARQUAGE DES FEEDBACKS COMME TRAITÉS")
        print("=" * 60)

        if feedback_df.empty:
            print("ℹ️ Aucun feedback à marquer")
            return True

        try:
            # Charger le fichier CSV complet
            df = pd.read_csv(self.feedback_csv_path)

            # Marquer les feedbacks utilisés comme traités
            feedback_ids = feedback_df['feedback_id'].tolist() if 'feedback_id' in feedback_df.columns else []

            if feedback_ids:
                df.loc[feedback_ids, 'processed'] = True
                processed_count = len(feedback_ids)
            else:
                # Si pas d'IDs spécifiques, marquer tous les feedbacks négatifs non traités
                mask = (df['user_satisfaction'] == 'no') & (df['processed'] == False)
                df.loc[mask, 'processed'] = True
                processed_count = mask.sum()

            # Ajouter une colonne de traitement si elle n'existe pas
            df['processed_at'] = df.apply(
                lambda row: datetime.now().isoformat() if row['processed'] else None,
                axis=1
            )

            # Sauvegarder
            df.to_csv(self.feedback_csv_path, index=False)
            print(f"✅ {processed_count} feedbacks marqués comme traités")

            return True

        except Exception as e:
            print(f"❌ Erreur marquage feedbacks: {e}")
            return False

    def run_complete_finetuning(self, dataset_path="./data/full_merged_dataset_fr_en_spam.csv"):
        """
        FONCTION PRINCIPALE: Exécute le processus complet de fine-tuning
        """
        print("\n" + "🎯" * 30)
        print("PROCESSUS COMPLET DE FINE-TUNING")
        print("🎯" * 30)

        # Étape 1: Charger les artefacts
        if not self.load_model_artifacts():
            print("❌ Échec du chargement des artefacts")
            return False

        # Étape 2: Extraire les feedbacks négatifs
        feedback_df = self.extract_negative_feedbacks()
        if feedback_df.empty:
            print("ℹ️ Aucun feedback négatif à traiter")
            return False

        print(f"📝 {len(feedback_df)} feedbacks négatifs trouvés")

        # Étape 3: Charger échantillon du dataset principal
        sample_df = self.load_dataset_sample(dataset_path, self.finetune_config['sample_size'])

        # Étape 4: Combiner les datasets
        combined_df = self.combine_datasets(feedback_df, sample_df)
        if combined_df.empty:
            print("❌ Aucune donnée disponible pour le fine-tuning")
            return False

        # Étape 5: Préparer les données
        data_results = self.prepare_fine_tuning_data(combined_df)
        if data_results[0] is None:
            print("❌ Échec de la préparation des données")
            return False

        X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val = data_results

        # Étape 6: Fine-tuning
        history = self.perform_fine_tuning(X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val)
        if history is None:
            print("❌ Échec du fine-tuning")
            return False

        # Étape 7: Évaluation
        metrics = self.evaluate_finetuned_model(X_text_val, X_num_val, y_val)

        # Étape 8: Sauvegarde
        save_result = self.save_finetuned_model()
        if save_result is None:
            print("❌ Échec de la sauvegarde")
            return False

        # Étape 9: Marquer les feedbacks comme traités
        self.mark_feedbacks_as_processed(feedback_df)

        # Résumé final
        print("\n" + "🎉" * 30)
        print("FINE-TUNING TERMINÉ AVEC SUCCÈS!")
        print("🎉" * 30)
        print(f"📊 Données utilisées: {len(combined_df)} échantillons")
        print(f"📝 Feedbacks traités: {len(feedback_df)}")
        print(f"📈 Modèle sauvegardé: {save_result['save_dir']}")
        print(f"🎯 Métriques finales:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")

        return True


    def evaluate_on_full_dataset(self, model_to_evaluate, dataset_path, model_name=""):
        """
        Évalue un modèle sur l'ENSEMBLE du dataset pour avoir une vraie mesure de performance
        """
        print(f"\n🧪 ÉVALUATION COMPLÈTE SUR TOUT LE DATASET - {model_name}")
        print("=" * 70)

        try:
            # Charger TOUT le dataset (pas d'échantillonnage)
            if not Path(dataset_path).exists():
                print(f"❌ Dataset non trouvé: {dataset_path}")
                return None

            print(f"📂 Chargement du dataset complet...")
            df = pd.read_csv(dataset_path)
            print(f"📊 Dataset complet: {len(df)} échantillons")

            # Afficher la distribution
            label_dist = df['label'].value_counts()
            print(f"📋 Distribution:")
            for label, count in label_dist.items():
                print(f"   {label}: {count} ({count / len(df) * 100:.1f}%)")

            # Préparation des données (TOUT le dataset)
            print(f"🔧 Préparation des données...")

            # Prétraitement
            processed_texts = []
            for _, row in df.iterrows():
                processed_text = self.preprocess_text(row['text'], row.get('language', 'en'))
                processed_texts.append(processed_text)

            # Séquences avec filtrage
            sequences = self.tokenizer.texts_to_sequences(processed_texts)

            # Détecter les paramètres du modèle
            model_input_shape = model_to_evaluate.inputs[0].shape
            actual_sequence_length = model_input_shape[1]

            embedding_layer = None
            for layer in model_to_evaluate.layers:
                if 'embedding' in layer.name.lower():
                    embedding_layer = layer
                    break
            actual_vocab_size = embedding_layer.input_dim if embedding_layer else 10001
            max_vocab_id = actual_vocab_size - 1

            # Filtrer les séquences
            filtered_sequences = []
            for sequence in sequences:
                filtered_sequence = [token_id for token_id in sequence if token_id <= max_vocab_id]
                filtered_sequences.append(filtered_sequence)

            # Padding
            X_text = pad_sequences(filtered_sequences, maxlen=actual_sequence_length, padding='post', truncating='post')

            # Features numériques
            X_num = self.extract_numerical_features(df['text'])
            X_num = self.scaler.transform(X_num)

            # Labels
            y_true = self.label_encoder.transform(df['label'])

            print(f"✅ Données préparées: {X_text.shape[0]} échantillons")

            # Prédiction sur TOUT le dataset
            print(f"🎯 Prédiction sur l'ensemble du dataset...")
            y_pred_proba = model_to_evaluate.predict([X_text, X_num], batch_size=128, verbose=1)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calcul des métriques complètes
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
                confusion_matrix

            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'auc': roc_auc_score(y_true, y_pred_proba),
                'total_samples': len(y_true)
            }

            # Matrice de confusion
            cm = confusion_matrix(y_true, y_pred)

            print(f"\n📊 RÉSULTATS COMPLETS - {model_name}")
            print("=" * 50)
            print(f"📈 Accuracy:  {metrics['accuracy']:.4f}")
            print(f"📈 Precision: {metrics['precision']:.4f}")
            print(f"📈 Recall:    {metrics['recall']:.4f}")
            print(f"📈 F1-Score:  {metrics['f1']:.4f}")
            print(f"📈 AUC:       {metrics['auc']:.4f}")
            print(f"📊 Échantillons: {metrics['total_samples']}")

            print(f"\n🔍 Matrice de confusion:")
            print(f"   Vrais Négatifs: {cm[0, 0]}  |  Faux Positifs: {cm[0, 1]}")
            print(f"   Faux Négatifs:  {cm[1, 0]}  |  Vrais Positifs: {cm[1, 1]}")

            return metrics

        except Exception as e:
            print(f"❌ Erreur évaluation complète: {e}")
            import traceback
            traceback.print_exc()
            return None


    def compare_models_and_auto_deploy(self, finetuned_model_dir, dataset_path):
        """
        Compare le modèle original vs fine-tuné sur TOUT le dataset et remplace automatiquement si meilleur
        """
        print(f"\n🏆 COMPARAISON MODÈLE ORIGINAL VS FINE-TUNÉ")
        print("=" * 80)

        try:
            # 1. Évaluer le modèle ORIGINAL
            print(f"\n1️⃣ ÉVALUATION DU MODÈLE ORIGINAL")
            original_metrics = self.evaluate_on_full_dataset(self.model, dataset_path, "MODÈLE ORIGINAL")

            if original_metrics is None:
                print("❌ Impossible d'évaluer le modèle original")
                return False

            # 2. Charger et évaluer le modèle FINE-TUNÉ
            print(f"\n2️⃣ ÉVALUATION DU MODÈLE FINE-TUNÉ")
            finetuned_model_path = finetuned_model_dir / "best_lstm_model.keras"

            if not finetuned_model_path.exists():
                print(f"❌ Modèle fine-tuné non trouvé: {finetuned_model_path}")
                return False

            finetuned_model = load_model(str(finetuned_model_path))
            finetuned_metrics = self.evaluate_on_full_dataset(finetuned_model, dataset_path, "MODÈLE FINE-TUNÉ")

            if finetuned_metrics is None:
                print("❌ Impossible d'évaluer le modèle fine-tuné")
                return False

            # 3. COMPARAISON DÉTAILLÉE
            print(f"\n3️⃣ COMPARAISON DÉTAILLÉE")
            print("=" * 50)

            metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            improvements = {}

            print(f"{'Métrique':<12} {'Original':<10} {'Fine-tuné':<10} {'Amélioration':<12}")
            print("-" * 50)

            for metric in metrics_to_compare:
                original_val = original_metrics[metric]
                finetuned_val = finetuned_metrics[metric]
                improvement = finetuned_val - original_val
                improvement_pct = (improvement / original_val) * 100 if original_val > 0 else 0

                improvements[metric] = improvement

                status = "🟢" if improvement > 0 else "🔴" if improvement < 0 else "🟡"
                print(
                    f"{metric.capitalize():<12} {original_val:<10.4f} {finetuned_val:<10.4f} {status} {improvement_pct:+6.2f}%")

            # 4. DÉCISION AUTOMATIQUE
            print(f"\n4️⃣ DÉCISION AUTOMATIQUE")
            print("=" * 30)

            # Critères de décision (le modèle doit être meilleur sur les métriques importantes)
            key_metrics = ['f1', 'accuracy']  # Métriques principales
            is_better = all(improvements[metric] >= 0 for metric in key_metrics)  # Au moins égal
            significant_improvement = any(
                improvements[metric] > 0.01 for metric in key_metrics)  # Amélioration significative

            print(f"🔍 Analyse:")
            print(f"   Toutes les métriques clés >= original: {'✅' if is_better else '❌'}")
            print(f"   Amélioration significative (>1%): {'✅' if significant_improvement else '❌'}")

            if is_better and significant_improvement:
                print(f"\n🎉 DÉCISION: REMPLACER LE MODÈLE")
                print("   Le modèle fine-tuné est significativement meilleur!")

                # 5. REMPLACEMENT AUTOMATIQUE
                return self.auto_deploy_finetuned_model(finetuned_model_dir)

            elif is_better:
                print(f"\n⚠️ DÉCISION: GARDER L'ORIGINAL")
                print("   Le modèle fine-tuné n'apporte pas d'amélioration significative")
                return False

            else:
                print(f"\n❌ DÉCISION: GARDER L'ORIGINAL")
                print("   Le modèle fine-tuné est moins performant!")
                return False

        except Exception as e:
            print(f"❌ Erreur comparaison: {e}")
            return False


    def auto_deploy_finetuned_model(self, finetuned_model_dir):
        """
        Remplace automatiquement le modèle en production par la version fine-tunée
        """
        print(f"\n🚀 DÉPLOIEMENT AUTOMATIQUE")
        print("=" * 40)

        try:
            import shutil
            from datetime import datetime

            # 1. Créer une sauvegarde du modèle actuel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"./model_backup_before_finetune_{timestamp}")

            print(f"💾 Sauvegarde du modèle actuel...")
            shutil.copytree(self.model_dir, backup_dir)
            print(f"✅ Sauvegarde créée: {backup_dir}")

            # 2. Remplacer les fichiers
            model_files = [
                "best_lstm_model.keras",
                "model_metadata.json"
            ]

            print(f"🔄 Remplacement des fichiers...")
            for file in model_files:
                source = finetuned_model_dir / file
                dest = self.model_dir / file

                if source.exists():
                    shutil.copy2(source, dest)
                    print(f"✅ {file} remplacé")
                else:
                    print(f"⚠️ {file} non trouvé dans le modèle fine-tuné")

            # 3. Mettre à jour les métadonnées avec info de déploiement
            metadata_path = self.model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                metadata.update({
                    'deployed_at': datetime.now().isoformat(),
                    'deployment_method': 'automatic_after_finetuning',
                    'backup_location': str(backup_dir),
                    'replaced_model': 'original_production_model'
                })

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"✅ Métadonnées de déploiement mises à jour")

            print(f"\n🎉 DÉPLOIEMENT RÉUSSI!")
            print("=" * 25)
            print(f"✅ Modèle fine-tuné maintenant en production")
            print(f"💾 Ancien modèle sauvegardé dans: {backup_dir}")
            print(f"🔄 Redémarrez l'API pour utiliser le nouveau modèle:")
            print(f"   docker-compose restart fastapi")

            return True

        except Exception as e:
            print(f"❌ Erreur déploiement: {e}")
            return False


    # MODIFIER LA FONCTION run_complete_finetuning

    def run_complete_finetuning(self, dataset_path="./data/full_merged_dataset_fr_en_spam.csv"):
        """
        FONCTION PRINCIPALE: Exécute le processus complet de fine-tuning avec évaluation et déploiement automatique
        """
        print("\n" + "🎯" * 30)
        print("PROCESSUS COMPLET DE FINE-TUNING + ÉVALUATION + DÉPLOIEMENT")
        print("🎯" * 30)

        # Étapes 1-8 identiques (chargement, fine-tuning, sauvegarde)
        if not self.load_model_artifacts():
            return False

        feedback_df = self.extract_negative_feedbacks()
        if feedback_df.empty:
            print("ℹ️ Aucun feedback négatif à traiter")
            return False

        sample_df = self.load_dataset_sample(dataset_path, self.finetune_config['sample_size'])
        combined_df = self.combine_datasets(feedback_df, sample_df)
        if combined_df.empty:
            return False

        data_results = self.prepare_fine_tuning_data(combined_df)
        if data_results[0] is None:
            return False

        X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val = data_results

        history = self.perform_fine_tuning(X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val)
        if history is None:
            return False

        metrics = self.evaluate_finetuned_model(X_text_val, X_num_val, y_val)
        save_result = self.save_finetuned_model()
        if save_result is None:
            return False

        # ✨ NOUVELLE ÉTAPE 9: ÉVALUATION COMPLÈTE ET DÉPLOIEMENT AUTOMATIQUE
        print(f"\n" + "🏆" * 30)
        print("ÉTAPE 9: ÉVALUATION COMPLÈTE ET DÉPLOIEMENT AUTOMATIQUE")
        print("🏆" * 30)

        deployment_success = self.compare_models_and_auto_deploy(save_result['save_dir'], dataset_path)

        # Marquer les feedbacks comme traités
        self.mark_feedbacks_as_processed(feedback_df)

        # Résumé final
        print("\n" + "🎉" * 30)
        print("FINE-TUNING TERMINÉ!")
        print("🎉" * 30)
        print(f"📊 Données utilisées: {len(combined_df)} échantillons pour fine-tuning")
        print(f"📈 Évaluation: Sur TOUT le dataset ({dataset_path})")
        print(f"🚀 Déploiement automatique: {'✅ RÉUSSI' if deployment_success else '❌ Pas nécessaire'}")

        if deployment_success:
            print(f"🎯 RÉSULTAT: Nouveau modèle déployé automatiquement!")
            print(f"🔄 Redémarrez l'API pour utiliser le nouveau modèle")
        else:
            print(f"🎯 RÉSULTAT: Modèle original conservé (meilleur performance)")

        return True

def main():
    """
    Fonction principale pour exécuter le fine-tuning
    """
    print("🚀 DÉMARRAGE DU FINE-TUNING MANAGER")
    print("=" * 50)

    # Initialiser le gestionnaire
    manager = FineTuningManager(
        model_dir="./model",
        data_dir="./data"
    )

    # Vérifier les prérequis
    print("\n🔍 Vérification des prérequis...")

    required_files = [
        manager.model_path,
        manager.tokenizer_path,
        manager.scaler_path,
        manager.label_encoder_path,
        manager.metadata_path
    ]

    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("❌ Fichiers manquants:")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Assurez-vous d'avoir entraîné le modèle principal d'abord")
        return False

    if not manager.feedback_csv_path.exists():
        print(f"❌ Fichier de feedbacks manquant: {manager.feedback_csv_path}")
        print("💡 Aucun feedback à traiter pour le moment")
        return False

    print("✅ Tous les prérequis sont satisfaits")

    # Exécuter le fine-tuning complet
    success = manager.run_complete_finetuning()

    if success:
        print("\n🎉 Fine-tuning réussi!")
        print("💡 Vous pouvez maintenant remplacer les fichiers du modèle dans l'API")
    else:
        print("\n❌ Fine-tuning échoué")
        print("💡 Vérifiez les logs pour plus de détails")

    return success


def create_replacement_script(source_dir, target_dir="./model"):
    """
    Fonction utilitaire pour créer un script de remplacement automatique
    """
    script_content = f"""#!/bin/bash
# Script de remplacement automatique du modèle fine-tuné

echo "🔄 Remplacement du modèle par la version fine-tunée..."

# Sauvegarder l'ancien modèle
BACKUP_DIR="./model_backup_$(date +%Y%m%d_%H%M%S)"
echo "💾 Sauvegarde de l'ancien modèle dans $BACKUP_DIR"
cp -r {target_dir} $BACKUP_DIR

# Remplacer par le nouveau modèle
echo "📋 Copie du nouveau modèle depuis {source_dir}"
cp -r {source_dir}/* {target_dir}/

echo "✅ Remplacement terminé!"
echo "💡 Redémarrez l'API Docker pour utiliser le nouveau modèle"
echo "💡 En cas de problème, restaurez depuis: $BACKUP_DIR"
"""

    script_path = Path(source_dir) / "replace_model.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Rendre exécutable
    import os
    os.chmod(script_path, 0o755)

    print(f"📝 Script de remplacement créé: {script_path}")
    return script_path


# Classes utilitaires pour la gestion des feedbacks
class FeedbackAnalyzer:
    """
    Analyse les patterns dans les feedbacks pour améliorer le modèle
    """

    def __init__(self, feedback_csv_path):
        self.feedback_csv_path = Path(feedback_csv_path)

    def analyze_feedback_patterns(self):
        """
        Analyse les patterns des feedbacks négatifs
        """
        if not self.feedback_csv_path.exists():
            print("❌ Fichier de feedbacks non trouvé")
            return {}

        df = pd.read_csv(self.feedback_csv_path)

        # Séparer les feedbacks positifs et négatifs
        positive_feedbacks = df[df['user_satisfaction'] == 'yes']
        negative_feedbacks = df[df['user_satisfaction'] == 'no']

        analysis = {
            'total_feedbacks': len(df),
            'positive_count': len(positive_feedbacks),
            'negative_count': len(negative_feedbacks),
            'negative_ratio': len(negative_feedbacks) / len(df) if len(df) > 0 else 0,
            'patterns': {}
        }

        if len(negative_feedbacks) > 0:
            # Analyser les erreurs par type de prédiction
            prediction_errors = negative_feedbacks['predicted_class'].value_counts()
            analysis['patterns']['prediction_errors'] = prediction_errors.to_dict()

            # Analyser par langue
            if 'language_detected' in negative_feedbacks.columns:
                language_errors = negative_feedbacks['language_detected'].value_counts()
                analysis['patterns']['language_errors'] = language_errors.to_dict()

            # Analyser par niveau de confiance
            if 'predicted_probability' in negative_feedbacks.columns:
                neg_probs = negative_feedbacks['predicted_probability']
                analysis['patterns']['avg_confidence_errors'] = float(neg_probs.mean())
                analysis['patterns']['confidence_distribution'] = {
                    'low_confidence_errors': len(neg_probs[neg_probs < 0.6]),
                    'medium_confidence_errors': len(neg_probs[(neg_probs >= 0.6) & (neg_probs < 0.8)]),
                    'high_confidence_errors': len(neg_probs[neg_probs >= 0.8])
                }

        return analysis

    def print_analysis(self):
        """
        Affiche l'analyse des feedbacks
        """
        analysis = self.analyze_feedback_patterns()

        print("\n📊 ANALYSE DES FEEDBACKS")
        print("=" * 40)
        print(f"Total des feedbacks: {analysis['total_feedbacks']}")
        print(f"Feedbacks positifs: {analysis['positive_count']}")
        print(f"Feedbacks négatifs: {analysis['negative_count']}")
        print(f"Taux d'erreur: {analysis['negative_ratio']:.2%}")

        if analysis['patterns']:
            print(f"\n🔍 Patterns d'erreurs:")

            if 'prediction_errors' in analysis['patterns']:
                print(f"   Erreurs par prédiction:")
                for pred, count in analysis['patterns']['prediction_errors'].items():
                    print(f"     {pred}: {count}")

            if 'language_errors' in analysis['patterns']:
                print(f"   Erreurs par langue:")
                for lang, count in analysis['patterns']['language_errors'].items():
                    print(f"     {lang}: {count}")

            if 'confidence_distribution' in analysis['patterns']:
                conf_dist = analysis['patterns']['confidence_distribution']
                print(f"   Distribution des erreurs par confiance:")
                print(f"     Faible confiance (<60%): {conf_dist['low_confidence_errors']}")
                print(f"     Confiance moyenne (60-80%): {conf_dist['medium_confidence_errors']}")
                print(f"     Haute confiance (>80%): {conf_dist['high_confidence_errors']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            # Mode analyse des feedbacks
            analyzer = FeedbackAnalyzer("./data/user_feedbacks.csv")
            analyzer.print_analysis()
        elif sys.argv[1] == "help":
            print("📋 UTILISATION DU FINE-TUNING MANAGER")
            print("=" * 40)
            print("python traitement.py              # Exécuter le fine-tuning")
            print("python traitement.py analyze      # Analyser les feedbacks")
            print("python traitement.py help         # Afficher cette aide")
        else:
            print("❌ Argument non reconnu. Utilisez 'help' pour voir les options")
    else:
        # Mode fine-tuning principal
        main()