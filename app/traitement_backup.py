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
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import nltk
from nltk.corpus import stopwords
import shutil
import os

# Configuration pour la reproductibilité
tf.random.set_seed(42)
np.random.seed(42)


# Ligne ~27, remplacez la fonction convert_numpy_types par :
def convert_numpy_types(obj):
    """Convertit récursivement les types NumPy/Pandas en types Python natifs SANS tout convertir en string"""
    import numpy as np
    import pandas as pd

    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)  # GARDER comme int Python
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)  # GARDER comme float Python
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'item'):  # Scalaires NumPy
        try:
            # 🔧 CORRECTION CRITIQUE: Préserver le type
            value = obj.item()
            if isinstance(value, (int, float, bool)):
                return value  # Garder le type original
            else:
                return str(value)
        except:
            return str(obj)
    elif hasattr(obj, 'tolist'):  # Arrays NumPy
        try:
            return obj.tolist()
        except:
            return str(obj)
    else:
        # 🔧 CRITIQUE: Ne convertir en string QUE si nécessaire
        if isinstance(obj, (int, float, bool, str)):
            return obj  # Types Python natifs : ne pas toucher !
        else:
            return str(obj)  # Conversion en string seulement pour les types non supportés

class IndividualFeedbackRetrainingManager:
    """
    Gestionnaire qui RE-ENTRAÎNE le modèle pour CHAQUE feedback négatif individuel
    et déploie immédiatement si la correction fonctionne
    """

    def __init__(self, model_dir="./model/model_prod", data_dir="./data"):
        """
        Initialise le gestionnaire de re-entraînement par feedback individuel

        Args:
            model_dir: Répertoire contenant les artefacts du modèle de production
            data_dir: Répertoire contenant les données et feedbacks
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)

        # Chemins des artefacts du modèle de production
        self.model_path = self.model_dir / "best_lstm_model.keras"
        self.tokenizer_path = self.model_dir / "tokenizer.pkl"
        self.scaler_path = self.model_dir / "scaler.pkl"
        self.label_encoder_path = self.model_dir / "label_encoder.pkl"
        self.metadata_path = self.model_dir / "model_metadata.json"
        self.suspicious_words_path = self.model_dir / "suspicious_words.json"

        # Chemin du fichier de feedbacks
        self.feedback_csv_path = self.data_dir / "user_feedbacks.csv"

        # Log des re-entraînements individuels
        self.retraining_log_path = self.data_dir / "individual_retraining_log.json"

        # Artefacts du modèle
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.suspicious_words_set = set()

        # Configuration re-entraînement pour feedback individuel
        self.retrain_config = {
            'learning_rate': 0.001,  # Très faible pour stabilité
            'epochs': 10,  # Peu d'époques pour éviter overfitting
            'batch_size': 5,  # Un seul feedback
            'patience': 3,  # Patience réduite
            'base_sample_size': 50,  # Échantillon de support
            'validation_split': 0.3,  # Plus de validation
            # Pondération intelligente

            'feedback_weight': 6.0,  # Prioritaire mais pas excessif
            'support_weight': 1.0,  # Poids normal

            'safety_backup': True,
            'min_confidence_threshold': 0.7,  # Seuil de confiance minimum
            'max_gradient_norm': 1.0  # Gradient clipping
        }

        # Critères de déploiement pour feedback individuel
        self.deployment_criteria = {
            'require_feedback_corrected': True,  # Le feedback DOIT être corrigé
            'min_confidence': 0.6,  # Confiance minimum dans la correction
            'allow_slight_degradation': 0.01,  # 1% de dégradation max sur performance globale
            'safety_threshold': 0.80,  # 80% accuracy minimum
            'max_consecutive_failures': 3  # Maximum d'échecs consécutifs avant arrêt
        }

        # Compteurs
        self.consecutive_failures = 0
        self.total_retrainings = 0
        self.successful_deployments = 0

        # Charger stopwords
        self._setup_stopwords()

        print("🎯 IndividualFeedbackRetrainingManager initialisé")
        print(f"   APPROCHE: UN feedback négatif = UN re-entraînement")
        print(f"   DÉPLOIEMENT: Immédiat si la correction fonctionne")
        print(f"   SÉCURITÉ: Confiance ≥ {self.deployment_criteria['min_confidence']:.0%}")

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
        except Exception as e:
            print(f"⚠️ Erreur stopwords: {e}")
            self.stop_words = {
                'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'},
                'fr': {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais'}
            }

    def load_model_artifacts(self):
        """
        ÉTAPE 1: Charge tous les artefacts du modèle existant
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 1: CHARGEMENT DES ARTEFACTS DU MODÈLE")
        print("=" * 60)

        try:
            # Charger le modèle EN PREMIER
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")

            self.model = load_model(str(self.model_path))
            print(f"✅ Modèle chargé: {self.model_path}")

            # Détecter la longueur de séquence
            model_input_shape = self.model.inputs[0].shape
            actual_sequence_length = model_input_shape[1]
            print(f"🔍 Longueur de séquence détectée: {actual_sequence_length}")

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

            # 🔧 CORRECTION: Chargement sécurisé des métadonnées
            try:
                if self.metadata_path.exists():
                    with open(self.metadata_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            self.metadata = json.loads(content)
                            print("✅ Métadonnées chargées depuis le fichier")
                        else:
                            print("⚠️ Fichier métadonnées vide")
                            self.metadata = {}
                else:
                    print("⚠️ Fichier métadonnées non trouvé")
                    self.metadata = {}
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"⚠️ Erreur lecture métadonnées: {e}")
                print("🔧 Création de nouvelles métadonnées")
                self.metadata = {}

            # Créer/corriger la configuration
            if 'config' not in self.metadata:
                self.metadata['config'] = {}

            # Utiliser la longueur détectée du modèle
            self.metadata['config']['max_sequence_length'] = actual_sequence_length
            print(f"🔧 Longueur de séquence configurée: {actual_sequence_length}")

            # Charger les mots suspects
            if self.suspicious_words_path.exists():
                try:
                    with open(self.suspicious_words_path, 'r') as f:
                        suspicious_data = json.load(f)
                    self.suspicious_words_set = set(
                        suspicious_data.get('en', []) + suspicious_data.get('fr', [])
                    )
                    print(f"✅ Mots suspects chargés: {len(self.suspicious_words_set)}")
                except Exception as e:
                    print(f"⚠️ Erreur mots suspects: {e}")
                    self.suspicious_words_set = set()
            else:
                print("⚠️ Fichier mots suspects non trouvé")
                self.suspicious_words_set = set()

            print(f"\n📋 Configuration finale:")
            print(f"   max_sequence_length: {self.metadata['config']['max_sequence_length']}")
            print(f"   Classes: {list(self.label_encoder.classes_)}")

            return True

        except Exception as e:
            print(f"❌ Erreur chargement artefacts: {e}")
            import traceback
            traceback.print_exc()
            return False
    def get_next_unprocessed_feedback(self):
        """
        Récupère le PROCHAIN feedback négatif non traité (un seul à la fois)
        """
        print("\n" + "=" * 60)
        print("RECHERCHE DU PROCHAIN FEEDBACK NÉGATIF À TRAITER")
        print("=" * 60)

        if not self.feedback_csv_path.exists():
            print(f"❌ Fichier feedback non trouvé: {self.feedback_csv_path}")
            return None

        try:
            df = pd.read_csv(self.feedback_csv_path)
            print(f"📊 Total feedbacks: {len(df)}")

            # Filtrer UN SEUL feedback négatif non traité (le plus ancien)
            unprocessed_negative = df[
                (df['user_satisfaction'] == 'no') &
                (df['processed'] == False)
            ].copy()

            if len(unprocessed_negative) == 0:
                print("ℹ️ Aucun feedback négatif à traiter")
                return None

            # Prendre le PREMIER (plus ancien) feedback non traité
            feedback_row = unprocessed_negative.iloc[0]

            # Déterminer le vrai label (inverse de la prédiction erronée)
            predicted_class = feedback_row['predicted_class']
            if predicted_class in ['phishing', 'spam']:
                true_label = 'benign'
            else:
                true_label = 'phishing'

            feedback_data = {
                'id': feedback_row.name,  # Index dans le DataFrame
                'text': feedback_row['email_text'],
                'label': true_label,
                'language': feedback_row['language_detected'],
                'original_prediction': predicted_class,
                'confidence': feedback_row['predicted_probability'],
                'timestamp': feedback_row.get('timestamp', datetime.now().isoformat())
            }

            print(f"🎯 FEEDBACK SÉLECTIONNÉ POUR RE-ENTRAÎNEMENT:")
            print(f"   ID: {feedback_data['id']}")
            print(f"   Texte: {feedback_data['text'][:100]}...")
            print(f"   Prédiction erronée: {feedback_data['original_prediction']}")
            print(f"   Correction attendue: {feedback_data['label']}")
            print(f"   Confiance originale: {feedback_data['confidence']:.3f}")

            return feedback_data

        except Exception as e:
            print(f"❌ Erreur récupération feedback: {e}")
            return None

    def create_individual_training_dataset(self, feedback_data, dataset_path):
        """
        Crée un dataset d'entraînement avec LE feedback + échantillon de support
        """
        print(f"\n📊 CRÉATION DATASET POUR FEEDBACK INDIVIDUEL")
        print("=" * 50)

        try:
            # Dataset avec LE feedback à corriger
            feedback_df = pd.DataFrame([{
                'text': feedback_data['text'],
                'label': feedback_data['label'],
                'language': feedback_data['language'],
                'source': 'feedback',
                'weight': self.retrain_config['feedback_weight']  # Utiliser la config
            }])

            print(f"📝 Feedback individuel ajouté (poids: {self.retrain_config['feedback_weight']})")

            # Ajouter un échantillon de support pour stabilité
            base_sample_df = pd.DataFrame()
            if Path(dataset_path).exists():
                df = pd.read_csv(dataset_path)

                # Échantillon équilibré pour stabilité
                sample_size = self.retrain_config['base_sample_size']
                if len(df) > sample_size:
                    # Échantillonnage stratifié équilibré
                    balanced_sample = []
                    for label in df['label'].unique():
                        label_df = df[df['label'] == label]
                        label_sample_size = sample_size // len(df['label'].unique())

                        if len(label_df) >= label_sample_size:
                            label_sample = label_df.sample(n=label_sample_size, random_state=42)
                        else:
                            label_sample = label_df

                        balanced_sample.append(label_sample)

                    base_sample_df = pd.concat(balanced_sample, ignore_index=True)
                else:
                    base_sample_df = df.copy()

                base_sample_df = base_sample_df[['text', 'label', 'language']].copy()
                base_sample_df['source'] = 'dataset'
                base_sample_df['weight'] = self.retrain_config['support_weight']

                print(f"📊 Échantillon de support: {len(base_sample_df)} échantillons")

            # Combiner feedback + support
            if not base_sample_df.empty:
                combined_df = pd.concat([feedback_df, base_sample_df], ignore_index=True)
            else:
                combined_df = feedback_df.copy()

            print(f"🔗 Dataset final: {len(combined_df)} échantillons")
            print(f"   Distribution: {combined_df['label'].value_counts().to_dict()}")
            print(f"   📊 Ratio feedback/support: 1/{len(base_sample_df)}")

            return combined_df

        except Exception as e:
            print(f"❌ Erreur création dataset: {e}")
            return pd.DataFrame()

    def preprocess_text(self, text, language='en'):
        """Préprocesse le texte"""
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                      ' URL_TOKEN ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL_TOKEN ', text)
        text = re.sub(r'\b\d+\b', ' NUM_TOKEN ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()
        stop_words_lang = self.stop_words.get(language, self.stop_words['en'])
        filtered_tokens = [token for token in tokens if len(token) > 2 and token not in stop_words_lang]

        return ' '.join(filtered_tokens)

    def extract_numerical_features(self, texts):
        """Extrait les features numériques"""
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

    def prepare_training_data(self, combined_df):
        """Prépare les données pour le re-entraînement"""
        print(f"\n🔧 PRÉPARATION DONNÉES RE-ENTRAÎNEMENT")
        print("=" * 50)

        try:
            # Paramètres du modèle
            model_input_shape = self.model.inputs[0].shape
            actual_sequence_length = model_input_shape[1]

            embedding_layer = None
            for layer in self.model.layers:
                if 'embedding' in layer.name.lower():
                    embedding_layer = layer
                    break

            actual_vocab_size = embedding_layer.input_dim if embedding_layer else 10001
            max_vocab_id = actual_vocab_size - 1

            # Prétraitement
            processed_texts = []
            for _, row in combined_df.iterrows():
                processed_text = self.preprocess_text(row['text'], row.get('language', 'en'))
                processed_texts.append(processed_text)

            # Séquences
            sequences = self.tokenizer.texts_to_sequences(processed_texts)
            filtered_sequences = []
            for sequence in sequences:
                filtered_sequence = [token_id for token_id in sequence if token_id <= max_vocab_id]
                filtered_sequences.append(filtered_sequence)

            X_text = pad_sequences(filtered_sequences, maxlen=actual_sequence_length, padding='post', truncating='post')

            # Features numériques
            X_num = self.extract_numerical_features(combined_df['text'])
            X_num = self.scaler.transform(X_num)

            # Labels
            y = self.label_encoder.transform(combined_df['label'])

            # Poids d'échantillons
            sample_weights = combined_df.get('weight', pd.Series([1.0] * len(combined_df))).values

            # Division adaptée aux petits datasets
            if len(combined_df) > 4:  # Assez pour split
                X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val, weights_train, weights_val = train_test_split(
                    X_text, X_num, y, sample_weights,
                    test_size=self.retrain_config['validation_split'],
                    random_state=42,
                    stratify=y if len(np.unique(y)) > 1 else None
                )
            else:
                # Dataset trop petit, utiliser tout pour l'entraînement
                X_text_train, X_text_val = X_text, X_text
                X_num_train, X_num_val = X_num, X_num
                y_train, y_val = y, y
                weights_train, weights_val = sample_weights, sample_weights

            print(f"✅ Données préparées:")
            print(f"   Train: {len(X_text_train)} échantillons")
            print(f"   Validation: {len(X_text_val)} échantillons")

            return {
                'X_text_train': X_text_train, 'X_text_val': X_text_val,
                'X_num_train': X_num_train, 'X_num_val': X_num_val,
                'y_train': y_train, 'y_val': y_val,
                'weights_train': weights_train, 'weights_val': weights_val
            }

        except Exception as e:
            print(f"❌ Erreur préparation données: {e}")
            return None

    # 1. CORRECTION DE create_backup_for_feedback
    def create_backup_for_feedback(self, feedback_id):
        """Crée une sauvegarde avant re-entraînement avec conversion de types NumPy"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Conversion explicite du feedback_id en int Python
        clean_feedback_id = int(feedback_id)

        backup_dir = self.data_dir / f"backup_feedback_{clean_feedback_id}_{timestamp}"

        print(f"\n💾 SAUVEGARDE AVANT RE-ENTRAÎNEMENT")
        print(f"   Feedback ID: {clean_feedback_id}")
        print(f"   Destination: {backup_dir}")

        try:
            backup_dir.mkdir(parents=True, exist_ok=True)

            artifacts_to_backup = [
                'best_lstm_model.keras',
                'tokenizer.pkl',
                'scaler.pkl',
                'label_encoder.pkl',
                'model_metadata.json',
                'suspicious_words.json'
            ]

            for artifact in artifacts_to_backup:
                source = self.model_dir / artifact
                dest = backup_dir / artifact
                if source.exists():
                    shutil.copy2(source, dest)

            # Métadonnées avec conversion de types
            backup_metadata = {
                'backup_timestamp': timestamp,
                'feedback_id': clean_feedback_id,
                'backup_reason': f'before_individual_retraining_{clean_feedback_id}'
            }

            # 🔧 CORRECTION CRITIQUE: Nettoyer self.metadata avant sérialisation
            if self.metadata:
                try:
                    backup_metadata['original_model_metadata'] = convert_numpy_types(self.metadata)
                except Exception as e:
                    print(f"⚠️ Erreur conversion métadonnées: {e}")
                    backup_metadata['original_model_metadata'] = {}

            # Sauvegarder avec conversion par défaut en string pour les types non supportés
            with open(backup_dir / 'backup_metadata.json', 'w') as f:
                json.dump(backup_metadata, f, indent=2, default=str)

            print(f"✅ Sauvegarde créée: {backup_dir}")
            return backup_dir

        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            # Créer au moins le dossier pour continuer
            try:
                backup_dir.mkdir(parents=True, exist_ok=True)
                return backup_dir
            except:
                return None
    def perform_individual_retraining(self, training_data, feedback_data):
        """
        RE-ENTRAÎNE le modèle pour corriger LE feedback spécifique
        """
        print(f"\n🎯 RE-ENTRAÎNEMENT POUR FEEDBACK #{feedback_data['id']}")
        print("=" * 60)
        print(f"🎯 OBJECTIF: Corriger '{feedback_data['original_prediction']}' → '{feedback_data['label']}'")

        try:
            # Créer une copie du modèle
            model_retrained = tf.keras.models.clone_model(self.model)
            model_retrained.set_weights(self.model.get_weights())

            # Configuration optimisée
            print(f"🔧 Configuration re-entraînement:")
            for key, value in self.retrain_config.items():
                if key not in ['base_sample_size', 'safety_backup']:
                    print(f"   {key}: {value}")

            # Optimiseur conservateur
            optimizer = Adam(learning_rate=self.retrain_config['learning_rate'])
            model_retrained.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Callbacks conservateurs
            callbacks = [
                EarlyStopping(
                    patience=self.retrain_config['patience'],
                    restore_best_weights=True,
                    monitor='loss',
                    min_delta=0.0001
                )
            ]

            # Extraire les données
            X_text_train = training_data['X_text_train']
            X_text_val = training_data['X_text_val']
            X_num_train = training_data['X_num_train']
            X_num_val = training_data['X_num_val']
            y_train = training_data['y_train']
            y_val = training_data['y_val']
            weights_train = training_data['weights_train']

            # Re-entraînement ciblé
            print(f"\n🚀 Début du re-entraînement ciblé...")

            history = model_retrained.fit(
                [X_text_train, X_num_train], y_train,
                validation_data=([X_text_val, X_num_val], y_val),
                batch_size=self.retrain_config['batch_size'],
                epochs=self.retrain_config['epochs'],
                sample_weight=weights_train,
                callbacks=callbacks,
                verbose=1
            )

            print(f"✅ Re-entraînement terminé")

            return model_retrained, history

        except Exception as e:
            print(f"❌ Erreur re-entraînement: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def validate_feedback_correction(self, retrained_model, feedback_data):
        """
        Valide que LE feedback spécifique est maintenant corrigé
        """
        print(f"\n🔍 VALIDATION DE LA CORRECTION DU FEEDBACK #{feedback_data['id']}")
        print("=" * 60)

        try:
            # Préparer le texte du feedback pour test
            processed_text = self.preprocess_text(feedback_data['text'], feedback_data.get('language', 'en'))

            # Paramètres du modèle
            model_input_shape = retrained_model.inputs[0].shape
            actual_sequence_length = model_input_shape[1]

            embedding_layer = None
            for layer in retrained_model.layers:
                if 'embedding' in layer.name.lower():
                    embedding_layer = layer
                    break
            actual_vocab_size = embedding_layer.input_dim if embedding_layer else 10001
            max_vocab_id = actual_vocab_size - 1

            # Créer séquence
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            if sequence[0]:
                sequence[0] = [token_id for token_id in sequence[0] if token_id <= max_vocab_id]

            X_text = pad_sequences(sequence, maxlen=actual_sequence_length, padding='post', truncating='post')

            # Features numériques
            X_num = self.extract_numerical_features([feedback_data['text']])
            X_num = self.scaler.transform(X_num)

            # Test avec le modèle original
            original_pred_proba = self.model.predict([X_text, X_num], verbose=0)[0][0]
            original_pred_class_int = int(original_pred_proba > 0.5)
            original_pred_class = self.label_encoder.inverse_transform([original_pred_class_int])[0]

            # Test avec le modèle re-entraîné
            new_pred_proba = retrained_model.predict([X_text, X_num], verbose=0)[0][0]
            new_pred_class_int = int(new_pred_proba > 0.5)
            new_pred_class = self.label_encoder.inverse_transform([new_pred_class_int])[0]

            # Calculer la confiance
            confidence_score = abs(new_pred_proba - 0.5) * 2  # 0 à 1

            # Vérifier si la correction a fonctionné
            expected_label = feedback_data['label']
            is_corrected = (new_pred_class == expected_label)
            confidence_sufficient = confidence_score >= self.deployment_criteria['min_confidence']

            print(f"📊 RÉSULTATS DE LA VALIDATION:")
            print(f"   Texte testé: {feedback_data['text'][:100]}...")
            print(f"   Label attendu: {expected_label}")
            print(f"   Prédiction originale: {original_pred_class} (prob: {original_pred_proba:.3f})")
            print(f"   Nouvelle prédiction: {new_pred_class} (prob: {new_pred_proba:.3f})")
            print(f"   Confiance: {confidence_score:.3f}")
            print(f"   Correction réussie: {'✅' if is_corrected else '❌'}")
            print(f"   Confiance suffisante: {'✅' if confidence_sufficient else '❌'}")

            validation_result = {
                'feedback_id': feedback_data['id'],
                'expected_label': expected_label,
                'original_prediction': original_pred_class,
                'original_probability': float(original_pred_proba),
                'new_prediction': new_pred_class,
                'new_probability': float(new_pred_proba),
                'confidence_score': confidence_score,
                'is_corrected': is_corrected,
                'confidence_sufficient': confidence_sufficient,
                'validation_passed': is_corrected and confidence_sufficient
            }

            if validation_result['validation_passed']:
                print(f"\n🎉 VALIDATION RÉUSSIE!")
                print(f"   Le feedback a été corrigé avec confiance suffisante")
                self.consecutive_failures = 0  # Reset compteur d'échecs
            else:
                print(f"\n❌ VALIDATION ÉCHOUÉE!")
                if not is_corrected:
                    print(f"   Le modèle n'a pas appris la correction")
                if not confidence_sufficient:
                    print(
                        f"   Confiance insuffisante ({confidence_score:.3f} < {self.deployment_criteria['min_confidence']:.3f})")
                self.consecutive_failures += 1

            return validation_result

        except Exception as e:
            print(f"❌ Erreur validation correction: {e}")
            self.consecutive_failures += 1
            return {
                'feedback_id': feedback_data['id'],
                'validation_passed': False,
                'error': str(e)
            }

    def trigger_api_reload(self):
        """Déclenche le rechargement du modèle dans l'API via un appel HTTP"""
        try:
            print(f"\n🔄 DÉCLENCHEMENT DU RECHARGEMENT AUTOMATIQUE...")

            api_urls = [
                "http://localhost:8000",
                "http://127.0.0.1:8000",
                "http://fastapi:8000"
            ]

            for api_url in api_urls:
                try:
                    print(f"   Tentative: {api_url}/reload-model")

                    import subprocess
                    import json

                    curl_command = [
                        'curl', '-s', '-X', 'POST',
                        f'{api_url}/reload-model',
                        '-H', 'Content-Type: application/json',
                        '--max-time', '10'
                    ]

                    result = subprocess.run(curl_command, capture_output=True, text=True, timeout=15)

                    if result.returncode == 0:
                        try:
                            response_data = json.loads(result.stdout)
                            if response_data.get('status') == 'success':
                                print(f"✅ RECHARGEMENT RÉUSSI via {api_url}!")
                                print(f"   Message: {response_data.get('message', 'OK')}")
                                print(f"   Version: {response_data.get('model_version', 'unknown')}")
                                return True
                        except json.JSONDecodeError:
                            print(f"   ⚠️ Réponse non-JSON: {result.stdout[:100]}")
                    else:
                        print(f"   ❌ Erreur curl (code {result.returncode}): {result.stderr}")

                except subprocess.TimeoutExpired:
                    print(f"   ❌ Timeout sur {api_url}")
                    continue
                except Exception as e:
                    print(f"   ❌ Erreur: {e}")
                    continue

            print(f"⚠️ Rechargement automatique échoué sur toutes les URLs")
            print(f"💡 Le modèle a été déployé mais nécessite un rechargement manuel")
            return False

        except Exception as e:
            print(f"❌ Erreur rechargement automatique: {e}")
            return False

    def deploy_retrained_model(self, retrained_model, feedback_data, backup_dir, validation_result):
        """Déploie immédiatement le modèle re-entraîné ET déclenche le rechargement de l'API"""
        print(f"\n🚀 DÉPLOIEMENT IMMÉDIAT DU MODÈLE RE-ENTRAÎNÉ")
        print("=" * 50)

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Sauvegarder temporairement le nouveau modèle
            temp_model_path = self.data_dir / f"retrained_model_feedback_{feedback_data['id']}_{timestamp}.keras"
            retrained_model.save(str(temp_model_path))

            # Remplacer le modèle de production
            production_model_path = self.model_dir / "best_lstm_model.keras"
            shutil.copy2(temp_model_path, production_model_path)

            # 🔧 CORRECTION CRITIQUE: Gestion sûre des métadonnées
            if self.metadata:
                updated_metadata = convert_numpy_types(self.metadata.copy())
            else:
                updated_metadata = {}

            # 🔧 CORRECTION: Conversion sûre de model_version
            current_version = updated_metadata.get('model_version', 1)
            if isinstance(current_version, str):
                try:
                    current_version = int(current_version)
                except (ValueError, TypeError):
                    current_version = 1

            new_version = current_version + 1

            # Nettoyer TOUS les types avant mise à jour
            clean_validation_result = convert_numpy_types(validation_result)
            clean_retraining_config = convert_numpy_types(self.retrain_config)

            # Mettre à jour avec des types Python natifs
            updated_metadata.update({
                'last_individual_retraining': timestamp,
                'last_feedback_processed': int(feedback_data['id']),
                'retraining_config': clean_retraining_config,
                'model_version': new_version,  # Déjà un int Python
                'backup_location': str(backup_dir),
                'deployment_method': 'individual_feedback_retraining',
                'total_individual_retrainings': int(self.total_retrainings + 1),
                'successful_deployments': int(self.successful_deployments + 1),
                'validation_result': clean_validation_result,
                'deployment_timestamp': timestamp,
                'auto_reload_trigger': True
            })

            # Nettoyer encore une fois les métadonnées finales
            final_clean_metadata = convert_numpy_types(updated_metadata)

            # Sauvegarder les métadonnées nettoyées
            with open(self.model_dir / "model_metadata.json", 'w') as f:
                json.dump(final_clean_metadata, f, indent=2, default=str)

            # Nettoyer le fichier temporaire
            temp_model_path.unlink()

            # Mettre à jour les compteurs
            self.total_retrainings += 1
            self.successful_deployments += 1

            # Mettre à jour self.metadata avec la version nettoyée
            self.metadata = final_clean_metadata

            # Recharger le modèle en mémoire
            self.model = load_model(str(production_model_path))

            print(f"✅ DÉPLOIEMENT RÉUSSI!")
            print(f"   Version du modèle: {new_version}")
            print(f"   Feedback traité: #{feedback_data['id']}")
            print(f"   Re-entraînements totaux: {self.total_retrainings}")
            print(f"   Déploiements réussis: {self.successful_deployments}")
            print(f"   Sauvegarde: {backup_dir}")

            # Déclencher le rechargement automatique de l'API
            self.trigger_api_reload()

            return True

        except Exception as e:
            print(f"❌ Erreur déploiement: {e}")
            import traceback
            traceback.print_exc()
            return False
    def log_retraining_attempt(self, feedback_data, validation_result, deployed, backup_dir):
        """Enregistre la tentative de re-entraînement"""
        try:
            # Nettoyer tous les types NumPy
            clean_feedback_data = convert_numpy_types({
                'text': feedback_data['text'][:200],
                'original_prediction': feedback_data['original_prediction'],
                'expected_correction': feedback_data['label'],
                'original_confidence': float(feedback_data['confidence'])
            })

            clean_validation_result = convert_numpy_types(validation_result)

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'feedback_id': int(feedback_data['id']),
                'feedback_details': clean_feedback_data,
                'validation_result': clean_validation_result,
                'deployed': bool(deployed),
                'backup_location': str(backup_dir),
                'total_retrainings': int(self.total_retrainings),
                'consecutive_failures': int(self.consecutive_failures)
            }

            # Charger le log existant
            retraining_log = []
            if self.retraining_log_path.exists():
                try:
                    with open(self.retraining_log_path, 'r') as f:
                        retraining_log = json.load(f)
                except:
                    retraining_log = []

            # Ajouter la nouvelle entrée
            retraining_log.append(log_entry)

            # Limiter à 50 entrées
            if len(retraining_log) > 50:
                retraining_log = retraining_log[-50:]

            # Sauvegarder
            self.data_dir.mkdir(exist_ok=True)
            with open(self.retraining_log_path, 'w') as f:
                json.dump(retraining_log, f, indent=2)

            print(f"📝 Tentative de re-entraînement enregistrée")

        except Exception as e:
            print(f"⚠️ Erreur enregistrement log: {e}")
            import traceback
            traceback.print_exc()

    def mark_feedback_as_processed(self, feedback_id, deployed=True, validation_result=None):
        """Marque LE feedback comme traité"""
        print(f"\n📝 MARQUAGE DU FEEDBACK #{feedback_id} COMME TRAITÉ")
        print("=" * 50)

        try:
            df = pd.read_csv(self.feedback_csv_path)

            # Marquer le feedback spécifique
            df.loc[feedback_id, 'processed'] = True
            df.loc[feedback_id, 'processed_at'] = datetime.now().isoformat()
            df.loc[feedback_id, 'deployed'] = bool(deployed)
            df.loc[feedback_id, 'retraining_method'] = 'individual_feedback'

            if validation_result:
                # Convertir les types NumPy
                df.loc[feedback_id, 'correction_validated'] = bool(validation_result.get('validation_passed', False))
                df.loc[feedback_id, 'new_prediction'] = str(validation_result.get('new_prediction', ''))
                df.loc[feedback_id, 'confidence_score'] = float(validation_result.get('confidence_score', 0.0))

            # Sauvegarder
            df.to_csv(self.feedback_csv_path, index=False)

            status = "et déployé" if deployed else "mais non déployé"
            print(f"✅ Feedback #{feedback_id} marqué comme traité {status}")
            return True

        except Exception as e:
            print(f"❌ Erreur marquage feedback: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_single_feedback(self, dataset_path="./data/full_merged_dataset_fr_en_spam.csv"):
        """FONCTION PRINCIPALE: Traite UN SEUL feedback avec re-entraînement immédiat"""
        print("\n" + "🎯" * 50)
        print("TRAITEMENT D'UN FEEDBACK INDIVIDUEL")
        print("RE-ENTRAÎNEMENT + VALIDATION + DÉPLOIEMENT IMMÉDIAT")
        print("🎯" * 50)

        start_time = datetime.now()

        # Étape 1: Charger les artefacts
        if not self.load_model_artifacts():
            print("❌ Échec du chargement des artefacts")
            return False

        # Étape 2: Récupérer le prochain feedback
        feedback_data = self.get_next_unprocessed_feedback()
        if feedback_data is None:
            print("ℹ️ Aucun feedback à traiter")
            return False

        # Vérifier les échecs consécutifs
        if self.consecutive_failures >= self.deployment_criteria['max_consecutive_failures']:
            print(f"🚨 ARRÊT: {self.consecutive_failures} échecs consécutifs")
            print("💡 Le modèle semble avoir des difficultés d'apprentissage")
            return False

        print(f"🎯 Traitement du feedback #{feedback_data['id']}")

        # Étape 3: Créer une sauvegarde
        backup_dir = self.create_backup_for_feedback(feedback_data['id'])
        if backup_dir is None:
            print("❌ Impossible de créer la sauvegarde - Arrêt")
            return False

        # Étape 4: Créer le dataset d'entraînement
        training_dataset = self.create_individual_training_dataset(feedback_data, dataset_path)
        if training_dataset.empty:
            print("❌ Impossible de créer le dataset d'entraînement")
            return False

        # Étape 5: Préparer les données
        training_data = self.prepare_training_data(training_dataset)
        if training_data is None:
            print("❌ Échec de la préparation des données")
            return False

        # Étape 6: Re-entraînement ciblé
        retrained_model, history = self.perform_individual_retraining(training_data, feedback_data)
        if retrained_model is None:
            print("❌ Échec du re-entraînement")
            validation_result = {'validation_passed': False, 'error': 'retraining_failed'}
            self.mark_feedback_as_processed(feedback_data['id'], deployed=False, validation_result=validation_result)
            self.log_retraining_attempt(feedback_data, validation_result, False, backup_dir)
            return False

        # Étape 7: Validation de la correction
        validation_result = self.validate_feedback_correction(retrained_model, feedback_data)

        # Étape 8: Décision de déploiement
        should_deploy = validation_result['validation_passed']
        deployment_success = False

        if should_deploy:
            deployment_success = self.deploy_retrained_model(retrained_model, feedback_data, backup_dir,
                                                             validation_result)
        else:
            print(f"\n🛑 PAS DE DÉPLOIEMENT")
            print(f"   Le feedback n'a pas été corrigé correctement")

        # Étape 9: Marquer comme traité
        self.mark_feedback_as_processed(feedback_data['id'], deployed=deployment_success,
                                        validation_result=validation_result)

        # Étape 10: Log de la tentative
        self.log_retraining_attempt(feedback_data, validation_result, deployment_success, backup_dir)

        # Résumé
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n🎉 RÉSUMÉ DU TRAITEMENT INDIVIDUEL")
        print("=" * 50)
        print(f"⏱️ Durée: {duration}")
        print(f"🎯 Feedback traité: #{feedback_data['id']}")
        print(f"🔧 Prédiction originale: {feedback_data['original_prediction']}")
        print(f"✅ Correction attendue: {feedback_data['label']}")

        if validation_result and 'new_prediction' in validation_result:
            print(f"🔄 Nouvelle prédiction: {validation_result['new_prediction']}")
            print(f"📊 Confiance: {validation_result.get('confidence_score', 0):.3f}")

        if deployment_success:
            print(f"\n🚀 RÉSULTAT: MODÈLE RE-ENTRAÎNÉ ET DÉPLOYÉ!")
            print(f"   Le feedback a été corrigé avec succès")
            print(f"   Version du modèle: {self.total_retrainings + 1}")
            print(f"💾 Sauvegarde: {backup_dir}")
            print(f"🔄 L'API devrait maintenant utiliser le nouveau modèle automatiquement")
        else:
            print(f"\n🛑 RÉSULTAT: Modèle original conservé")
            print(f"💡 Le re-entraînement n'a pas réussi à corriger le feedback")
            print(f"📊 Échecs consécutifs: {self.consecutive_failures}")

        return deployment_success

    def run_continuous_individual_processing(self, dataset_path="./data/test_dataset.csv", max_iterations=None):
        """Traite en continu CHAQUE feedback individuellement"""
        print("\n" + "🔄" * 50)
        print("TRAITEMENT CONTINU PAR FEEDBACK INDIVIDUEL")
        print("CHAQUE FEEDBACK = UN RE-ENTRAÎNEMENT + UN DÉPLOIEMENT POTENTIEL")
        print("🔄" * 50)

        processed_count = 0
        deployed_count = 0
        iteration = 0

        while True:
            iteration += 1
            print(f"\n{'=' * 20} ITÉRATION {iteration} {'=' * 20}")

            if max_iterations and iteration > max_iterations:
                print(f"🛑 Limite d'itérations atteinte ({max_iterations})")
                break

            # Vérifier les échecs consécutifs
            if self.consecutive_failures >= self.deployment_criteria['max_consecutive_failures']:
                print(f"🚨 ARRÊT: Trop d'échecs consécutifs ({self.consecutive_failures})")
                break

            # Traiter le prochain feedback
            success = self.process_single_feedback(dataset_path)

            if success is True:
                processed_count += 1
                deployed_count += 1
                print(f"✅ Feedback traité et modèle déployé!")
            elif success is False and self.get_next_unprocessed_feedback() is not None:
                processed_count += 1
                print(f"⚠️ Feedback traité mais modèle non déployé")
            else:
                print(f"ℹ️ Aucun feedback disponible - Arrêt du traitement")
                break

            print(f"\n📊 Bilan à ce stade:")
            print(f"   Feedbacks traités: {processed_count}")
            print(f"   Modèles déployés: {deployed_count}")
            print(
                f"   Taux de succès: {deployed_count / processed_count * 100:.1f}%" if processed_count > 0 else "   Taux de succès: 0%")
            print(f"   Échecs consécutifs: {self.consecutive_failures}")

        print(f"\n🎉 TRAITEMENT CONTINU TERMINÉ")
        print(f"📊 Statistiques finales:")
        print(f"   Total feedbacks traités: {processed_count}")
        print(f"   Total modèles déployés: {deployed_count}")
        print(
            f"   Taux de succès: {deployed_count / processed_count * 100:.1f}%" if processed_count > 0 else "   Taux de succès: 0%")
        print(f"   Re-entraînements totaux: {self.total_retrainings}")

        return processed_count, deployed_count


def main():
    """Fonction principale pour le re-entraînement par feedback individuel"""
    print("🚀 RE-ENTRAÎNEMENT PAR FEEDBACK INDIVIDUEL")
    print("=" * 60)
    print("🎯 PHILOSOPHIE: Chaque feedback négatif = Un re-entraînement")

    # Initialiser le gestionnaire
    manager = IndividualFeedbackRetrainingManager(
        model_dir="./model/model_prod",
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
        return False

    if not manager.feedback_csv_path.exists():
        print(f"❌ Fichier de feedbacks manquant: {manager.feedback_csv_path}")
        return False

    print("✅ Tous les prérequis sont satisfaits")

    # Mode automatique : traite automatiquement UN feedback
    print("\n🎯 MODE AUTOMATIQUE : Traitement d'un feedback individuel")
    success = manager.process_single_feedback()
    return success


class FeedbackRetrainingMonitor:
    """Moniteur pour le re-entraînement par feedback individuel"""

    def __init__(self, retraining_log_path):
        self.retraining_log_path = Path(retraining_log_path)

    def analyze_retraining_performance(self):
        """Analyse les performances du re-entraînement"""
        if not self.retraining_log_path.exists():
            print("❌ Aucun log de re-entraînement disponible")
            return None

        try:
            with open(self.retraining_log_path, 'r') as f:
                retraining_log = json.load(f)

            if not retraining_log:
                print("❌ Log de re-entraînement vide")
                return None

            print("\n📊 ANALYSE DES RE-ENTRAÎNEMENTS INDIVIDUELS")
            print("=" * 60)

            total_attempts = len(retraining_log)
            successful_deployments = len([entry for entry in retraining_log if entry['deployed']])
            success_rate = successful_deployments / total_attempts if total_attempts > 0 else 0

            print(f"📈 Tentatives totales: {total_attempts}")
            print(f"✅ Déploiements réussis: {successful_deployments}")
            print(f"📊 Taux de succès: {success_rate:.1%}")

            # Analyse des échecs
            failed_attempts = [entry for entry in retraining_log if not entry['deployed']]
            if failed_attempts:
                print(f"❌ Échecs: {len(failed_attempts)}")

                # Raisons d'échec
                correction_failures = len([entry for entry in failed_attempts
                                           if not entry['validation_result'].get('is_corrected', False)])
                confidence_failures = len([entry for entry in failed_attempts
                                           if entry['validation_result'].get('is_corrected', False)
                                           and not entry['validation_result'].get('confidence_sufficient', False)])

                print(f"   Échecs de correction: {correction_failures}")
                print(f"   Échecs de confiance: {confidence_failures}")

            # Tendance récente
            recent_attempts = retraining_log[-10:] if len(retraining_log) >= 10 else retraining_log
            recent_success_rate = len([entry for entry in recent_attempts if entry['deployed']]) / len(recent_attempts)

            print(f"📈 Tendance récente (10 derniers): {recent_success_rate:.1%}")

            # Patterns de feedback
            feedback_types = {}
            for entry in retraining_log:
                original_pred = entry['feedback_details']['original_prediction']
                expected_correction = entry['feedback_details']['expected_correction']
                pattern = f"{original_pred} → {expected_correction}"
                feedback_types[pattern] = feedback_types.get(pattern, 0) + 1

            print(f"\n🔍 Patterns de correction les plus fréquents:")
            for pattern, count in sorted(feedback_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {pattern}: {count} fois")

            return {
                'total_attempts': total_attempts,
                'successful_deployments': successful_deployments,
                'success_rate': success_rate,
                'recent_success_rate': recent_success_rate,
                'feedback_patterns': feedback_types
            }

        except Exception as e:
            print(f"❌ Erreur analyse: {e}")
            return None


def restore_from_backup(backup_dir, model_dir="./model/model_prod"):
    """Fonction utilitaire pour restaurer un modèle depuis une sauvegarde"""
    backup_path = Path(backup_dir)
    model_path = Path(model_dir)

    if not backup_path.exists():
        print(f"❌ Sauvegarde non trouvée: {backup_path}")
        return False

    try:
        print(f"🔄 Restauration depuis: {backup_path}")

        # Sauvegarder l'état actuel
        current_backup = model_path.parent / f"current_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(model_path, current_backup)
        print(f"💾 État actuel sauvegardé: {current_backup}")

        # Restaurer depuis la sauvegarde
        for item in backup_path.iterdir():
            if item.is_file() and item.name != 'backup_metadata.json':
                dest = model_path / item.name
                shutil.copy2(item, dest)
                print(f"✅ {item.name} restauré")

        print(f"🎉 Restauration terminée avec succès!")
        return True

    except Exception as e:
        print(f"❌ Erreur lors de la restauration: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "auto":
            # Mode automatique - traite un feedback
            manager = IndividualFeedbackRetrainingManager()
            success = manager.process_single_feedback()
            sys.exit(0 if success else 1)

        elif command == "continuous":
            # Mode continu - traite tous les feedbacks individuellement
            manager = IndividualFeedbackRetrainingManager()
            try:
                processed, deployed = manager.run_continuous_individual_processing()
                print(f"✅ Traitement terminé: {processed} traités, {deployed} déployés")
                sys.exit(0)
            except KeyboardInterrupt:
                print("\n👋 Arrêté par l'utilisateur")
                sys.exit(0)

        elif command == "monitor":
            # Mode monitoring
            monitor = FeedbackRetrainingMonitor("./data/individual_retraining_log.json")
            monitor.analyze_retraining_performance()
            sys.exit(0)

        elif command == "restore":
            # Mode restauration
            if len(sys.argv) > 2:
                backup_dir = sys.argv[2]
                restore_from_backup(backup_dir)
            else:
                print("❌ Usage: python traitement.py restore <backup_directory>")
            sys.exit(0)

        elif command == "help":
            print("📋 UTILISATION DU RE-ENTRAÎNEMENT INDIVIDUEL:")
            print("=" * 60)
            print("python traitement.py                    # Mode automatique (traite 1 feedback)")
            print("python traitement.py auto               # Traiter un feedback")
            print("python traitement.py continuous         # Traiter tous individuellement")
            print("python traitement.py monitor            # Monitoring des performances")
            print("python traitement.py restore <backup>   # Restaurer depuis sauvegarde")
            print("python traitement.py help               # Afficher cette aide")
            print("\n🎯 Le re-entraînement individuel effectue:")
            print("   • Prend le prochain feedback négatif non traité")
            print("   • Re-entraîne le modèle spécifiquement pour ce feedback")
            print("   • Valide que la correction fonctionne")
            print("   • Déploie immédiatement si la validation réussit")
            print("   • Déclenche automatiquement le rechargement de l'API")
            print("\n⚖️ Avantages:")
            print("   • Correction immédiate des erreurs")
            print("   • Apprentissage ciblé sur chaque problème")
            print("   • Déploiement rapide des améliorations")
            print("   • Adaptation continue du modèle")
            print("   • Rechargement automatique sans redémarrage")
            print("\n🎯 PHILOSOPHIE:")
            print("   Chaque feedback négatif = Une opportunité d'amélioration immédiate")

        else:
            print("❌ Commande non reconnue. Utilisez 'help' pour voir les options")
    else:
        # Mode automatique par défaut
        main()