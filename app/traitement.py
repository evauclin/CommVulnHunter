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


class AdvancedFineTuningManager:
    """
    Gestionnaire avancé pour le fine-tuning avec comparaison et déploiement automatique
    """

    def __init__(self, model_dir="./model/model_prod", data_dir="./data"):
        """
        Initialise le gestionnaire de fine-tuning avancé

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

        # Artefacts du modèle
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = None
        self.suspicious_words_set = set()

        # Configuration fine-tuning améliorée
        self.finetune_config = {
            'learning_rate': 0.01,  # Learning rate très faible pour éviter la dérive
            'epochs': 10,  # Peu d'époques pour éviter l'overfitting
            'batch_size': 16,  # Petit batch size pour plus de stabilité
            'patience': 5,  # Patience très réduite
            'validation_split': 0.25,  # Split plus important pour validation
            'sample_size': 500,  # Échantillon plus grand pour meilleure représentativité
            'min_improvement_threshold': 0.005,  # Amélioration minimale requise (0.5%)
            'safety_backup': True,  # Sauvegarde de sécurité
            'evaluation_dataset_size': None  # Utiliser tout le dataset pour l'évaluation
        }

        # Seuils de décision pour le déploiement
        self.deployment_criteria = {
            'primary_metrics': ['f1', 'accuracy'],  # Métriques principales
            'min_improvement': 0.005,  # Amélioration minimale requise
            'allow_slight_degradation': 0.002,  # Dégradation acceptable sur certaines métriques
            'require_significant_improvement': True,  # Exiger une amélioration significative
            'safety_checks': True  # Vérifications de sécurité
        }

        # Charger stopwords
        self._setup_stopwords()

        print("🎯 AdvancedFineTuningManager initialisé")
        print(f"   Model dir: {self.model_dir}")
        print(f"   Data dir: {self.data_dir}")
        print(
            f"   Critères de déploiement: amélioration minimale {self.deployment_criteria['min_improvement'] * 100:.1f}%")

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
            self.stop_words = {
                'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'},
                'fr': {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais'}
            }

    def load_model_artifacts(self):
        """
        ÉTAPE 1: Charge tous les artefacts du modèle existant
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 1: CHARGEMENT DES ARTEFACTS DU MODÈLE DE PRODUCTION")
        print("=" * 60)

        try:
            # Charger le modèle EN PREMIER pour obtenir la vraie longueur de séquence
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")

            self.model = load_model(str(self.model_path))
            print(f"✅ Modèle de production chargé: {self.model_path}")

            # Détecter la vraie longueur de séquence depuis le modèle
            model_input_shape = self.model.inputs[0].shape
            actual_sequence_length = model_input_shape[1]
            print(f"🔍 Longueur de séquence détectée: {actual_sequence_length}")

            # Charger les autres artefacts
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Tokenizer chargé (vocab: {len(self.tokenizer.word_index)})")

            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Scaler chargé")

            with open(self.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Label encoder chargé: {self.label_encoder.classes_}")

            # Charger les métadonnées
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)

            # Corriger la longueur de séquence dans les métadonnées
            if 'config' not in self.metadata:
                self.metadata['config'] = {}
            self.metadata['config']['max_sequence_length'] = actual_sequence_length

            # Charger les mots suspects
            if self.suspicious_words_path.exists():
                with open(self.suspicious_words_path, 'r') as f:
                    suspicious_data = json.load(f)
                self.suspicious_words_set = set(
                    suspicious_data.get('en', []) + suspicious_data.get('fr', [])
                )
                print(f"✅ Mots suspects chargés: {len(self.suspicious_words_set)}")

            print(f"\n📋 Configuration du modèle de production:")
            print(f"   max_sequence_length: {self.metadata['config']['max_sequence_length']}")
            print(f"   classes: {self.metadata.get('classes', self.label_encoder.classes_)}")

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

            # Analyser les patterns d'erreurs
            self._analyze_feedback_patterns(feedback_df)

            return feedback_df

        except Exception as e:
            print(f"❌ Erreur extraction feedbacks: {e}")
            return pd.DataFrame()

    def _analyze_feedback_patterns(self, feedback_df):
        """Analyse les patterns dans les feedbacks négatifs"""
        print(f"\n🔍 ANALYSE DES PATTERNS D'ERREURS:")

        # Distribution par prédiction originale
        pred_dist = feedback_df['original_prediction'].value_counts()
        print(f"   Erreurs par prédiction:")
        for pred, count in pred_dist.items():
            print(f"     {pred}: {count}")

        # Distribution par langue
        if 'language' in feedback_df.columns:
            lang_dist = feedback_df['language'].value_counts()
            print(f"   Erreurs par langue:")
            for lang, count in lang_dist.items():
                print(f"     {lang}: {count}")

        # Analyse de la confiance
        if 'confidence' in feedback_df.columns:
            avg_confidence = feedback_df['confidence'].mean()
            print(f"   Confiance moyenne des erreurs: {avg_confidence:.3f}")

            high_conf_errors = len(feedback_df[feedback_df['confidence'] > 0.8])
            if high_conf_errors > 0:
                print(f"   ⚠️ Erreurs haute confiance (>80%): {high_conf_errors} - Problème grave détecté!")

    def load_dataset_sample(self, dataset_path, sample_size=500):
        """
        ÉTAPE 3: Charge un échantillon stratifié du dataset principal
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 3: CHARGEMENT ÉCHANTILLON DATASET PRINCIPAL")
        print("=" * 60)

        try:
            if not Path(dataset_path).exists():
                print(f"⚠️ Dataset principal non trouvé: {dataset_path}")
                return pd.DataFrame()

            df = pd.read_csv(dataset_path)
            print(f"📊 Dataset principal: {len(df)} échantillons")

            # Échantillonnage stratifié amélioré
            if len(df) > sample_size:
                df['stratify_col'] = df['label'].astype(str) + '_' + df['language'].astype(str)

                # Calculer les proportions pour chaque strate
                strata_props = df['stratify_col'].value_counts(normalize=True)

                # Échantillonner proportionnellement
                sample_dfs = []
                for stratum, prop in strata_props.items():
                    stratum_df = df[df['stratify_col'] == stratum]
                    stratum_sample_size = max(1, int(sample_size * prop))

                    if len(stratum_df) > stratum_sample_size:
                        stratum_sample = stratum_df.sample(n=stratum_sample_size, random_state=42)
                    else:
                        stratum_sample = stratum_df

                    sample_dfs.append(stratum_sample)

                sample_df = pd.concat(sample_dfs, ignore_index=True)

                # Si on dépasse encore, échantillonner aléatoirement
                if len(sample_df) > sample_size:
                    sample_df = sample_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

                print(f"📋 Échantillon stratifié: {len(sample_df)} échantillons")
            else:
                sample_df = df.copy()
                print(f"📋 Dataset complet utilisé: {len(sample_df)}")

            # Afficher la distribution finale
            print(f"\n📊 Distribution de l'échantillon:")
            label_dist = sample_df['label'].value_counts()
            for label, count in label_dist.items():
                print(f"   {label}: {count} ({count / len(sample_df) * 100:.1f}%)")

            if 'language' in sample_df.columns:
                lang_dist = sample_df['language'].value_counts()
                print(f"\n🌍 Distribution des langues:")
                for lang, count in lang_dist.items():
                    print(f"   {lang}: {count} ({count / len(sample_df) * 100:.1f}%)")

            return sample_df[['text', 'label', 'language']].copy()

        except Exception as e:
            print(f"❌ Erreur chargement dataset: {e}")
            return pd.DataFrame()

    def combine_datasets(self, feedback_df, sample_df):
        """
        ÉTAPE 4: Combine intelligemment les feedbacks avec l'échantillon du dataset
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 4: COMBINAISON INTELLIGENTE DES DATASETS")
        print("=" * 60)

        if feedback_df.empty and sample_df.empty:
            print("❌ Aucune donnée disponible pour le fine-tuning")
            return pd.DataFrame()

        datasets_to_combine = []

        if not feedback_df.empty:
            feedback_clean = feedback_df[['text', 'label', 'language']].copy()
            feedback_clean['source'] = 'feedback'
            feedback_clean['weight'] = 2.0  # Pondération plus forte pour les feedbacks
            datasets_to_combine.append(feedback_clean)
            print(f"📝 Feedbacks: {len(feedback_clean)} échantillons (poids: 2.0)")

        if not sample_df.empty:
            sample_clean = sample_df[['text', 'label', 'language']].copy()
            sample_clean['source'] = 'dataset'
            sample_clean['weight'] = 1.0
            datasets_to_combine.append(sample_clean)
            print(f"📊 Dataset principal: {len(sample_clean)} échantillons (poids: 1.0)")

        if not datasets_to_combine:
            return pd.DataFrame()

        # Combiner
        combined_df = pd.concat(datasets_to_combine, ignore_index=True)

        # Équilibrer les classes si nécessaire
        combined_df = self._balance_classes(combined_df)

        print(f"🔗 Dataset final: {len(combined_df)} échantillons")

        # Statistiques finales
        print(f"\n📋 Distribution finale:")
        label_dist = combined_df['label'].value_counts()
        for label, count in label_dist.items():
            print(f"   {label}: {count} ({count / len(combined_df) * 100:.1f}%)")

        return combined_df

    def _balance_classes(self, df):
        """Équilibre les classes pour éviter le biais"""
        print(f"\n⚖️ ÉQUILIBRAGE DES CLASSES:")

        original_dist = df['label'].value_counts()
        print(f"   Distribution originale: {dict(original_dist)}")

        # Si le déséquilibre est trop important (ratio > 3:1), sous-échantillonner
        max_count = original_dist.max()
        min_count = original_dist.min()

        if max_count / min_count > 3:
            print(f"   Déséquilibre détecté (ratio {max_count / min_count:.1f}:1)")

            # Limiter la classe majoritaire
            target_size = min(max_count, min_count * 2)  # Ratio max 2:1

            balanced_dfs = []
            for label in df['label'].unique():
                label_df = df[df['label'] == label]
                if len(label_df) > target_size:
                    # Prioriser les feedbacks lors du sous-échantillonnage
                    feedback_samples = label_df[label_df['source'] == 'feedback']
                    dataset_samples = label_df[label_df['source'] == 'dataset']

                    remaining_size = target_size - len(feedback_samples)
                    if remaining_size > 0 and len(dataset_samples) > 0:
                        dataset_subsample = dataset_samples.sample(
                            n=min(remaining_size, len(dataset_samples)),
                            random_state=42
                        )
                        label_balanced = pd.concat([feedback_samples, dataset_subsample])
                    else:
                        label_balanced = feedback_samples[:target_size]
                else:
                    label_balanced = label_df

                balanced_dfs.append(label_balanced)

            df = pd.concat(balanced_dfs, ignore_index=True)

            final_dist = df['label'].value_counts()
            print(f"   Distribution équilibrée: {dict(final_dist)}")
        else:
            print(f"   Distribution acceptable (ratio {max_count / min_count:.1f}:1)")

        return df

    def preprocess_text(self, text, language='en'):
        """Préprocesse le texte avec la même méthode que le modèle original"""
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
        """Extrait les features numériques avec la même méthode que le modèle original"""
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

    def prepare_fine_tuning_data(self, combined_df):
        """
        ÉTAPE 5: Prépare les données pour le fine-tuning avec vérifications de sécurité
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 5: PRÉPARATION SÉCURISÉE DES DONNÉES")
        print("=" * 60)

        if combined_df.empty:
            print("❌ Aucune donnée à préparer")
            return None, None, None, None, None, None

        try:
            # Détecter les paramètres du modèle
            model_input_shape = self.model.inputs[0].shape
            actual_sequence_length = model_input_shape[1]

            # Détecter la taille du vocabulaire
            embedding_layer = None
            for layer in self.model.layers:
                if 'embedding' in layer.name.lower():
                    embedding_layer = layer
                    break

            actual_vocab_size = embedding_layer.input_dim if embedding_layer else 10001
            max_vocab_id = actual_vocab_size - 1

            print(f"🔧 Paramètres détectés:")
            print(f"   Longueur de séquence: {actual_sequence_length}")
            print(f"   Taille du vocabulaire: {actual_vocab_size}")

            # Prétraitement des textes
            print("🔧 Prétraitement des textes...")
            processed_texts = []
            for _, row in combined_df.iterrows():
                processed_text = self.preprocess_text(row['text'], row.get('language', 'en'))
                processed_texts.append(processed_text)

            # Création des séquences avec filtrage sécurisé
            print("📝 Création des séquences sécurisées...")
            sequences = self.tokenizer.texts_to_sequences(processed_texts)

            # Filtrer les tokens invalides
            filtered_sequences = []
            total_tokens_removed = 0

            for sequence in sequences:
                filtered_sequence = [token_id for token_id in sequence if token_id <= max_vocab_id]
                total_tokens_removed += len(sequence) - len(filtered_sequence)
                filtered_sequences.append(filtered_sequence)

            if total_tokens_removed > 0:
                print(f"🔧 Tokens invalides supprimés: {total_tokens_removed}")

            # Padding des séquences
            X_text = pad_sequences(filtered_sequences, maxlen=actual_sequence_length, padding='post', truncating='post')

            # Features numériques
            print("🔢 Extraction des features numériques...")
            X_num = self.extract_numerical_features(combined_df['text'])
            X_num = self.scaler.transform(X_num)

            # Labels
            print("🏷️ Préparation des labels...")
            y = self.label_encoder.transform(combined_df['label'])

            # Division train/validation avec stratification
            print("📋 Division stratifiée train/validation...")

            # Créer les poids d'échantillons si disponibles
            sample_weights = combined_df.get('weight', pd.Series([1.0] * len(combined_df))).values

            X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val, weights_train, weights_val = train_test_split(
                X_text, X_num, y, sample_weights,
                test_size=self.finetune_config['validation_split'],
                random_state=42,
                stratify=y
            )

            # Vérifications de sécurité
            self._perform_safety_checks(X_text_train, X_num_train, y_train, actual_sequence_length, actual_vocab_size)

            print(f"✅ Données préparées avec succès:")
            print(f"   Train: {len(X_text_train)} échantillons")
            print(f"   Validation: {len(X_text_val)} échantillons")
            print(f"   Distribution train: {np.bincount(y_train)}")
            print(f"   Distribution val: {np.bincount(y_val)}")

            return X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val

        except Exception as e:
            print(f"❌ Erreur préparation données: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None, None

    def _perform_safety_checks(self, X_text, X_num, y, expected_seq_length, expected_vocab_size):
        """Effectue des vérifications de sécurité sur les données"""
        print(f"🛡️ VÉRIFICATIONS DE SÉCURITÉ:")

        # Vérifier les dimensions
        assert X_text.shape[
                   1] == expected_seq_length, f"Longueur de séquence incorrecte: {X_text.shape[1]} != {expected_seq_length}"
        assert X_num.shape[1] == 10, f"Nombre de features incorrecte: {X_num.shape[1]} != 10"

        # Vérifier les valeurs
        max_token_id = X_text.max()
        assert max_token_id < expected_vocab_size, f"Token ID trop élevé: {max_token_id} >= {expected_vocab_size}"

        # Vérifier l'équilibre des classes
        class_dist = np.bincount(y)
        class_ratio = class_dist.max() / class_dist.min() if class_dist.min() > 0 else float('inf')
        if class_ratio > 5:
            print(f"   ⚠️ Déséquilibre des classes détecté: {class_ratio:.1f}:1")

        # Vérifier les NaN
        assert not np.isnan(X_text).any(), "NaN détectés dans X_text"
        assert not np.isnan(X_num).any(), "NaN détectés dans X_num"

        print(f"   ✅ Toutes les vérifications passées")

    def create_production_backup(self):
        """Crée une sauvegarde complète du modèle de production"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.data_dir / f"production_backup_{timestamp}"

        print(f"\n💾 CRÉATION SAUVEGARDE DE PRODUCTION")
        print(f"   Destination: {backup_dir}")

        try:
            # Créer le dossier de sauvegarde
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Copier tous les artefacts
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
                    print(f"   ✅ {artifact} sauvegardé")
                else:
                    print(f"   ⚠️ {artifact} non trouvé")

            # Créer un fichier de métadonnées de sauvegarde
            backup_metadata = {
                'backup_timestamp': timestamp,
                'original_model_metadata': self.metadata,
                'backup_reason': 'before_finetuning',
                'restoration_command': f'cp -r {backup_dir}/* {self.model_dir}/'
            }

            with open(backup_dir / 'backup_metadata.json', 'w') as f:
                json.dump(backup_metadata, f, indent=2)

            print(f"✅ Sauvegarde de production créée avec succès")

            return backup_dir

        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return None

    def perform_fine_tuning(self, X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val):
        """
        ÉTAPE 6: Effectue le fine-tuning conservateur du modèle
        """
        print("\n" + "=" * 60)
        print("ÉTAPE 6: FINE-TUNING CONSERVATEUR")
        print("=" * 60)

        try:
            # Créer une copie du modèle pour le fine-tuning
            model_for_finetuning = tf.keras.models.clone_model(self.model)
            model_for_finetuning.set_weights(self.model.get_weights())

            # Configuration conservatrice
            print(f"🎯 Configuration fine-tuning:")
            for key, value in self.finetune_config.items():
                if key != 'evaluation_dataset_size':
                    print(f"   {key}: {value}")

            # Optimiseur avec learning rate très faible
            new_optimizer = Adam(learning_rate=self.finetune_config['learning_rate'])
            model_for_finetuning.compile(
                optimizer=new_optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall')]
            )

            # Callbacks conservateurs
            callbacks = [
                EarlyStopping(
                    patience=self.finetune_config['patience'],
                    restore_best_weights=True,
                    monitor='val_loss',
                    min_delta=0.001  # Seuil minimal d'amélioration
                ),
                ReduceLROnPlateau(
                    factor=0.5,
                    patience=1,  # Réduire rapidement si pas d'amélioration
                    min_lr=1e-9,
                    monitor='val_loss'
                )
            ]

            # Calcul des poids de classe
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            print(f"⚖️ Poids de classe: {class_weight_dict}")

            # Évaluation avant fine-tuning - CORRECTION ICI
            print(f"\n📊 ÉVALUATION AVANT FINE-TUNING:")
            eval_results_before = self.model.evaluate([X_text_val, X_num_val], y_val, verbose=0)

            # Le modèle peut retourner plusieurs métriques, extraire loss et accuracy
            if isinstance(eval_results_before, list):
                val_loss_before = eval_results_before[0]  # Loss toujours en premier
                val_acc_before = eval_results_before[1] if len(eval_results_before) > 1 else 0.0  # Accuracy en second
            else:
                val_loss_before = eval_results_before
                val_acc_before = 0.0

            print(f"   Loss: {val_loss_before:.4f}, Accuracy: {val_acc_before:.4f}")

            # Fine-tuning
            print(f"\n🚀 Début du fine-tuning conservateur...")
            history = model_for_finetuning.fit(
                [X_text_train, X_num_train], y_train,
                validation_data=([X_text_val, X_num_val], y_val),
                batch_size=self.finetune_config['batch_size'],
                epochs=self.finetune_config['epochs'],
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )

            # Évaluation après fine-tuning - CORRECTION ICI AUSSI
            print(f"\n📊 ÉVALUATION APRÈS FINE-TUNING:")
            eval_results_after = model_for_finetuning.evaluate([X_text_val, X_num_val], y_val, verbose=0)

            # Même logique pour les résultats après fine-tuning
            if isinstance(eval_results_after, list):
                val_loss_after = eval_results_after[0]
                val_acc_after = eval_results_after[1] if len(eval_results_after) > 1 else 0.0
            else:
                val_loss_after = eval_results_after
                val_acc_after = 0.0

            print(f"   Loss: {val_loss_after:.4f}, Accuracy: {val_acc_after:.4f}")

            improvement = val_acc_after - val_acc_before
            print(f"   Amélioration: {improvement:+.4f} ({improvement * 100:+.2f}%)")

            if improvement > 0:
                print("✅ Fine-tuning terminé avec amélioration!")
                # Remplacer le modèle principal par la version fine-tunée
                self.finetuned_model = model_for_finetuning
                return history
            else:
                print("⚠️ Fine-tuning n'a pas amélioré le modèle")
                self.finetuned_model = None
                return None

        except Exception as e:
            print(f"❌ Erreur fine-tuning: {e}")
            import traceback
            traceback.print_exc()
            self.finetuned_model = None
            return None

    def evaluate_model_comprehensive(self, model, X_text, X_num, y, model_name="Model"):
        """
        Évalue un modèle de manière complète avec toutes les métriques importantes
        """
        print(f"\n📊 ÉVALUATION COMPLÈTE - {model_name}")
        print("=" * 50)

        try:
            # Prédictions
            y_pred_proba = model.predict([X_text, X_num], verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Métriques principales
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted'),
                'auc': roc_auc_score(y, y_pred_proba),
                'total_samples': len(y)
            }

            # Métriques par classe
            precision_per_class = precision_score(y, y_pred, average=None)
            recall_per_class = recall_score(y, y_pred, average=None)
            f1_per_class = f1_score(y, y_pred, average=None)

            # Matrice de confusion
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # Métriques détaillées
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

            metrics.update({
                'specificity': specificity,
                'sensitivity': sensitivity,
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'f1_per_class': f1_per_class.tolist()
            })

            # Affichage des résultats
            print(f"📈 Métriques principales:")
            print(f"   Accuracy:    {metrics['accuracy']:.4f}")
            print(f"   Precision:   {metrics['precision']:.4f}")
            print(f"   Recall:      {metrics['recall']:.4f}")
            print(f"   F1-Score:    {metrics['f1']:.4f}")
            print(f"   AUC:         {metrics['auc']:.4f}")
            print(f"   Specificity: {metrics['specificity']:.4f}")
            print(f"   Sensitivity: {metrics['sensitivity']:.4f}")

            print(f"\n🔍 Matrice de confusion:")
            class_names = self.label_encoder.classes_
            print(f"   Vrais {class_names[0]}: {tn:4d} | Faux {class_names[1]}: {fp:4d}")
            print(f"   Faux {class_names[0]}: {fn:4d} | Vrais {class_names[1]}: {tp:4d}")

            return metrics

        except Exception as e:
            print(f"❌ Erreur évaluation {model_name}: {e}")
            return {}

    def load_full_dataset_for_evaluation(self, dataset_path):
        """
        Charge le dataset complet pour l'évaluation finale
        """
        print(f"\n📂 CHARGEMENT DATASET COMPLET POUR ÉVALUATION")
        print("=" * 50)

        try:
            if not Path(dataset_path).exists():
                print(f"❌ Dataset non trouvé: {dataset_path}")
                return None, None, None

            # Charger tout le dataset
            df = pd.read_csv(dataset_path)
            print(f"📊 Dataset complet: {len(df)} échantillons")

            # Limitation pour éviter les problèmes de mémoire
            max_eval_size = self.finetune_config.get('evaluation_dataset_size', 5000)
            if max_eval_size and len(df) > max_eval_size:
                print(f"🔄 Limitation à {max_eval_size} échantillons pour l'évaluation")
                df = df.sample(n=max_eval_size, random_state=42).reset_index(drop=True)

            # Préparation des données
            processed_texts = []
            for _, row in df.iterrows():
                processed_text = self.preprocess_text(row['text'], row.get('language', 'en'))
                processed_texts.append(processed_text)

            # Séquences
            sequences = self.tokenizer.texts_to_sequences(processed_texts)

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

            # Filtrer les séquences
            filtered_sequences = []
            for sequence in sequences:
                filtered_sequence = [token_id for token_id in sequence if token_id <= max_vocab_id]
                filtered_sequences.append(filtered_sequence)

            X_text = pad_sequences(filtered_sequences, maxlen=actual_sequence_length, padding='post', truncating='post')

            # Features numériques
            X_num = self.extract_numerical_features(df['text'])
            X_num = self.scaler.transform(X_num)

            # Labels
            y = self.label_encoder.transform(df['label'])

            print(f"✅ Dataset d'évaluation préparé: {X_text.shape[0]} échantillons")

            return X_text, X_num, y

        except Exception as e:
            print(f"❌ Erreur chargement dataset d'évaluation: {e}")
            return None, None, None

    def compare_models_and_decide(self, dataset_path):
        """
        Compare le modèle original avec le modèle fine-tuné et décide du déploiement
        """
        print("\n" + "🏆" * 60)
        print("ÉTAPE 7: COMPARAISON MODÈLE ORIGINAL VS FINE-TUNÉ")
        print("🏆" * 60)

        if not hasattr(self, 'finetuned_model') or self.finetuned_model is None:
            print("❌ Aucun modèle fine-tuné disponible pour la comparaison")
            return False, None

        # Charger le dataset d'évaluation
        X_text_eval, X_num_eval, y_eval = self.load_full_dataset_for_evaluation(dataset_path)

        if X_text_eval is None:
            print("❌ Impossible de charger le dataset d'évaluation")
            return False, None

        # Évaluer le modèle original
        print(f"\n1️⃣ ÉVALUATION DU MODÈLE ORIGINAL")
        original_metrics = self.evaluate_model_comprehensive(
            self.model, X_text_eval, X_num_eval, y_eval, "MODÈLE ORIGINAL"
        )

        # Évaluer le modèle fine-tuné
        print(f"\n2️⃣ ÉVALUATION DU MODÈLE FINE-TUNÉ")
        finetuned_metrics = self.evaluate_model_comprehensive(
            self.finetuned_model, X_text_eval, X_num_eval, y_eval, "MODÈLE FINE-TUNÉ"
        )

        # Comparaison détaillée
        comparison_result = self._detailed_comparison(original_metrics, finetuned_metrics)

        return comparison_result['should_deploy'], comparison_result

    def _detailed_comparison(self, original_metrics, finetuned_metrics):
        """
        Effectue une comparaison détaillée entre les deux modèles
        """
        print(f"\n3️⃣ COMPARAISON DÉTAILLÉE ET DÉCISION")
        print("=" * 50)

        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity', 'sensitivity']
        improvements = {}
        significant_improvements = []
        degradations = []

        print(f"{'Métrique':<12} {'Original':<10} {'Fine-tuné':<10} {'Diff':<10} {'%':<8} {'Statut'}")
        print("-" * 70)

        for metric in metrics_to_compare:
            if metric in original_metrics and metric in finetuned_metrics:
                original_val = original_metrics[metric]
                finetuned_val = finetuned_metrics[metric]
                improvement = finetuned_val - original_val
                improvement_pct = (improvement / original_val) * 100 if original_val > 0 else 0

                improvements[metric] = improvement

                # Déterminer le statut
                if improvement > self.deployment_criteria['min_improvement']:
                    status = "🟢 MIEUX"
                    if metric in self.deployment_criteria['primary_metrics']:
                        significant_improvements.append(metric)
                elif improvement < -self.deployment_criteria['allow_slight_degradation']:
                    status = "🔴 PIRE"
                    degradations.append(metric)
                else:
                    status = "🟡 ÉGAL"

                print(f"{metric.capitalize():<12} {original_val:<10.4f} {finetuned_val:<10.4f} "
                      f"{improvement:<+10.4f} {improvement_pct:<+8.2f}% {status}")

        # Analyse spécifique pour la détection de spam/phishing
        self._analyze_classification_performance(original_metrics, finetuned_metrics)

        # Décision de déploiement
        decision_result = self._make_deployment_decision(
            improvements, significant_improvements, degradations
        )

        return decision_result

    def _analyze_classification_performance(self, original_metrics, finetuned_metrics):
        """
        Analyse spécifique pour la performance de classification spam/phishing
        """
        print(f"\n🔍 ANALYSE SPÉCIALISÉE DÉTECTION SPAM/PHISHING:")

        # Analyser les faux positifs/négatifs
        print(f"   Modèle Original:")
        print(f"     Faux Positifs: {original_metrics.get('false_positives', 0)} (légitime classé spam)")
        print(f"     Faux Négatifs: {original_metrics.get('false_negatives', 0)} (spam non détecté)")

        print(f"   Modèle Fine-tuné:")
        print(f"     Faux Positifs: {finetuned_metrics.get('false_positives', 0)} (légitime classé spam)")
        print(f"     Faux Négatifs: {finetuned_metrics.get('false_negatives', 0)} (spam non détecté)")

        # Calcul de l'impact
        fp_change = finetuned_metrics.get('false_positives', 0) - original_metrics.get('false_positives', 0)
        fn_change = finetuned_metrics.get('false_negatives', 0) - original_metrics.get('false_negatives', 0)

        print(f"   Impact:")
        if fp_change < 0:
            print(f"     ✅ Réduction des faux positifs: {-fp_change}")
        elif fp_change > 0:
            print(f"     ⚠️ Augmentation des faux positifs: {fp_change}")

        if fn_change < 0:
            print(f"     ✅ Réduction des faux négatifs: {-fn_change}")
        elif fn_change > 0:
            print(f"     ⚠️ Augmentation des faux négatifs: {fn_change}")

        # Score de risque
        risk_score = (fp_change * 0.5) + (fn_change * 1.0)  # Les FN sont plus graves
        if risk_score > 5:
            print(f"     🚨 RISQUE ÉLEVÉ: Score de risque = {risk_score}")
        elif risk_score > 0:
            print(f"     ⚠️ Risque modéré: Score de risque = {risk_score}")
        else:
            print(f"     ✅ Risque acceptable: Score de risque = {risk_score}")

    def _make_deployment_decision(self, improvements, significant_improvements, degradations):
        """
        Prend la décision finale de déploiement basée sur les critères stricts
        """
        print(f"\n4️⃣ DÉCISION DE DÉPLOIEMENT")
        print("=" * 30)

        should_deploy = False
        reason = ""
        confidence = 0.0

        # Critère 1: Amélioration sur les métriques principales
        primary_improved = all(
            metric in significant_improvements
            for metric in self.deployment_criteria['primary_metrics']
        )

        # Critère 2: Pas de dégradation significative sur les métriques critiques
        critical_degradation = any(
            metric in degradations
            for metric in ['accuracy', 'f1', 'auc']
        )

        # Critère 3: Amélioration globale
        avg_improvement = np.mean([
            improvements.get(metric, 0)
            for metric in self.deployment_criteria['primary_metrics']
        ])

        print(f"📋 Analyse des critères:")
        print(f"   Métriques principales améliorées: {'✅' if primary_improved else '❌'}")
        print(f"   Absence de dégradation critique: {'✅' if not critical_degradation else '❌'}")
        print(f"   Amélioration moyenne: {avg_improvement * 100:+.2f}%")

        # Logique de décision
        if primary_improved and not critical_degradation and avg_improvement > self.deployment_criteria[
            'min_improvement']:
            should_deploy = True
            confidence = min(avg_improvement * 10, 1.0)
            reason = f"Amélioration significative détectée ({avg_improvement * 100:.2f}%)"

        elif not primary_improved:
            reason = "Pas d'amélioration sur les métriques principales"

        elif critical_degradation:
            reason = "Dégradation critique détectée"

        else:
            reason = f"Amélioration insuffisante ({avg_improvement * 100:.2f}% < {self.deployment_criteria['min_improvement'] * 100:.2f}%)"

        # Affichage de la décision
        if should_deploy:
            print(f"\n🎉 DÉCISION: DÉPLOYER LE MODÈLE FINE-TUNÉ")
            print(f"   Raison: {reason}")
            print(f"   Confiance: {confidence * 100:.1f}%")
        else:
            print(f"\n🛑 DÉCISION: CONSERVER LE MODÈLE ORIGINAL")
            print(f"   Raison: {reason}")

        return {
            'should_deploy': should_deploy,
            'reason': reason,
            'confidence': confidence,
            'improvements': improvements,
            'significant_improvements': significant_improvements,
            'degradations': degradations,
            'avg_improvement': avg_improvement
        }

    def deploy_finetuned_model(self, backup_dir):
        """
        Déploie le modèle fine-tuné en remplaçant le modèle de production
        """
        print(f"\n🚀 DÉPLOIEMENT DU MODÈLE FINE-TUNÉ")
        print("=" * 40)

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Sauvegarder le modèle fine-tuné temporairement
            temp_model_path = self.data_dir / f"finetuned_model_{timestamp}.keras"
            self.finetuned_model.save(str(temp_model_path))
            print(f"💾 Modèle fine-tuné sauvegardé temporairement: {temp_model_path}")

            # Remplacer le modèle de production
            production_model_path = self.model_dir / "best_lstm_model.keras"
            shutil.copy2(temp_model_path, production_model_path)
            print(f"✅ Modèle de production remplacé dans {production_model_path}")

            # Mettre à jour les métadonnées
            updated_metadata = self.metadata.copy()
            updated_metadata.update({
                'finetuned_at': timestamp,
                'finetuning_config': self.finetune_config,
                'deployment_criteria': self.deployment_criteria,
                'backup_location': str(backup_dir),
                'model_version': updated_metadata.get('model_version', 1) + 1,
                'deployment_method': 'automatic_finetuning',
                'previous_model_backup': str(backup_dir)
            })

            metadata_path = self.model_dir / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(updated_metadata, f, indent=2)
            print(f"✅ Métadonnées mises à jour")

            # Nettoyer le fichier temporaire
            temp_model_path.unlink()

            print(f"\n🎉 DÉPLOIEMENT RÉUSSI!")
            print(f"   Version du modèle: {updated_metadata['model_version']}")
            print(f"   Sauvegarde disponible: {backup_dir}")
            print(f"   Redémarrez l'API pour utiliser le nouveau modèle:")
            print(f"   docker-compose restart fastapi")

            return True

        except Exception as e:
            print(f"❌ Erreur lors du déploiement: {e}")
            return False

    def mark_feedbacks_as_processed(self, feedback_df):
        """
        Marque les feedbacks comme traités dans le fichier CSV
        """
        print(f"\n📝 MARQUAGE DES FEEDBACKS COMME TRAITÉS")
        print("=" * 40)

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
                df.loc[feedback_ids, 'processed_at'] = datetime.now().isoformat()
                processed_count = len(feedback_ids)
            else:
                # Fallback: marquer tous les feedbacks négatifs non traités
                mask = (df['user_satisfaction'] == 'no') & (df['processed'] == False)
                df.loc[mask, 'processed'] = True
                df.loc[mask, 'processed_at'] = datetime.now().isoformat()
                processed_count = mask.sum()

            # Sauvegarder
            df.to_csv(self.feedback_csv_path, index=False)
            print(f"✅ {processed_count} feedbacks marqués comme traités")

            return True

        except Exception as e:
            print(f"❌ Erreur marquage feedbacks: {e}")
            return False

    def run_complete_advanced_finetuning(self, dataset_path="./data/test_dataset.csv"):
        """
        FONCTION PRINCIPALE: Processus complet de fine-tuning avancé avec comparaison et déploiement automatique
        """
        print("\n" + "🎯" * 30)
        print("PROCESSUS COMPLET DE FINE-TUNING AVANCÉ")
        print("AVEC COMPARAISON ET DÉPLOIEMENT AUTOMATIQUE")
        print("🎯" * 30)

        start_time = datetime.now()

        # Étape 1: Charger les artefacts
        if not self.load_model_artifacts():
            print("❌ Échec du chargement des artefacts")
            return False

        # Étape 2: Extraire les feedbacks négatifs
        feedback_df = self.extract_negative_feedbacks()
        if feedback_df.empty:
            print("ℹ️ Aucun feedback négatif à traiter - Fine-tuning non nécessaire")
            return False

        print(f"📝 {len(feedback_df)} feedbacks négatifs trouvés pour le fine-tuning")

        # Étape 3: Créer une sauvegarde de sécurité
        backup_dir = self.create_production_backup()
        if backup_dir is None:
            print("❌ Impossible de créer la sauvegarde - Arrêt par sécurité")
            return False

        # Étape 4: Charger échantillon du dataset principal
        sample_df = self.load_dataset_sample(dataset_path, self.finetune_config['sample_size'])

        # Étape 5: Combiner les datasets intelligemment
        combined_df = self.combine_datasets(feedback_df, sample_df)
        if combined_df.empty:
            print("❌ Aucune donnée disponible pour le fine-tuning")
            return False

        # Étape 6: Préparer les données
        data_results = self.prepare_fine_tuning_data(combined_df)
        if data_results[0] is None:
            print("❌ Échec de la préparation des données")
            return False

        X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val = data_results

        # Étape 7: Fine-tuning conservateur
        history = self.perform_fine_tuning(X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val)
        if history is None:
            print("❌ Fine-tuning n'a pas amélioré le modèle - Conservation du modèle original")
            return False

        # Étape 8: Comparaison complète et décision de déploiement
        should_deploy, comparison_result = self.compare_models_and_decide(dataset_path)

        # Étape 9: Déploiement conditionnel
        deployment_success = False
        if should_deploy:
            deployment_success = self.deploy_finetuned_model(backup_dir)

            if deployment_success:
                # Marquer les feedbacks comme traités seulement en cas de succès
                self.mark_feedbacks_as_processed(feedback_df)
            else:
                print("❌ Échec du déploiement - Conservation du modèle original")

        # Résumé final
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "🎉" * 50)
        print("RÉSUMÉ DU PROCESSUS DE FINE-TUNING AVANCÉ")
        print("🎉" * 50)

        print(f"⏱️ Durée totale: {duration}")
        print(f"📊 Feedbacks traités: {len(feedback_df)}")
        print(f"📈 Données d'entraînement: {len(combined_df)} échantillons")

        if comparison_result:
            print(f"📋 Amélioration moyenne: {comparison_result['avg_improvement'] * 100:+.2f}%")
            print(f"🎯 Métriques améliorées: {comparison_result['significant_improvements']}")

        if should_deploy and deployment_success:
            print(f"🚀 RÉSULTAT: NOUVEAU MODÈLE DÉPLOYÉ AVEC SUCCÈS!")
            print(f"💾 Sauvegarde disponible: {backup_dir}")
            print(f"🔄 Redémarrez l'API pour utiliser le nouveau modèle:")
            print(f"   docker-compose restart fastapi")
        elif should_deploy and not deployment_success:
            print(f"⚠️ RÉSULTAT: Déploiement échoué - Modèle original conservé")
            print(f"🔄 Restauration possible depuis: {backup_dir}")
        else:
            print(f"🛑 RÉSULTAT: Modèle original conservé (performance insuffisante)")
            print(f"💡 Raison: {comparison_result.get('reason', 'Critères non satisfaits')}")

        return deployment_success


def main():
    """
    Fonction principale pour exécuter le fine-tuning avancé
    """
    print("🚀 DÉMARRAGE DU FINE-TUNING MANAGER AVANCÉ")
    print("=" * 60)

    # Initialiser le gestionnaire avancé
    manager = AdvancedFineTuningManager(
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
        print("\n💡 Assurez-vous d'avoir entraîné le modèle principal d'abord")
        return False

    if not manager.feedback_csv_path.exists():
        print(f"❌ Fichier de feedbacks manquant: {manager.feedback_csv_path}")
        print("💡 Aucun feedback à traiter pour le moment")
        return False

    print("✅ Tous les prérequis sont satisfaits")

    # Exécuter le fine-tuning avancé complet
    success = manager.run_complete_advanced_finetuning()

    if success:
        print("\n🎉 Fine-tuning avancé réussi avec déploiement automatique!")
        print("💡 Le nouveau modèle est maintenant en production")
        print("🔄 Redémarrez l'API Docker pour utiliser le nouveau modèle")
    else:
        print("\n🛑 Fine-tuning terminé sans déploiement")
        print("💡 Le modèle original reste en production")
        print("📊 Consultez les logs pour plus de détails")

    return success


class FeedbackAnalyzer:
    """
    Analyseur avancé des patterns dans les feedbacks pour le monitoring
    """

    def __init__(self, feedback_csv_path):
        self.feedback_csv_path = Path(feedback_csv_path)

    def analyze_comprehensive_patterns(self):
        """
        Analyse complète des patterns dans les feedbacks
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
            'patterns': {},
            'recommendations': []
        }

        # Analyse temporelle
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date

            daily_errors = df[df['user_satisfaction'] == 'no'].groupby('date').size()
            if len(daily_errors) > 1:
                recent_trend = daily_errors.tail(7).mean() - daily_errors.head(-7).mean()
                analysis['patterns']['error_trend'] = {
                    'recent_avg': float(daily_errors.tail(7).mean()),
                    'historical_avg': float(daily_errors.head(-7).mean()),
                    'trend': 'increasing' if recent_trend > 0 else 'decreasing'
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
                analysis['patterns']['confidence_analysis'] = {
                    'avg_confidence_errors': float(neg_probs.mean()),
                    'high_confidence_errors': len(neg_probs[neg_probs > 0.8]),
                    'medium_confidence_errors': len(neg_probs[(neg_probs >= 0.6) & (neg_probs < 0.8)]),
                    'low_confidence_errors': len(neg_probs[neg_probs < 0.6])
                }

            # Générer des recommandations
            self._generate_recommendations(analysis, negative_feedbacks)

        return analysis

    def _generate_recommendations(self, analysis, negative_feedbacks):
        """
        Génère des recommandations basées sur l'analyse des feedbacks
        """
        recommendations = []

        # Recommandations basées sur le taux d'erreur
        error_rate = analysis['negative_ratio']
        if error_rate > 0.15:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'ERROR_RATE',
                'message': f"Taux d'erreur élevé ({error_rate:.1%}). Fine-tuning urgent recommandé."
            })
        elif error_rate > 0.10:
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'ERROR_RATE',
                'message': f"Taux d'erreur modéré ({error_rate:.1%}). Surveillance accrue recommandée."
            })

        # Recommandations basées sur les erreurs haute confiance
        if 'confidence_analysis' in analysis['patterns']:
            high_conf_errors = analysis['patterns']['confidence_analysis']['high_confidence_errors']
            if high_conf_errors > 5:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'type': 'HIGH_CONFIDENCE_ERRORS',
                    'message': f"{high_conf_errors} erreurs haute confiance détectées. Problème de modèle possible."
                })

        # Recommandations basées sur les patterns de langue
        if 'language_errors' in analysis['patterns']:
            lang_errors = analysis['patterns']['language_errors']
            total_errors = sum(lang_errors.values())
            for lang, count in lang_errors.items():
                if count / total_errors > 0.7:
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'type': 'LANGUAGE_BIAS',
                        'message': f"Biais détecté pour la langue {lang} ({count}/{total_errors} erreurs)"
                    })

        # Recommandation de fine-tuning
        if len(negative_feedbacks) >= 5:
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'FINETUNING_READY',
                'message': f"Seuil de fine-tuning atteint ({len(negative_feedbacks)} feedbacks négatifs)"
            })

        analysis['recommendations'] = recommendations

    def print_comprehensive_analysis(self):
        """
        Affiche l'analyse complète des feedbacks
        """
        analysis = self.analyze_comprehensive_patterns()

        print("\n📊 ANALYSE COMPLÈTE DES FEEDBACKS")
        print("=" * 50)
        print(f"📈 Total des feedbacks: {analysis['total_feedbacks']}")
        print(f"✅ Feedbacks positifs: {analysis['positive_count']}")
        print(f"❌ Feedbacks négatifs: {analysis['negative_count']}")
        print(f"📊 Taux d'erreur: {analysis['negative_ratio']:.2%}")

        if analysis['patterns']:
            print(f"\n🔍 PATTERNS DÉTECTÉS:")

            if 'error_trend' in analysis['patterns']:
                trend_data = analysis['patterns']['error_trend']
                trend_icon = "📈" if trend_data['trend'] == 'increasing' else "📉"
                print(f"   {trend_icon} Tendance des erreurs: {trend_data['trend']}")
                print(f"      Moyenne récente: {trend_data['recent_avg']:.1f}/jour")
                print(f"      Moyenne historique: {trend_data['historical_avg']:.1f}/jour")

            if 'prediction_errors' in analysis['patterns']:
                print(f"   🎯 Erreurs par prédiction:")
                for pred, count in analysis['patterns']['prediction_errors'].items():
                    print(f"      {pred}: {count}")

            if 'language_errors' in analysis['patterns']:
                print(f"   🌍 Erreurs par langue:")
                for lang, count in analysis['patterns']['language_errors'].items():
                    print(f"      {lang}: {count}")

            if 'confidence_analysis' in analysis['patterns']:
                conf_data = analysis['patterns']['confidence_analysis']
                print(f"   🎯 Analyse de confiance:")
                print(f"      Confiance moyenne des erreurs: {conf_data['avg_confidence_errors']:.3f}")
                print(f"      Erreurs haute confiance (>80%): {conf_data['high_confidence_errors']}")
                print(f"      Erreurs confiance moyenne (60-80%): {conf_data['medium_confidence_errors']}")
                print(f"      Erreurs faible confiance (<60%): {conf_data['low_confidence_errors']}")

        if analysis['recommendations']:
            print(f"\n💡 RECOMMANDATIONS:")
            for rec in analysis['recommendations']:
                priority_icon = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "💡"}.get(rec['priority'], "ℹ️")
                print(f"   {priority_icon} [{rec['priority']}] {rec['message']}")

        return analysis


class ModelPerformanceMonitor:
    """
    Moniteur de performance pour suivre l'évolution du modèle
    """

    def __init__(self, model_dir="./model/model_prod", data_dir="./data"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.performance_log_path = self.data_dir / "model_performance_log.json"

    def log_performance(self, metrics, model_version, deployment_info=None):
        """
        Enregistre les performances du modèle
        """
        timestamp = datetime.now().isoformat()

        performance_entry = {
            'timestamp': timestamp,
            'model_version': model_version,
            'metrics': metrics,
            'deployment_info': deployment_info or {}
        }

        # Charger l'historique existant
        if self.performance_log_path.exists():
            with open(self.performance_log_path, 'r') as f:
                performance_log = json.load(f)
        else:
            performance_log = []

        # Ajouter la nouvelle entrée
        performance_log.append(performance_entry)

        # Limiter à 100 entrées max
        if len(performance_log) > 100:
            performance_log = performance_log[-100:]

        # Sauvegarder
        self.data_dir.mkdir(exist_ok=True)
        with open(self.performance_log_path, 'w') as f:
            json.dump(performance_log, f, indent=2)

        print(f"📊 Performance enregistrée pour le modèle v{model_version}")

    def analyze_performance_trend(self):
        """
        Analyse les tendances de performance
        """
        if not self.performance_log_path.exists():
            print("📊 Aucun historique de performance disponible")
            return None

        with open(self.performance_log_path, 'r') as f:
            performance_log = json.load(f)

        if len(performance_log) < 2:
            print("📊 Historique insuffisant pour l'analyse des tendances")
            return None

        print("\n📈 ANALYSE DES TENDANCES DE PERFORMANCE")
        print("=" * 45)

        # Analyser les dernières performances
        recent_entries = performance_log[-5:]  # 5 dernières entrées

        metrics_to_track = ['accuracy', 'f1', 'precision', 'recall']

        for metric in metrics_to_track:
            values = [entry['metrics'].get(metric, 0) for entry in recent_entries if metric in entry['metrics']]
            if len(values) >= 2:
                trend = "↗️" if values[-1] > values[0] else "↘️" if values[-1] < values[0] else "→"
                change = values[-1] - values[0]
                print(f"   {metric.capitalize()}: {values[-1]:.4f} {trend} ({change:+.4f})")

        return performance_log


def restore_from_backup(backup_dir, model_dir="./model/model_prod"):
    """
    Fonction utilitaire pour restaurer un modèle depuis une sauvegarde
    """
    backup_path = Path(backup_dir)
    model_path = Path(model_dir)

    if not backup_path.exists():
        print(f"❌ Sauvegarde non trouvée: {backup_path}")
        return False

    try:
        print(f"🔄 Restauration depuis: {backup_path}")

        # Sauvegarder l'état actuel avant restauration
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
        print(f"🔄 Redémarrez l'API pour utiliser le modèle restauré")
        return True

    except Exception as e:
        print(f"❌ Erreur lors de la restauration: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "analyze":
            # Mode analyse des feedbacks
            print("🔍 ANALYSE DES FEEDBACKS")
            analyzer = FeedbackAnalyzer("./data/user_feedbacks.csv")
            analyzer.print_comprehensive_analysis()

        elif command == "monitor":
            # Mode monitoring des performances
            print("📊 MONITORING DES PERFORMANCES")
            monitor = ModelPerformanceMonitor()
            monitor.analyze_performance_trend()

        elif command == "restore":
            # Mode restauration
            if len(sys.argv) > 2:
                backup_dir = sys.argv[2]
                restore_from_backup(backup_dir)
            else:
                print("❌ Usage: python traitement.py restore <backup_directory>")

        elif command == "help":
            print("📋 UTILISATION DU FINE-TUNING MANAGER AVANCÉ")
            print("=" * 50)
            print("python traitement.py                    # Exécuter le fine-tuning avancé")
            print("python traitement.py analyze            # Analyser les feedbacks")
            print("python traitement.py monitor            # Monitoring des performances")
            print("python traitement.py restore <backup>   # Restaurer depuis une sauvegarde")
            print("python traitement.py help               # Afficher cette aide")
            print("\n🎯 Le mode par défaut effectue:")
            print("   • Analyse des feedbacks négatifs")
            print("   • Fine-tuning conservateur")
            print("   • Comparaison avec le modèle de production (./model/model_prod)")
            print("   • Déploiement automatique si amélioration significative")
            print("   • Sauvegarde de sécurité automatique")

        else:
            print("❌ Commande non reconnue. Utilisez 'help' pour voir les options")
    else:
        # Mode fine-tuning principal
        main()