# --- TRAITEMENT.PY - VERSION FINALE "AUTORITAIRE" ---

import pandas as pd
import numpy as np
import json
import pickle
import re
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import shutil
import sys

# --- Configuration Globale ---
tf.random.set_seed(42)
np.random.seed(42)


def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    return obj


class AuthoritativeFinetuner:
    """
    Gestionnaire de finetuning avec une approche "autoritaire" :
    1. Gèle le corps du modèle (extraction de features).
    2. Remplace et ré-entraîne complètement la tête de classification.
    3. Utilise un dataset de finetuning très ciblé pour forcer la correction.
    """

    def __init__(self, model_dir="./model/model_prod", data_dir="./data"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.feedback_csv_path = self.data_dir / "user_feedbacks.csv"

        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.label_encoder = None
        self.metadata = {}
        self.MAX_SEQUENCE_LENGTH = 200

        self.config = {
            'epochs': 12,
            'batch_size': 8,
            'learning_rate': 0.001,  # LR plus élevé pour la nouvelle tête
            'patience': 3,
            'replay_samples': 50,  # Moins de mémoire, plus de focus
            'min_confidence_threshold': 0.70,  # Seuil de confiance plus strict
        }

        print("✨ AuthoritativeFinetuner initialisé.")

    def load_artifacts(self):
        """Charge tous les artefacts nécessaires."""
        print("\n" + "=" * 50 + "\nÉTAPE 1: CHARGEMENT DES ARTEFACTS\n" + "=" * 50)
        try:
            self.model = load_model(self.model_dir / "best_lstm_model.keras")
            self.MAX_SEQUENCE_LENGTH = self.model.input_shape[0][1]
            with open(self.model_dir / "tokenizer.pkl", 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open(self.model_dir / "scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            with open(self.model_dir / "label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            if (self.model_dir / "model_metadata.json").exists():
                with open(self.model_dir / "model_metadata.json", 'r') as f: self.metadata = json.load(f)
            print(f"✅ Artefacts chargés. Longueur de séquence: {self.MAX_SEQUENCE_LENGTH}")
            return True
        except Exception as e:
            print(f"❌ Erreur critique au chargement: {e}")
            return False

    def get_next_feedback(self):
        """Récupère le feedback à traiter."""
        if not self.feedback_csv_path.exists(): return None
        df = pd.read_csv(self.feedback_csv_path)
        unprocessed = df[(df['user_satisfaction'] == 'no') & (df['processed'] == False)]
        if unprocessed.empty: return None
        feedback_row = unprocessed.iloc[0]
        true_label = 'benign' if feedback_row['predicted_class'] in ['phishing', 'spam'] else 'phishing'
        return {'id': feedback_row.name, 'text': feedback_row['email_text'], 'label': true_label,
                'original_prediction': feedback_row['predicted_class']}

    # --- Fonctions de prétraitement (identiques à l'entraînement) ---
    def preprocess_text(self, text):
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r'http[s]?://\S+', ' URL_TOKEN ', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL_TOKEN ', text)
        text = re.sub(r'\b\d+\b', ' NUM_TOKEN ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but'}
        tokens = [token for token in text.split() if len(token) > 1 and token not in stop_words]
        return ' '.join(tokens)

    def prepare_data_for_model(self, texts):
        processed_texts = [self.preprocess_text(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        padded = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

        # Simulation de l'extraction de features numériques (on ne les utilise pas pour l'entraînement de la tête)
        num_features_dim = self.model.input_shape[1][1]
        numerical = np.zeros((len(texts), num_features_dim))

        return [padded, numerical]

    def create_targeted_dataset(self, feedback_data):
        """Crée un dataset très ciblé pour forcer la correction."""
        print(f"🧠 Création du dataset CIBLÉ...")

        # 1. Beaucoup de copies du feedback
        num_feedback_copies = 50
        feedback_list = [{'text': feedback_data['text'], 'label': feedback_data['label']}] * num_feedback_copies

        training_df_list = [pd.DataFrame(feedback_list)]

        # 2. Quelques exemples "difficiles" de la classe opposée pour la robustesse
        dataset_path = self.data_dir / "full_merged_dataset_fr_en_spam.csv"
        if dataset_path.exists():
            full_df = pd.read_csv(dataset_path).dropna(subset=['text', 'label'])

            # Ajouter 10 exemples de la classe opposée
            other_class_label = 'phishing' if feedback_data['label'] == 'benign' else 'benign'
            other_df = full_df[full_df['label'] == other_class_label]
            if len(other_df) >= 10:
                training_df_list.append(other_df.sample(n=10, random_state=42))

        # 3. Combiner et préparer
        training_df = pd.concat(training_df_list, ignore_index=True).sample(frac=1).reset_index(drop=True)
        print(
            f"   -> Dataset Final: {len(training_df)} exemples | Distribution: {training_df['label'].value_counts().to_dict()}")

        X_text_padded, _ = self.prepare_data_for_model(training_df['text'].tolist())
        y = self.label_encoder.transform(training_df['label'])

        return X_text_padded, y

    def build_and_train_new_head(self, X_train_text, y_train):
        """Construit, attache et entraîne une nouvelle tête de classification."""
        # 1. Geler le corps du modèle original
        for layer in self.model.layers:
            if 'output' not in layer.name and 'dense' not in layer.name:
                layer.trainable = False

        # 2. Identifier la couche de sortie du "corps" du modèle (avant la tête)
        # On remonte jusqu'à la couche de Concaténation
        body_output_layer = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Concatenate):
                body_output_layer = layer
                break

        if body_output_layer is None:
            raise ValueError("Impossible de trouver la couche de 'Concatenate' pour brancher la nouvelle tête.")

        print(f"✅ Corps du modèle gelé. Point de branchement: '{body_output_layer.name}'")

        # 3. Créer la nouvelle tête de classification
        # On utilise une architecture simple mais efficace
        head_input = body_output_layer.output
        new_head = Dense(32, activation='relu', name='new_head_dense_1')(head_input)
        new_head = BatchNormalization(name='new_head_bn_1')(new_head)
        new_head = Dropout(0.5, name='new_head_dropout_1')(new_head)
        new_head = Dense(16, activation='relu', name='new_head_dense_2')(new_head)
        output = Dense(1, activation='sigmoid', name='new_output')(new_head)

        # 4. Créer le nouveau modèle avec le corps gelé et la nouvelle tête
        new_model = Model(inputs=self.model.inputs, outputs=output)

        print("\nArchitecture de la nouvelle tête:")
        # Pour afficher seulement la nouvelle partie, on peut créer un mini-modèle
        temp_head_model = Model(inputs=head_input, outputs=output)
        temp_head_model.summary()

        # 5. Compiler et entraîner SEULEMENT la nouvelle tête
        new_model.compile(optimizer=Adam(learning_rate=self.config['learning_rate']),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=self.config['patience'], restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
        ]

        # On pré-calcule les features du corps pour accélérer l'entraînement de la tête
        body_model = Model(inputs=self.model.inputs, outputs=body_output_layer.output)

        # Créer les données d'entraînement (textes et features numériques vides)
        num_features_dim = self.model.input_shape[1][1]
        X_train_num_dummy = np.zeros((len(X_train_text), num_features_dim))

        print("\n⚡ Pré-calcul des features du corps du modèle...")
        features_from_body = body_model.predict([X_train_text, X_train_num_dummy], batch_size=self.config['batch_size'])
        print("   -> Features calculées.")

        print(f"\n🚀 Entraînement de la nouvelle tête sur {len(features_from_body)} exemples...")
        history = temp_head_model.fit(features_from_body, y_train,
                                      epochs=self.config['epochs'],
                                      batch_size=self.config['batch_size'],
                                      validation_split=0.2,
                                      callbacks=callbacks,
                                      verbose=1)

        return new_model, history

    def run(self):
        """Orchestre le processus de finetuning autoritaire."""
        if not self.load_artifacts(): return

        feedback_data = self.get_next_feedback()
        if not feedback_data:
            print("\n✅ Aucun feedback à traiter.")
            return

        print("\n" + "🎯" * 15 + f"\n🎯 Traitement Feedback ID: {feedback_data['id']}\n" + "🎯" * 15)
        print(f"   Correction attendue: '{feedback_data['original_prediction']}' → '{feedback_data['label']}'")

        # 1. Créer le dataset de finetuning
        X_text_train, y_train = self.create_targeted_dataset(feedback_data)

        # 2. Remplacer et entraîner la tête du modèle
        finetuned_model, history = self.build_and_train_new_head(X_text_train, y_train)

        # 3. Valider le résultat final
        print("\n" + "🔬" * 15 + "\n🔬 VALIDATION FINALE\n" + "🔬" * 15)
        pred_data = self.prepare_data_for_model([feedback_data['text']])
        new_proba = finetuned_model.predict(pred_data, verbose=0)[0][0]
        new_pred_class = self.label_encoder.inverse_transform([int(new_proba > 0.5)])[0]
        new_confidence = abs(new_proba - 0.5) * 2

        print(f"   Prédiction finale sur le feedback: '{new_pred_class}' (Confiance: {new_confidence:.3f})")

        feedback_corrected = (new_pred_class == feedback_data['label'])
        confidence_sufficient = new_confidence >= self.config['min_confidence_threshold']

        if feedback_corrected and confidence_sufficient:
            print("\n🎉 SUCCÈS ! La correction a été apprise avec succès.")
            self.deploy(finetuned_model, feedback_data)
            self.mark_as_processed(feedback_data['id'], deployed=True)
        else:
            print("\n❌ ÉCHEC. La nouvelle tête n'a pas réussi à corriger le feedback de manière fiable.")
            print(f"   Corrigé: {feedback_corrected}, Confiance Suffisante: {confidence_sufficient}")
            self.mark_as_processed(feedback_data['id'], deployed=False)

    def deploy(self, model_to_deploy, feedback_data):
        """Sauvegarde le nouveau modèle."""
        print("\n" + "🚀" * 15 + " DÉPLOIEMENT " + "🚀" * 15)
        model_to_deploy.save(self.model_dir / "best_lstm_model.keras")
        self.metadata['model_version'] = round(self.metadata.get('model_version', 1.0) + 0.1, 1)
        self.metadata['last_finetuning'] = {'timestamp': datetime.now().isoformat(),
                                            'feedback_id': int(feedback_data['id'])}
        with open(self.model_dir / "model_metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2, default=convert_numpy_types)
        print(f"✅ Modèle déployé ! Version: {self.metadata['model_version']:.1f}")

    def mark_as_processed(self, feedback_id, deployed=False):
        """Met à jour le statut du feedback."""
        df = pd.read_csv(self.feedback_csv_path)
        df.loc[feedback_id, 'processed'] = True
        df.loc[feedback_id, 'processed_at'] = datetime.now().isoformat()
        df.loc[feedback_id, 'deployed'] = deployed
        df.to_csv(self.feedback_csv_path, index=False)
        print(f"📝 Feedback #{feedback_id} marqué comme traité (Déployé: {deployed}).")


# --- Point d'Entrée du Script ---
if __name__ == "__main__":
    print("=" * 60 + "\n🚀 Démarrage du Finetuning AUTORITAIRE\n" + "=" * 60)

    manager = AuthoritativeFinetuner()

    try:
        manager.run()
        print("\n🎉 Processus de finetuning terminé.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)