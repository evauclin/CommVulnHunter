# feedback_collector.py
"""
Système de collecte de feedback et réentraînement automatique du modèle ML
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sqlite3
import threading
import time
from pathlib import Path


class FeedbackCollector:
    """Collecte et traite les feedbacks utilisateurs"""

    def __init__(self, db_path="feedback.db", model_dir="model/"):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.init_database()

    def init_database(self):
        """Initialise la base de données des feedbacks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedbacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_text TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                user_satisfaction TEXT NOT NULL,
                correct_label TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retraining_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                num_feedbacks INTEGER,
                old_accuracy REAL,
                new_accuracy REAL,
                model_version TEXT,
                status TEXT
            )
        ''')

        conn.commit()
        conn.close()
        print("✅ Base de données feedback initialisée")

    def save_feedback(self, email_text, predicted_label, user_satisfaction,
                      confidence_score=None, correct_label=None):
        """Sauvegarde un feedback utilisateur"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Si l'utilisateur n'est pas satisfait, on déduit le bon label
        if user_satisfaction == "no" and not correct_label:
            correct_label = "benign" if predicted_label == "phishing" else "phishing"
        elif user_satisfaction == "yes":
            correct_label = predicted_label

        cursor.execute('''
            INSERT INTO feedbacks 
            (email_text, predicted_label, user_satisfaction, correct_label, confidence_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (email_text, predicted_label, user_satisfaction, correct_label, confidence_score))

        conn.commit()
        conn.close()

        print(f"📝 Feedback sauvegardé: {user_satisfaction} pour prédiction {predicted_label}")

        # Vérifier si on doit déclencher un réentraînement
        self.check_retraining_trigger()

    def get_feedback_stats(self):
        """Obtient les statistiques des feedbacks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Statistiques générales
        cursor.execute("SELECT COUNT(*) FROM feedbacks")
        total_feedbacks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedbacks WHERE user_satisfaction = 'no'")
        negative_feedbacks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedbacks WHERE processed = FALSE")
        unprocessed_feedbacks = cursor.fetchone()[0]

        # Feedbacks par type de prédiction
        cursor.execute('''
            SELECT predicted_label, user_satisfaction, COUNT(*) 
            FROM feedbacks 
            GROUP BY predicted_label, user_satisfaction
        ''')
        feedback_breakdown = cursor.fetchall()

        conn.close()

        return {
            'total_feedbacks': total_feedbacks,
            'negative_feedbacks': negative_feedbacks,
            'unprocessed_feedbacks': unprocessed_feedbacks,
            'accuracy_rate': (total_feedbacks - negative_feedbacks) / max(total_feedbacks, 1),
            'feedback_breakdown': feedback_breakdown
        }

    def check_retraining_trigger(self):
        """Vérifie si un réentraînement doit être déclenché"""
        stats = self.get_feedback_stats()

        # Conditions de déclenchement du réentraînement
        should_retrain = (
                stats['unprocessed_feedbacks'] >= 50 or  # Au moins 50 nouveaux feedbacks
                (stats['total_feedbacks'] >= 20 and stats['accuracy_rate'] < 0.8)  # Précision < 80%
        )

        if should_retrain:
            print("🚨 Déclenchement du réentraînement automatique")
            self.trigger_retraining()

    def trigger_retraining(self):
        """Déclenche le réentraînement en arrière-plan"""

        def retrain():
            retrainer = ModelRetrainer(self.db_path, self.model_dir)
            retrainer.retrain_model()

        # Lancer en thread séparé pour ne pas bloquer l'API
        thread = threading.Thread(target=retrain)
        thread.daemon = True
        thread.start()

    def export_training_data(self):
        """Exporte les feedbacks comme données d'entraînement"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query('''
            SELECT email_text as text, correct_label as label, confidence_score
            FROM feedbacks 
            WHERE correct_label IS NOT NULL AND processed = FALSE
        ''', conn)

        conn.close()

        return df


class ModelRetrainer:
    """Gère le réentraînement du modèle avec les feedbacks"""

    def __init__(self, db_path="feedback.db", model_dir="model/"):
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.load_model_artifacts()

    def load_model_artifacts(self):
        """Charge le modèle et ses artefacts"""
        try:
            self.model = load_model(self.model_dir / 'best_lstm_model.keras')

            with open(self.model_dir / 'tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)

            with open(self.model_dir / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            with open(self.model_dir / 'label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)

            print("✅ Modèle et artefacts chargés")
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            raise

    def preprocess_feedback_data(self, df):
        """Préprocesse les données de feedback pour l'entraînement"""
        from train_model import LSTMPhishingDetector

        # Utiliser le même preprocessing que l'entraînement initial
        detector = LSTMPhishingDetector()
        detector.tokenizer = self.tokenizer
        detector.scaler = self.scaler
        detector.label_encoder = self.label_encoder

        # Préparer les séquences
        X_text, X_num = detector.prepare_sequences(df, is_training=False)
        y = self.label_encoder.transform(df['label'])

        return X_text, X_num, y

    def incremental_training(self, X_text_new, X_num_new, y_new, epochs=5):
        """Effectue un entraînement incrémental"""
        print(f"🔄 Entraînement incrémental sur {len(y_new)} échantillons...")

        # Compilation avec un learning rate plus faible pour l'ajustement fin
        from tensorflow.keras.optimizers import Adam
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),  # Learning rate réduit
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        # Entraînement incrémental
        history = self.model.fit(
            [X_text_new, X_num_new], y_new,
            batch_size=16,
            epochs=epochs,
            verbose=1,
            validation_split=0.2
        )

        return history

    def evaluate_improvement(self, X_text_test, X_num_test, y_test):
        """Évalue l'amélioration du modèle"""
        y_pred_proba = self.model.predict([X_text_test, X_num_test])
        y_pred = (y_pred_proba > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return accuracy, report

    def retrain_model(self):
        """Processus complet de réentraînement"""
        print("🚀 DÉBUT DU RÉENTRAÎNEMENT AUTOMATIQUE")
        print("=" * 50)

        try:
            # 1. Collecter les feedbacks non traités
            collector = FeedbackCollector(self.db_path)
            feedback_df = collector.export_training_data()

            if len(feedback_df) < 10:
                print("⚠️ Pas assez de feedbacks pour réentraîner (minimum 10)")
                return

            print(f"📊 {len(feedback_df)} nouveaux feedbacks collectés")

            # 2. Préprocesser les données
            X_text_new, X_num_new, y_new = self.preprocess_feedback_data(feedback_df)

            # 3. Évaluation avant réentraînement
            old_accuracy, _ = self.evaluate_improvement(X_text_new, X_num_new, y_new)
            print(f"📈 Précision avant réentraînement: {old_accuracy:.4f}")

            # 4. Sauvegarder le modèle actuel comme backup
            backup_path = self.model_dir / f"backup_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
            self.model.save(backup_path)
            print(f"💾 Sauvegarde créée: {backup_path}")

            # 5. Entraînement incrémental
            history = self.incremental_training(X_text_new, X_num_new, y_new)

            # 6. Évaluation après réentraînement
            new_accuracy, report = self.evaluate_improvement(X_text_new, X_num_new, y_new)
            print(f"📈 Précision après réentraînement: {new_accuracy:.4f}")

            # 7. Décider si on garde le nouveau modèle
            if new_accuracy > old_accuracy * 0.95:  # Au moins 95% de la performance précédente
                # Sauvegarder le nouveau modèle
                new_model_path = self.model_dir / 'best_lstm_model.keras'
                self.model.save(new_model_path)

                # Marquer les feedbacks comme traités
                self.mark_feedbacks_processed()

                # Logger le réentraînement
                self.log_retraining(len(feedback_df), old_accuracy, new_accuracy, "SUCCESS")

                print("✅ Nouveau modèle sauvegardé avec succès")

            else:
                # Restaurer le modèle précédent
                self.model = load_model(backup_path)
                self.log_retraining(len(feedback_df), old_accuracy, new_accuracy, "REVERTED")

                print("❌ Performance dégradée - modèle précédent restauré")

        except Exception as e:
            print(f"❌ Erreur durant le réentraînement: {e}")
            self.log_retraining(0, 0, 0, f"ERROR: {e}")

    def mark_feedbacks_processed(self):
        """Marque les feedbacks comme traités"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("UPDATE feedbacks SET processed = TRUE WHERE processed = FALSE")
        conn.commit()
        conn.close()

        print("✅ Feedbacks marqués comme traités")

    def log_retraining(self, num_feedbacks, old_accuracy, new_accuracy, status):
        """Enregistre les logs de réentraînement"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO retraining_logs 
            (num_feedbacks, old_accuracy, new_accuracy, model_version, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (num_feedbacks, old_accuracy, new_accuracy,
              datetime.now().strftime('%Y%m%d_%H%M%S'), status))

        conn.commit()
        conn.close()


# Intégration avec l'API FastAPI
class FeedbackAPI:
    """Extension de l'API pour gérer les feedbacks"""

    def __init__(self):
        self.collector = FeedbackCollector()

    def submit_feedback(self, email_text: str, predicted_label: str,
                        user_satisfaction: str, confidence_score: float = None):
        """Endpoint pour soumettre un feedback"""
        try:
            self.collector.save_feedback(
                email_text=email_text,
                predicted_label=predicted_label,
                user_satisfaction=user_satisfaction,
                confidence_score=confidence_score
            )

            return {
                "status": "success",
                "message": "Feedback enregistré",
                "feedback_stats": self.collector.get_feedback_stats()
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Erreur lors de l'enregistrement: {e}"
            }

    def get_feedback_dashboard(self):
        """Endpoint pour les statistiques de feedback"""
        stats = self.collector.get_feedback_stats()

        # Historique des réentraînements
        conn = sqlite3.connect(self.collector.db_path)
        retraining_history = pd.read_sql_query(
            "SELECT * FROM retraining_logs ORDER BY timestamp DESC LIMIT 10",
            conn
        ).to_dict('records')
        conn.close()

        return {
            "feedback_stats": stats,
            "retraining_history": retraining_history,
            "last_update": datetime.now().isoformat()
        }


# Script de démarrage du monitoring automatique
def start_feedback_monitoring():
    """Lance le monitoring automatique des feedbacks"""
    collector = FeedbackCollector()

    def monitor_loop():
        while True:
            time.sleep(3600)  # Vérifier toutes les heures
            collector.check_retraining_trigger()

    # Lancer en arrière-plan
    monitor_thread = threading.Thread(target=monitor_loop)
    monitor_thread.daemon = True
    monitor_thread.start()

    print("🔄 Monitoring des feedbacks démarré (vérification horaire)")


if __name__ == "__main__":
    # Test du système de feedback
    print("🧪 Test du système de feedback")

    # Initialiser
    collector = FeedbackCollector()

    # Exemples de feedbacks
    test_feedbacks = [
        ("Urgent: Click here to verify your account", "phishing", "yes", 0.85),
        ("Meeting scheduled for tomorrow at 2 PM", "phishing", "no", 0.75),
        ("Win $1000 now! Click here!", "phishing", "yes", 0.92),
        ("Thanks for the report, great work!", "benign", "yes", 0.65),
    ]

    for email_text, predicted, satisfaction, confidence in test_feedbacks:
        collector.save_feedback(email_text, predicted, satisfaction, confidence)

    # Afficher les stats
    stats = collector.get_feedback_stats()
    print(f"📊 Statistiques: {stats}")

    # Test du réentraînement (avec données factices)
    if stats['unprocessed_feedbacks'] >= 4:
        print("\n🚀 Test du réentraînement...")
        retrainer = ModelRetrainer()
        # retrainer.retrain_model()  # Décommenter pour tester réellement