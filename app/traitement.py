



def mark_negative_feedbacks_processed():
    """Marque UNIQUEMENT les feedbacks négatifs comme traités"""
    try:
        if not FEEDBACK_CSV_PATH.exists():
            return

        df = pd.read_csv(FEEDBACK_CSV_PATH)

        # Marquer seulement les feedbacks négatifs ET non traités
        mask = (df['user_satisfaction'] == 'no') & (df['processed'] == False)
        df.loc[mask, 'processed'] = True

        processed_count = mask.sum()

        df.to_csv(FEEDBACK_CSV_PATH, index=False)

        print(f"✅ {processed_count} feedbacks négatifs marqués comme traités")
        print(f"📝 Feedbacks positifs conservés pour analyse future")

    except Exception as e:
        print(f"❌ Erreur lors du marquage des feedbacks négatifs: {e}")


def evaluate_and_compare_models(new_detector, X_text_test, X_num_test, y_test):
    """
    Compare les performances du nouveau modèle vs ancien modèle
    Retourne True si le nouveau modèle est meilleur
    """
    try:
        print("🏆 COMPARAISON DES PERFORMANCES")
        print("=" * 40)

        # 1. Évaluer l'ancien modèle (modèle actuel)
        global model
        if model is None:
            print("⚠️ Aucun modèle ancien à comparer, acceptation du nouveau")
            return True

        y_test_encoded = new_detector.label_encoder.transform(y_test)

        # Prédictions ancien modèle
        old_pred_proba = model.predict([X_text_test, X_num_test], verbose=0)
        old_pred = (old_pred_proba > 0.5).astype(int)

        # Prédictions nouveau modèle
        new_pred_proba = new_detector.model.predict([X_text_test, X_num_test], verbose=0)
        new_pred = (new_pred_proba > 0.5).astype(int)

        # Calculer les métriques
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        old_accuracy = accuracy_score(y_test_encoded, old_pred)
        new_accuracy = accuracy_score(y_test_encoded, new_pred)

        old_f1 = f1_score(y_test_encoded, old_pred)
        new_f1 = f1_score(y_test_encoded, new_pred)

        old_precision = precision_score(y_test_encoded, old_pred)
        new_precision = precision_score(y_test_encoded, new_pred)

        old_recall = recall_score(y_test_encoded, old_pred)
        new_recall = recall_score(y_test_encoded, new_pred)

        # Affichage des résultats
        print(f"📊 ANCIEN MODÈLE:")
        print(f"  Accuracy:  {old_accuracy:.4f}")
        print(f"  F1-Score:  {old_f1:.4f}")
        print(f"  Precision: {old_precision:.4f}")
        print(f"  Recall:    {old_recall:.4f}")

        print(f"📊 NOUVEAU MODÈLE:")
        print(f"  Accuracy:  {new_accuracy:.4f}")
        print(f"  F1-Score:  {new_f1:.4f}")
        print(f"  Precision: {new_precision:.4f}")
        print(f"  Recall:    {new_recall:.4f}")

        # Calculer l'amélioration
        accuracy_improvement = new_accuracy - old_accuracy
        f1_improvement = new_f1 - old_f1

        print(f"📈 AMÉLIORATION:")
        print(f"  Accuracy:  {accuracy_improvement:+.4f}")
        print(f"  F1-Score:  {f1_improvement:+.4f}")

        # Critères de décision (vous pouvez ajuster ces seuils)
        MIN_ACCURACY_IMPROVEMENT = 0.01  # 1% d'amélioration minimum
        MIN_F1_IMPROVEMENT = 0.01

        is_better = (
                accuracy_improvement >= MIN_ACCURACY_IMPROVEMENT or
                f1_improvement >= MIN_F1_IMPROVEMENT
        )

        if is_better:
            print("✅ NOUVEAU MODÈLE ACCEPTÉ - Performances améliorées!")
        else:
            print("❌ NOUVEAU MODÈLE REJETÉ - Pas d'amélioration significative")

        return is_better, {
            'old_accuracy': old_accuracy,
            'new_accuracy': new_accuracy,
            'old_f1': old_f1,
            'new_f1': new_f1,
            'accuracy_improvement': accuracy_improvement,
            'f1_improvement': f1_improvement
        }

    except Exception as e:
        print(f"❌ Erreur lors de la comparaison: {e}")
        # En cas d'erreur, accepter le nouveau modèle par sécurité
        return True, {}


# --- Fonctions de Gestion des Feedbacks Intelligents ---
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


def prepare_negative_feedback_dataset() -> Optional[pd.DataFrame]:
    """Prépare un dataset UNIQUEMENT avec les feedbacks négatifs"""
    try:
        if not FEEDBACK_CSV_PATH.exists():
            return None

        df = pd.read_csv(FEEDBACK_CSV_PATH)

        # Filtrer: feedbacks négatifs ET non traités
        negative_df = df[
            (df['user_satisfaction'] == 'no') &
            (df['processed'] == False)
            ].copy()

        if len(negative_df) == 0:
            return None

        # Pour les feedbacks négatifs, inverser la prédiction = label correct
        def get_correct_label(row):
            return 'legitimate' if row['predicted_class'] == 'phishing' else 'phishing'

        negative_df['correct_label'] = negative_df.apply(get_correct_label, axis=1)

        # Créer le dataset final
        retrain_df = pd.DataFrame({
            'text': negative_df['email_text'],
            'label': negative_df['correct_label'],
            'language': negative_df['language_detected']
        })

        print(f"📋 Dataset de feedbacks négatifs: {len(retrain_df)} échantillons")
        print(f"  Distribution: {retrain_df['label'].value_counts().to_dict()}")

        return retrain_df

    except Exception as e:
        print(f"❌ Erreur lors de la préparation du dataset négatif: {e}")
        return None