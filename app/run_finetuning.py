#!/usr/bin/env python3
"""
Script de gestion du fine-tuning pour l'API de détection de phishing

Usage:
    python run_finetuning.py                # Exécuter le fine-tuning
    python run_finetuning.py --check        # Vérifier les conditions
    python run_finetuning.py --analyze      # Analyser les feedbacks
    python run_finetuning.py --deploy       # Déployer le modèle fine-tuné
    python run_finetuning.py --help         # Afficher l'aide
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import subprocess
import shutil
import json

# Ajouter le répertoire courant au path pour importer le module de traitement
sys.path.append('.')

try:
    from traitement import FineTuningManager, FeedbackAnalyzer
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Assurez-vous que traitement.py est dans le même répertoire")
    sys.exit(1)


def check_prerequisites():
    """
    Vérifie les prérequis pour le fine-tuning
    """
    print("🔍 VÉRIFICATION DES PRÉREQUIS")
    print("=" * 40)

    # Vérifier les fichiers du modèle
    model_files = [
        "./model/best_lstm_model.keras",
        "./model/tokenizer.pkl",
        "./model/scaler.pkl",
        "./model/label_encoder.pkl",
        "./model/model_metadata.json"
    ]

    missing_files = []
    for file_path in model_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Fichiers manquants:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ Tous les fichiers du modèle sont présents")

    # Vérifier le fichier de feedbacks
    feedback_file = "./data/user_feedbacks.csv"
    if not Path(feedback_file).exists():
        print(f"❌ Fichier de feedbacks manquant: {feedback_file}")
        return False

    # Compter les feedbacks négatifs
    try:
        df = pd.read_csv(feedback_file)
        negative_count = len(df[
                                 (df['user_satisfaction'] == 'no') &
                                 (df['processed'] == False)
                                 ])

        print(f"📊 Feedbacks négatifs non traités: {negative_count}")

        if negative_count < 5:
            print(f"⚠️ Pas assez de feedbacks négatifs ({negative_count}/5)")
            print("💡 Collectez plus de feedbacks négatifs avant le fine-tuning")
            return False
        else:
            print("✅ Seuil de feedbacks négatifs atteint")

    except Exception as e:
        print(f"❌ Erreur lecture feedbacks: {e}")
        return False

    # Vérifier le dataset principal
    dataset_file = "./data/full_merged_dataset_fr_en_spam.csv"
    if not Path(dataset_file).exists():
        print(f"⚠️ Dataset principal non trouvé: {dataset_file}")
        print("💡 Le fine-tuning utilisera seulement les feedbacks")
    else:
        print("✅ Dataset principal disponible")

    print("\n✅ Tous les prérequis sont satisfaits")
    return True


def run_finetuning():
    """
    Exécute le processus de fine-tuning
    """
    print("🚀 DÉMARRAGE DU FINE-TUNING")
    print("=" * 40)

    # Vérifier les prérequis
    if not check_prerequisites():
        print("❌ Prérequis non satisfaits, arrêt du processus")
        return False

    print("\n🎯 Initialisation du gestionnaire de fine-tuning...")

    try:
        # Initialiser le gestionnaire
        manager = FineTuningManager(
            model_dir="./model",
            data_dir="./data"
        )

        # Exécuter le fine-tuning complet
        success = manager.run_complete_finetuning()

        if success:
            print("\n🎉 FINE-TUNING RÉUSSI!")
            print("=" * 30)
            print("📋 Prochaines étapes:")
            print("   1. Vérifiez les métriques du modèle fine-tuné")
            print("   2. Utilisez --deploy pour déployer le nouveau modèle")
            print("   3. Redémarrez l'API Docker")
            return True
        else:
            print("\n❌ FINE-TUNING ÉCHOUÉ")
            print("💡 Vérifiez les logs pour plus de détails")
            return False

    except Exception as e:
        print(f"❌ Erreur durant le fine-tuning: {e}")
        return False


def analyze_feedbacks():
    """
    Analyse les patterns dans les feedbacks
    """
    print("📊 ANALYSE DES FEEDBACKS")
    print("=" * 30)

    try:
        analyzer = FeedbackAnalyzer("./data/user_feedbacks.csv")
        analyzer.print_analysis()
        return True
    except Exception as e:
        print(f"❌ Erreur analyse: {e}")
        return False


def deploy_finetuned_model():
    """
    Déploie le modèle fine-tuné le plus récent
    """
    print("🚀 DÉPLOIEMENT DU MODÈLE FINE-TUNÉ")
    print("=" * 40)

    # Chercher le modèle fine-tuné le plus récent
    data_dir = Path("./data")
    finetuned_dirs = list(data_dir.glob("finetuned_model_*"))

    if not finetuned_dirs:
        print("❌ Aucun modèle fine-tuné trouvé")
        print("💡 Exécutez d'abord le fine-tuning")
        return False

    # Trier par date (le plus récent en premier)
    finetuned_dirs.sort(reverse=True)
    latest_model_dir = finetuned_dirs[0]

    print(f"📂 Modèle le plus récent: {latest_model_dir}")

    # Vérifier que tous les fichiers nécessaires sont présents
    required_files = [
        "best_lstm_model.keras",
        "tokenizer.pkl",
        "scaler.pkl",
        "label_encoder.pkl",
        "model_metadata.json"
    ]

    missing_files = []
    for file in required_files:
        if not (latest_model_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print("❌ Fichiers manquants dans le modèle fine-tuné:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    # Créer une sauvegarde du modèle actuel
    backup_dir = Path(f"./model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"💾 Sauvegarde du modèle actuel dans: {backup_dir}")

    try:
        shutil.copytree("./model", backup_dir)
        print("✅ Sauvegarde créée")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        print("⚠️ Continuer sans sauvegarde? (y/N): ", end="")
        if input().lower() != 'y':
            return False

    # Remplacer les fichiers du modèle
    print("🔄 Remplacement des fichiers du modèle...")

    try:
        for file in required_files:
            source = latest_model_dir / file
            dest = Path("./model") / file
            shutil.copy2(source, dest)
            print(f"✅ {file} remplacé")

        print("\n🎉 DÉPLOIEMENT RÉUSSI!")
        print("=" * 25)
        print("📋 Actions nécessaires:")
        print("   1. Redémarrez l'API Docker")
        print("   2. Testez les prédictions")
        print("   3. Surveillez les performances")
        print(f"   4. En cas de problème, restaurez depuis: {backup_dir}")

        return True

    except Exception as e:
        print(f"❌ Erreur lors du remplacement: {e}")

        # Tentative de restauration
        if backup_dir.exists():
            print("🔄 Tentative de restauration...")
            try:
                shutil.rmtree("./model")
                shutil.copytree(backup_dir, "./model")
                print("✅ Modèle restauré")
            except Exception as restore_error:
                print(f"❌ Erreur restauration: {restore_error}")
                print("⚠️ ATTENTION: Le modèle pourrait être dans un état incohérent")

        return False


def show_status():
    """
    Affiche le statut actuel du système
    """
    print("📊 STATUT DU SYSTÈME")
    print("=" * 25)

    # Statut du modèle principal
    model_files = [
        "./model/best_lstm_model.keras",
        "./model/tokenizer.pkl",
        "./model/scaler.pkl",
        "./model/label_encoder.pkl",
        "./model/model_metadata.json"
    ]

    model_ok = all(Path(f).exists() for f in model_files)
    print(f"🤖 Modèle principal: {'✅ OK' if model_ok else '❌ Incomplet'}")

    # Statut des feedbacks
    feedback_file = Path("./data/user_feedbacks.csv")
    if feedback_file.exists():
        try:
            df = pd.read_csv(feedback_file)
            total = len(df)
            negative = len(df[df['user_satisfaction'] == 'no'])
            negative_unprocessed = len(df[
                                           (df['user_satisfaction'] == 'no') &
                                           (df['processed'] == False)
                                           ])

            print(f"📝 Feedbacks:")
            print(f"   Total: {total}")
            print(f"   Négatifs: {negative}")
            print(f"   Négatifs non traités: {negative_unprocessed}")
            print(f"   Fine-tuning ready: {'✅ Oui' if negative_unprocessed >= 5 else '❌ Non'}")

        except Exception as e:
            print(f"📝 Feedbacks: ❌ Erreur lecture ({e})")
    else:
        print("📝 Feedbacks: ❌ Fichier non trouvé")

    # Modèles fine-tunés disponibles
    data_dir = Path("./data")
    finetuned_dirs = list(data_dir.glob("finetuned_model_*"))
    print(f"🎯 Modèles fine-tunés: {len(finetuned_dirs)} disponible(s)")

    if finetuned_dirs:
        latest = max(finetuned_dirs)
        print(f"   Plus récent: {latest.name}")


def main():
    """
    Fonction principale avec gestion des arguments
    """
    parser = argparse.ArgumentParser(
        description="Gestionnaire de fine-tuning pour l'API de détection de phishing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python run_finetuning.py                # Exécuter le fine-tuning
  python run_finetuning.py --check        # Vérifier les conditions
  python run_finetuning.py --analyze      # Analyser les feedbacks
  python run_finetuning.py --deploy       # Déployer le modèle
  python run_finetuning.py --status       # Afficher le statut
        """
    )

    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Vérifier les prérequis pour le fine-tuning'
    )

    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Analyser les patterns dans les feedbacks'
    )

    parser.add_argument(
        '--deploy', '-d',
        action='store_true',
        help='Déployer le modèle fine-tuné le plus récent'
    )

    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Afficher le statut du système'
    )

    args = parser.parse_args()

    # Si aucun argument, exécuter le fine-tuning
    if not any([args.check, args.analyze, args.deploy, args.status]):
        return run_finetuning()

    # Exécuter l'action demandée
    if args.status:
        show_status()

    if args.check:
        return check_prerequisites()

    if args.analyze:
        return analyze_feedbacks()

    if args.deploy:
        return deploy_finetuned_model()

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Opération interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        sys.exit(1)