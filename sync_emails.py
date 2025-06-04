#!/usr/bin/env python3
"""
sync_emails.py
Script simple pour synchroniser automatiquement les emails
"""

import time
import schedule
from email_fetcher import fetch_emails_from_gmail
from datetime import datetime
import os


def sync_job():
    """Tâche de synchronisation"""
    print(f"\n🔄 Synchronisation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        emails = fetch_emails_from_gmail()
        if emails:
            print(f"✅ {len(emails)} emails synchronisés")
        else:
            print("⚠️ Aucun email récupéré")
    except Exception as e:
        print(f"❌ Erreur de synchronisation: {e}")


def main():
    """Fonction principale"""
    print("📅 Planificateur d'emails démarré")
    print("🔄 Synchronisation toutes les 30 minutes")
    print("⏹️  Ctrl+C pour arrêter")

    # Planifier la synchronisation toutes les 30 minutes
    schedule.every(30).minutes.do(sync_job)

    # Synchronisation initiale
    sync_job()

    # Boucle principale
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Vérifier toutes les minutes
    except KeyboardInterrupt:
        print("\n🛑 Planificateur arrêté")


if __name__ == "__main__":
    # Vérifier la configuration
    if not os.path.exists(".env"):
        print("❌ Fichier .env manquant")
        print("Exécutez d'abord: python install_dependencies.py")
        exit(1)

    main()