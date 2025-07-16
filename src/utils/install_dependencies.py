#!/usr/bin/env python3
"""
install_dependencies.py
Automatic dependency installation script for Email Filter
"""

import subprocess
import sys
import os


def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def check_python_version():
    """Check the Python version"""
    if sys.version_info < (3, 6):
        print("❌ Python 3.6+ requis. Version actuelle:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} détecté")
    return True


def main():
    print("🚀 Installation des dépendances pour Email Filter")
    print("=" * 50)

    # Vérification de Python
    if not check_python_version():
        sys.exit(1)

    # Liste des dépendances nécessaires
    dependencies = [
        "python-dotenv",  # Gestion des variables d'environnement
        "pandas",  # Manipulation de données (optionnel)
    ]

    print("\n📦 Installation des dépendances...")

    success_count = 0
    for package in dependencies:
        print(f"   Installation de {package}...", end=" ")
        if install_package(package):
            print("✅")
            success_count += 1
        else:
            print("❌")

    print(f"\n📊 Résultat: {success_count}/{len(dependencies)} packages installés")

    # Vérification de la configuration
    print("\n🔧 Vérification de la configuration...")

    if not os.path.exists(".env"):
        print("⚠️  Fichier .env manquant")
        print("   Créez un fichier .env avec vos identifiants Gmail")

        # Proposer de créer le fichier .env
        create_env = input("   Voulez-vous créer le fichier .env maintenant ? (o/n): ")
        if create_env.lower() in ["o", "oui", "y", "yes"]:
            username = input("   Email Gmail: ")
            password = input("   Mot de passe d'application: ")

            with open(".env", "w") as f:
                f.write(f"EMAIL_USERNAME={username}\n")
                f.write(f"EMAIL_PASSWORD={password}\n")

            print("✅ Fichier .env créé")
    else:
        print("✅ Fichier .env trouvé")

    # Vérification des répertoires
    os.makedirs("src/pages", exist_ok=True)
    os.makedirs("src/pages/docs/javascript", exist_ok=True)
    print("✅ Structure de répertoires vérifiée")

    print("\n🎉 Installation terminée!")
    print("\n📋 Prochaines étapes:")
    print("   1. Configurez votre fichier .env avec vos identifiants Gmail")
    print("   2. Lancez: python email_fetcher.py")
    print("   3. Ouvrez votre application web")


if __name__ == "__main__":
    main()
