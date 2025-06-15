"""
simple_classifier.py
Classificateur ML uniquement pour les emails
Suppression du mode aléatoire - API ML ou rien
"""

import requests
from typing import Dict, Any


class EmailClassifier:
    """
    Classificateur d'emails avec modèle ML uniquement
    Pas de mode aléatoire - soit ML soit erreur
    """

    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialise le classificateur ML

        Args:
            api_url: URL de l'API FastAPI du modèle ML
        """
        self.api_url = api_url
        self.stats = {
            "total_classified": 0,
            "spam_detected": 0,
            "important_detected": 0,
            "errors": 0,
        }

    def classify(self, from_addr: str, subject: str, body: str) -> str:
        """ """
        self.stats["total_classified"] += 1

        result = self._classify_ml(from_addr, subject, body)

        # Mettre à jour les stats
        if result == "SPAM":
            self.stats["spam_detected"] += 1
        elif result == "IMPORTANT":
            self.stats["important_detected"] += 1
        else:  # ERROR
            self.stats["errors"] += 1

        return result

    def clean_text(self, text):
        if not text or not isinstance(text, str):
            return ""

        # Supprimer les espaces supplémentaires
        no_extra_spaces = " ".join(text.split())

        # Remplacer les guillemets pour éviter les conflits
        safe_quotes = (
            no_extra_spaces.replace(
                '"', '"'
            ).replace(  # Remplacer les guillemets doubles
                "'", "  "
            )  # Remplacer les guillemets simples
        )
        return safe_quotes

    def _classify_ml(self, from_addr: str, subject: str, body: str) -> str:
        """
        Classification avec modèle ML via API FastAPI

        Args:
            from_addr: Adresse de l'expéditeur
            subject: Sujet de l'email
            body: Corps de l'email

        Returns:
            "SPAM", "IMPORTANT" ou "ERROR"
        """
        # Combiner le contenu de l'email
        email_text = f"From: {from_addr} Subject: {subject} Body: {body}"

        # Nettoyer le texte
        cleaned_text = self.clean_text(email_text)

        try:
            # Appel à l'API ML
            response = requests.post(
                f"{self.api_url}/predict", json={"text": cleaned_text}, timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                # Si phishing détecté → SPAM, sinon → IMPORTANT
                if result["prediction"] == "phishing":
                    return "SPAM"
                else:
                    return "IMPORTANT"
            else:
                print(f"Erreur API ML: Status {response.status_code}")
                return "ERROR"

        except requests.exceptions.ConnectionError:
            print(
                "Erreur : Impossible de se connecter à l'API ML. Vérifiez que l'API est démarrée."
            )
            return "ERROR"
        except requests.exceptions.Timeout:
            print("Erreur : Timeout lors de l'appel à l'API ML.")
            return "ERROR"
        except Exception as e:
            print(f"Erreur lors de l'appel à l'API ML: {e}")
            return "ERROR"

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de classification"""
        total = self.stats["total_classified"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "spam_rate": round((self.stats["spam_detected"] / total) * 100, 2),
            "error_rate": round((self.stats["errors"] / total) * 100, 2),
            "success_rate": round(((total - self.stats["errors"]) / total) * 100, 2),
        }

    def reset_stats(self):
        """Reset les statistiques"""
        self.stats = {
            "total_classified": 0,
            "spam_detected": 0,
            "important_detected": 0,
            "errors": 0,
        }

    def test_api_connection(self) -> bool:
        """Teste la connexion avec l'API ML"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


# Instance globale pour l'utilisation dans d'autres modules
default_classifier = EmailClassifier()


def classify_email(from_addr: str, subject: str, body: str) -> str:
    """
    Fonction de classification pour l'utilisation externe

    Args:
        from_addr: Adresse de l'expéditeur
        subject: Sujet de l'email
        body: Corps de l'email

    Returns:
        "SPAM", "IMPORTANT" ou "ERROR"
    """
    return default_classifier.classify(from_addr, subject, body)


def get_classification_stats() -> Dict[str, Any]:
    """Retourne les statistiques de classification"""
    return default_classifier.get_stats()


def test_ml_api() -> bool:
    """Teste la disponibilité de l'API ML"""
    return default_classifier.test_api_connection()


if __name__ == "__main__":
    # Test du classificateur ML
    print(" Test du classificateur ML")

    # Tester la connexion API
    print("\n Test de connexion à l'API ML...")
    if test_ml_api():
        print(" API ML disponible")
    else:
        print(" API ML non disponible")
        print("Démarrez l'API avec: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        exit(1)

    # Exemples d'emails
    test_emails = [
        ("alice@example.com", "Réunion demain", "Bonjour, nous avons une réunion..."),
        ("spam@fake.com", "URGENT: Gagnez 1000€", "Cliquez ici maintenant..."),
        ("noreply@google.com", "Votre facture", "Merci pour votre achat..."),
        ("promo@spammy.tk", "Offre limitée!!!", "Ne ratez pas cette occasion..."),
    ]

    classifier = EmailClassifier()

    print("\n Résultats de classification ML:")
    for from_addr, subject, body in test_emails:
        result = classifier.classify(from_addr, subject, body)
        print(f"    {subject[:20]:<20} → {result}")

    print(f"\n Statistiques finales:")
    stats = classifier.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n Classification ML uniquement - Pas de mode aléatoire !")
