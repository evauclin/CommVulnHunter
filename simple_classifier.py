"""
simple_classifier.py
Classificateur simple pour les emails avec mode aléatoire
Préparé pour l'intégration d'un modèle ML futur
"""

import random
import re
from typing import Dict, Any
import requests


class EmailClassifier:
    """
    Classificateur d'emails simple
    Mode aléatoire en attendant l'implémentation d'un modèle ML
    """

    def __init__(self, mode: str = "random"):
        """
        Initialise le classificateur

        Args:
            mode: "random" pour classification aléatoire, "ml" pour modèle ML (futur)
        """
        self.mode = mode
        self.stats = {
            "total_classified": 0,
            "spam_detected": 0,
            "important_detected": 0
        }

    def classify(self, from_addr: str, subject: str, body: str) -> str:
        """
        Classifie un email

        Args:
            from_addr: Adresse de l'expéditeur
            subject: Sujet de l'email
            body: Corps de l'email

        Returns:
            "SPAM" ou "IMPORTANT"
        """
        self.stats["total_classified"] += 1

        if self.mode == "random":
            result = self._classify_random(from_addr, subject, body)
        elif self.mode == "ml":
            result = self._classify_ml(from_addr, subject, body)
        else:
            # Fallback mode aléatoire
            result = self._classify_random(from_addr, subject, body)

        # Mettre à jour les stats
        if result == "SPAM":
            self.stats["spam_detected"] += 1
        else:
            self.stats["important_detected"] += 1

        return result

    def _classify_random(self, from_addr: str, subject: str, body: str) -> str:
        """
        Classification aléatoire avec seed reproductible
        """
        # Créer un seed basé sur le contenu pour la reproductibilité
        content_hash = hash(f"{from_addr}{subject}{body[:100]}")
        random.seed(abs(content_hash) % 1000000)

        # Distribution réaliste : 70% IMPORTANT, 30% SPAM
        classification = random.choices(
            ['IMPORTANT', 'SPAM'],
            weights=[70, 30],
            k=1
        )[0]

        # Reset du seed
        random.seed()

        return classification

    def _classify_ml(self, from_addr: str, subject: str, body: str) -> str:
        """
        Classification avec modèle ML (à implémenter)

        TODO: Intégrer ici votre modèle de machine learning
        - Préprocessing des données
        - Vectorisation du texte
        - Prédiction avec le modèle
        - Post-processing des résultats
        """
        email_text = f"From: {from_addr} Subject: {subject} Body: {body}"

        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"text": email_text},
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()

                # Si phishing détecté → SPAM, sinon → IMPORTANT
                if result["prediction"] == "phishing":
                    return "SPAM"
                else:
                    return "IMPORTANT"
            else:
                return "IMPORTANT"
        except requests.RequestException as e:
            print(f"Erreur lors de l'appel à l'API de classification ML: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de classification"""
        total = self.stats["total_classified"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "spam_rate": round((self.stats["spam_detected"] / total) * 100, 2),
            "accuracy_rate": "N/A (mode aléatoire)" if self.mode == "random" else "TBD"
        }

    def reset_stats(self):
        """Reset les statistiques"""
        self.stats = {
            "total_classified": 0,
            "spam_detected": 0,
            "important_detected": 0
        }


# Instance globale pour l'utilisation dans d'autres modules
default_classifier = EmailClassifier(mode="random")


def classify_email(from_addr: str, subject: str, body: str) -> str:
    """
    Fonction de classification simple pour l'utilisation externe

    Args:
        from_addr: Adresse de l'expéditeur
        subject: Sujet de l'email
        body: Corps de l'email

    Returns:
        "SPAM" ou "IMPORTANT"
    """
    return default_classifier.classify(from_addr, subject, body)


def get_classification_stats() -> Dict[str, Any]:
    """Retourne les statistiques de classification"""
    return default_classifier.get_stats()


def set_classification_mode(mode: str):
    """
    Change le mode de classification

    Args:
        mode: "random" ou "ml"
    """
    global default_classifier
    default_classifier.mode = mode
    print(f"🔄 Mode de classification changé vers: {mode}")


# Fonction legacy pour compatibilité
def classify_email_random(from_addr: str, subject: str, body: str) -> str:
    """Classification aléatoire (fonction legacy)"""
    return classify_email(from_addr, subject, body)


if __name__ == "__main__":
    # Test du classificateur
    print("🧪 Test du classificateur d'emails")

    # Exemples d'emails
    test_emails = [
        ("alice@example.com", "Réunion demain", "Bonjour, nous avons une réunion..."),
        ("spam@fake.com", "URGENT: Gagnez 1000€", "Cliquez ici maintenant..."),
        ("noreply@google.com", "Votre facture", "Merci pour votre achat..."),
        ("promo@spammy.tk", "Offre limitée!!!", "Ne ratez pas cette occasion..."),
    ]

    classifier = EmailClassifier(mode="random")

    print("\n📊 Résultats de classification:")
    for from_addr, subject, body in test_emails:
        result = classifier.classify(from_addr, subject, body)
        print(f"   📧 {subject[:20]:<20} → {result}")

    print(f"\n📈 Statistiques: {classifier.get_stats()}")

    print("\n✨ Prêt pour l'intégration d'un modèle ML !")