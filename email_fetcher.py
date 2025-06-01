"""
email_fetcher.py
Script simple pour récupérer les emails depuis Gmail
et les convertir au format compatible avec votre application existante
"""

import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import os
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


def fetch_emails_from_gmail():
    """Récupère les emails depuis Gmail et les formate pour l'application"""

    # Configuration Gmail
    username = os.getenv('EMAIL_USERNAME', 'farinesfari@gmail.com')
    password = os.getenv('EMAIL_PASSWORD', 'aelk usor ixat xoyd')

    try:
        # Connexion à Gmail
        print("🔌 Connexion à Gmail...")
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)
        print("✅ Connexion réussie")

        emails_data = []

        # 1. Récupérer les emails importants (boîte de réception)
        mail.select("INBOX")
        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()[-30:]  # 30 derniers emails

        for email_id in email_ids:
            email_data = process_email(mail, email_id, "IMPORTANT")
            if email_data:
                emails_data.append(email_data)

        # 2. Récupérer les spams
        try:
            mail.select("[Gmail]/Spam")
            status, messages = mail.search(None, "ALL")
            spam_ids = messages[0].split()[-15:]  # 15 derniers spams

            for email_id in spam_ids:
                email_data = process_email(mail, email_id, "SPAM")
                if email_data:
                    emails_data.append(email_data)
        except:
            print("⚠️ Impossible d'accéder aux spams")

        mail.logout()

        # Générer les fichiers pour l'application
        generate_files(emails_data)

        print(f"✅ {len(emails_data)} emails récupérés avec succès")
        return emails_data

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return []


def process_email(mail, email_id, email_type):
    """Traite un email individuel"""
    try:
        res, msg_data = mail.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])

                # Extraction des informations
                subject = decode_header_text(msg.get("Subject", "Sans objet"))
                from_addr = decode_header_text(msg.get("From", "unknown@example.com"))
                date_str = format_date(msg.get("Date"))
                body = extract_body(msg)

                return {
                    "type": email_type,
                    "from": from_addr,
                    "to": "moi@monemail.com",
                    "date": date_str,
                    "subject": subject,
                    "body": body
                }
    except Exception as e:
        print(f"⚠️ Erreur traitement email: {e}")
        return None


def decode_header_text(header_value):
    """Décode un en-tête d'email"""
    if not header_value:
        return ""

    try:
        decoded_parts = decode_header(header_value)
        decoded_text = ""
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                if encoding:
                    decoded_text += part.decode(encoding)
                else:
                    decoded_text += part.decode('utf-8', errors='ignore')
            else:
                decoded_text += part
        return decoded_text.strip()
    except:
        return str(header_value)


def format_date(date_str):
    """Formate la date"""
    try:
        if date_str:
            date_obj = parsedate_to_datetime(date_str)
            return date_obj.strftime("%Y-%m-%d %H:%M")
        else:
            return datetime.now().strftime("%Y-%m-%d %H:%M")
    except:
        return datetime.now().strftime("%Y-%m-%d %H:%M")


def extract_body(msg):
    """Extrait le corps de l'email"""
    body = ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode('utf-8', errors='ignore')
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode('utf-8', errors='ignore')

        # Nettoyage
        body = body.strip()[:500]  # Limiter à 500 caractères
        return body or "Contenu non disponible"
    except:
        return "Contenu non disponible"


def generate_files(emails_data):
    """Génère les fichiers pour l'application web"""

    # CHEMINS CORRIGÉS - Vers le dossier src/pages/
    text_output_path = "src/pages/archive_mails_live.txt"
    json_output_path = "src/pages/live_emails.json"

    # Créer le dossier s'il n'existe pas
    os.makedirs("src/pages", exist_ok=True)

    # 1. Fichier texte (format compatible avec votre application actuelle)
    text_content = []
    for email_data in emails_data:
        email_block = f"""[{email_data['type']}]
From: {email_data['from']}
To: {email_data['to']}
Date: {email_data['date']}
Subject: {email_data['subject']}

{email_data['body']}

--------------------------------------------"""
        text_content.append(email_block)

    # Sauvegarde du fichier texte
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_content))

    print(f"✅ Fichier généré: {text_output_path}")

    # 2. Fichier JSON (pour une utilisation future)
    import json
    json_data = {
        "generated_at": datetime.now().isoformat(),
        "total_emails": len(emails_data),
        "emails": emails_data
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Fichier généré: {json_output_path}")


if __name__ == "__main__":
    print("📧 Récupération des emails depuis Gmail...")

    # Vérification de la configuration
    if not os.getenv("EMAIL_PASSWORD"):
        print("⚠️ Créez un fichier .env avec EMAIL_PASSWORD=votre_mot_de_passe")

    emails = fetch_emails_from_gmail()
    print("🎉 Terminé!")