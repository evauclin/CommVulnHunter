"""
gmail_fetcher.py - Module pour récupérer les emails depuis Gmail
Adapté pour accepter les identifiants utilisateur
"""

import imaplib
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import os
import csv
import json
import re
from datetime import datetime
import hashlib


def fetch_emails_from_gmail(username, password):
    """
    Fetch emails from Gmail using provided credentials
    
    Args:
        username (str): Gmail address
        password (str): Gmail app password
        
    Returns:
        tuple: (success: bool, message: str, emails_data: list)
    """
    
    try:
        # Connexion à Gmail
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)

        emails_data = []

        # 1. Récupérer les emails importants (boîte de réception)
        mail.select("INBOX")
        status, messages = mail.search(None, "ALL")
        
        if messages[0]:
            email_ids = messages[0].split()[-50:]  # 50 derniers emails

            for i, email_id in enumerate(email_ids):
                email_data = process_email(mail, email_id, "IMPORTANT")
                if email_data:
                    emails_data.append(email_data)

        # 2. Récupérer les spams
        try:
            mail.select("[Gmail]/Spam")
            status, messages = mail.search(None, "ALL")
            
            if messages[0]:
                spam_ids = messages[0].split()[-25:]  # 25 derniers spams

                for email_id in spam_ids:
                    email_data = process_email(mail, email_id, "SPAM")
                    if email_data:
                        emails_data.append(email_data)
        except Exception as e:
            pass  # Impossible d'accéder aux spams

        mail.logout()

        # Générer les fichiers CSV et JSON
        success_files = generate_csv_files(emails_data, username)
        
        if success_files:
            message = f"{len(emails_data)} emails récupérés et sauvegardés"
            return True, message, emails_data
        else:
            return False, "Erreur lors de la génération des fichiers", []

    except imaplib.IMAP4.error as e:
        error_msg = f"Erreur de connexion IMAP: {str(e)}"
        return False, error_msg, []
    except Exception as e:
        error_msg = f"Erreur: {str(e)}"
        return False, error_msg, []


def process_email(mail, email_id, email_type):
    """Process an individual email"""
    try:
        res, msg_data = mail.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])

                # Extraction des informations
                subject = decode_header_text(msg.get("Subject", "Sans objet"))
                from_addr = decode_header_text(msg.get("From", "unknown@example.com"))
                to_addr = decode_header_text(msg.get("To", "moi@monemail.com"))
                date_str = format_date(msg.get("Date"))
                body = extract_body(msg)
                message_id = msg.get("Message-ID", "")

                return {
                    "id": generate_email_id(from_addr, subject, date_str),
                    "type": email_type,
                    "from": from_addr,
                    "to": to_addr,
                    "date": date_str,
                    "subject": subject,
                    "body": body,
                    "message_id": message_id,
                    "processed_at": datetime.now().isoformat(),
                }
    except Exception as e:
        return None


def generate_email_id(from_addr, subject, date_str):
    """Generate a unique ID for the email"""
    content = f"{from_addr}-{subject}-{date_str}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]


def decode_header_text(header_value):
    """Decode an email header"""
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
                    decoded_text += part.decode("utf-8", errors="ignore")
            else:
                decoded_text += part
        return clean_text_for_csv(decoded_text.strip())
    except:
        return clean_text_for_csv(str(header_value))


def clean_text_for_csv(text):
    """Clean text for CSV format"""
    if not text:
        return ""

    # Supprimer les caractères de contrôle
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Remplacer les retours à la ligne par des espaces
    text = re.sub(r"\r?\n", " ", text)

    # Nettoyer les espaces multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def format_date(date_str):
    """Format date for CSV"""
    try:
        if date_str:
            date_obj = parsedate_to_datetime(date_str)
            return date_obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_body(msg):
    """Extract email body"""
    body = ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                        break
                elif part.get_content_type() == "text/html" and not body:
                    # Fallback vers HTML si pas de texte plain
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_body = payload.decode("utf-8", errors="ignore")
                        # Extraire le texte du HTML (simple)
                        body = extract_text_from_html(html_body)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                content_type = msg.get_content_type()
                if content_type == "text/html":
                    body = extract_text_from_html(
                        payload.decode("utf-8", errors="ignore")
                    )
                else:
                    body = payload.decode("utf-8", errors="ignore")

        # Nettoyer et limiter le contenu
        body = clean_text_for_csv(body)
        return (
            body[:1000] if body else "Contenu non disponible"
        )  # Limiter à 1000 caractères
    except:
        return "Contenu non disponible"


def extract_text_from_html(html_content):
    """Extract text from HTML content"""
    # Supprimer les scripts et styles
    html_content = re.sub(
        r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE
    )
    html_content = re.sub(
        r"<style[^>]*>.*?</style>", "", html_content, flags=re.DOTALL | re.IGNORECASE
    )

    # Supprimer les balises HTML
    text = re.sub(r"<[^>]+>", " ", html_content)

    # Décoder les entités HTML communes
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')

    return text


def generate_csv_files(emails_data, username=None):
    """Generate CSV and JSON files"""
    
    # Créer le hash de l'email pour le nom du dossier
    if username:
        # Hash de l'email pour le nom du dossier - Normalisation identique à JavaScript
        normalized_email = username.strip().lower()
        email_hash = hashlib.sha256(normalized_email.encode('utf-8')).hexdigest()[:12]
        # Utiliser le chemin monté dans le volume Docker
        output_dir = os.path.join("/shared", "data", "emails", email_hash)
    else:
        # Utiliser le chemin monté dans le volume Docker
        output_dir = "/shared/data"
    
    os.makedirs(output_dir, exist_ok=True)

    csv_file_path = os.path.join(output_dir, "emails_live.csv")
    json_file_path = os.path.join(output_dir, "emails_live.json")

    # Génération des fichiers

    # 1. Fichier CSV principal
    fieldnames = [
        "id",
        "type",
        "from",
        "to",
        "date",
        "subject",
        "body",
        "message_id",
        "processed_at",
    ]

    try:
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL
            )
            writer.writeheader()

            for email_data in emails_data:
                writer.writerow(email_data)

        # Fichier CSV généré

        # 2. Fichier JSON (pour compatibilité et backup)
        json_data = {
            "generated_at": datetime.now().isoformat(),
            "total_emails": len(emails_data),
            "source": "gmail_imap",
            "format_version": "2.0",
            "emails": emails_data,
        }

        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # Fichier JSON généré

        # 3. Génération des statistiques
        generate_statistics(emails_data, output_dir)
        
        return True

    except Exception as e:
        return False


def generate_statistics(emails_data, output_dir):
    """Generate statistics file"""

    stats = {
        "total_emails": len(emails_data),
        "important_emails": len([e for e in emails_data if e["type"] == "IMPORTANT"]),
        "spam_emails": len([e for e in emails_data if e["type"] == "SPAM"]),
        "generated_at": datetime.now().isoformat(),
    }

    # Statistiques par domaine
    domains = {}
    for email_data in emails_data:
        from_addr = email_data.get("from", "")
        if "@" in from_addr:
            domain = from_addr.split("@")[-1].split(">")[0].strip()
            if domain not in domains:
                domains[domain] = {"total": 0, "spam": 0, "important": 0}
            domains[domain]["total"] += 1
            domains[domain][email_data["type"].lower()] += 1

    stats["domains"] = domains
    stats["success_rate"] = (
        round((stats["important_emails"] / stats["total_emails"]) * 100, 2)
        if stats["total_emails"] > 0
        else 0
    )

    # Sauvegarder les statistiques
    stats_file = os.path.join(output_dir, "email_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Statistiques générées


def validate_gmail_credentials(username, password):
    """
    Validate Gmail credentials without fetching emails
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)
        mail.logout()
        return True, "Identifiants Gmail valides"
    except imaplib.IMAP4.error as e:
        return False, f"Identifiants Gmail invalides: {str(e)}"
    except Exception as e:
        return False, f"Erreur de connexion: {str(e)}"