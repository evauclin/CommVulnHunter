# Utiliser une image de base Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements
COPY ./app/requirements.txt /app/requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copier le reste de l'application
COPY ./app /app

# Exposer le port sur lequel l'application FastAPI va tourner
EXPOSE 80

# Commande pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
