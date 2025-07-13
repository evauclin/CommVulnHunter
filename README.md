Voici une version améliorée et précise du fichier README pour le projet "CommVulnHunter", qui reflète l'architecture et les fonctionnalités avancées décrites dans les fichiers fournis.

***

# CommVulnHunter - Détecteur de Phishing Auto-Améliorable

**CommVulnHunter** est une solution complète de détection de phishing et de spam basée sur l'intelligence artificielle. Sa caractéristique principale est un **système d'apprentissage en boucle fermée** qui lui permet de s'améliorer continuellement à partir des retours des utilisateurs, en ré-entraînant et en redéployant automatiquement son modèle de Machine Learning sans interruption de service.

## 🎯 Fonctionnalités Clés

-   **🤖 Détection par IA** : Utilise un modèle de deep learning (LSTM) pour analyser et classifier les emails (Phishing/Spam ou Important).
-   **🔄 Apprentissage en Boucle Fermée** : Le système apprend des erreurs signalées par les utilisateurs. Un retour négatif déclenche un processus d'amélioration automatique.
-   **🧠 Auto-Fine-Tuning** : Le modèle est automatiquement ré-entraîné avec les données de feedback pour corriger ses erreurs et s'adapter à de nouvelles menaces.
-   **⚡ Déploiement à Chaud** : Le nouveau modèle amélioré est déployé et chargé par l'API sans nécessiter de redémarrage, garantissant une haute disponibilité.
-   **📊 Tableau de Bord Interactif** : Une interface web complète pour visualiser les emails, les filtrer, effectuer des recherches et interagir avec le modèle d'IA.
-   **⚙️ Traitement par Lots** : Capacité à analyser des fichiers CSV contenant des milliers d'emails et à les ré-étiqueter selon les prédictions du modèle.
-   **☁️ Prêt pour le Cloud** : Inclut une configuration Terraform pour un déploiement simple sur AWS.

## 🏗️ Architecture du Système

Le projet est articulé autour de trois composants principaux qui créent une boucle d'amélioration continue :

1.  **Frontend (Dashboard Web)** (`index.html`, `csvEmailLoader.js`)
    -   Interface utilisateur pour visualiser les emails et les statistiques.
    -   Permet à l'utilisateur de demander une analyse IA pour un email sélectionné.
    -   **Point crucial** : Permet à l'utilisateur de soumettre un feedback ("Correct" / "Incorrect") sur la classification de l'IA.

2.  **Backend (API FastAPI)** (`main.py`)
    -   Sert le modèle de Machine Learning via des endpoints RESTful (`/predict/...`).
    -   Reçoit et sauvegarde les feedbacks des utilisateurs dans un fichier `user_feedbacks.csv`.
    -   Déclenche le script de fine-tuning lorsque les conditions sont remplies (ex: un certain nombre de feedbacks négatifs).
    -   Possède un endpoint (`/reload-model`) pour recharger à chaud le modèle depuis le disque après un ré-entraînement.

3.  **Moteur de Fine-Tuning** (`traitement.py`)
    -   Script indépendant qui agit comme le "cerveau" de l'apprentissage.
    -   Lit les feedbacks négatifs non traités.
    -   Crée un micro-dataset d'entraînement ciblé pour corriger une erreur spécifique.
    -   Ré-entraîne le modèle de production existant.
    -   Valide que le nouveau modèle corrige bien l'erreur sans régresser sur d'autres exemples.
    -   Si la validation est réussie, il **déploie le nouveau modèle** en remplaçant l'ancien.
    -   Enfin, il notifie l'API via un appel HTTP pour qu'elle charge le nouveau modèle.



## 🛠️ Technologies Utilisées

-   **Backend** : Python, FastAPI, TensorFlow/Keras, Pandas, NLTK
-   **Frontend** : HTML5, JavaScript, Bootstrap 5
-   **Déploiement** : Docker, Docker Compose, Terraform

## 🚀 Installation et Lancement

### Prérequis

-   Docker
-   Docker Compose

### Déploiement Local

Le moyen le plus simple de lancer l'ensemble des services (API, frontend) est d'utiliser Docker Compose.

```bash
# Construire et lancer les conteneurs en arrière-plan
docker-compose up --build -d
```

-   Le tableau de bord sera accessible à l'adresse : `http://localhost:80` (ou un autre port que vous configurez).
-   L'API sera accessible sur `http://localhost:8000`.

### Déploiement sur AWS avec Terraform

Le projet inclut une configuration pour déployer l'infrastructure sur AWS.

```bash
# Se placer dans le dossier terraform
cd terraform

# Initialiser Terraform
terraform init

# Planifier l'infrastructure (vérifier ce qui sera créé)
terraform plan

# Appliquer la configuration pour créer l'infrastructure
terraform apply
```

## 🕹️ Comment Utiliser

1.  **Lancer l'application** en utilisant l'une des méthodes ci-dessus.
2.  **Accéder au tableau de bord** dans votre navigateur.
3.  **Charger les emails** en utilisant l'un des boutons :
    -   `Charger avec IA` : Envoie le fichier `emails_live.csv` à l'API pour une analyse complète et retourne les résultats.
    -   `CSV Brut` : Charge les données sans passer par l'IA.
    -   `Demo` : Charge un jeu de données de démonstration.
4.  **Sélectionner un email** dans la liste pour voir ses détails.
5.  **Analyser l'email** : Cliquer sur le bouton `ANALYSER AVEC IA`. Le résultat de la classification (IMPORTANT ou SPAM) s'affichera.
6.  **Donner un feedback** : Si la classification est incorrecte, utilisez les boutons "pouce" (👍 / 👎) pour signaler l'erreur.
7.  **Observer l'apprentissage** : Dans les logs du conteneur de l'API, vous verrez le feedback être enregistré. Lorsque le script `traitement.py` s'exécute, il utilisera ce feedback pour améliorer et redéployer le modèle automatiquement.

## 📄 Rôles des Fichiers Clés

-   `main.py`: Le serveur API FastAPI qui expose le modèle ML et gère les feedbacks.
-   `traitement.py`: Le script autonome pour le ré-entraînement, la validation et le déploiement du modèle.
-   `index.html`: Le tableau de bord web interactif.
-   `csvEmailLoader.js`: La logique frontend pour charger et formater les données CSV.
-   `docker-compose.yml`: Fichier de configuration pour orchestrer les conteneurs localement.
-   `README.md`: Ce fichier.
