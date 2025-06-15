# 📧 Email Filter CSV - Mode Simple

Version simplifiée avec classification aléatoire, prête pour l'intégration d'un modèle ML.

## 🎯 Approche Simplifiée

- **🎲 Classification aléatoire** : Distribution réaliste 70% IMPORTANT / 30% SPAM
- **📊 Format CSV structuré** : Données prêtes pour l'entraînement ML
- **🔮 Préparé pour ML** : Architecture modulaire pour futurs modèles
- **⚡ Performance optimisée** : Pas de traitement complexe

## 🚀 Installation Express

```bash
# 1. Installer les dépendances
pip install python-dotenv schedule

# 2. Configuration .env
EMAIL_USERNAME=votre@gmail.com
EMAIL_PASSWORD=votre_mot_de_passe_app

# 3. Lancement
python email_fetcher_csv.py
```

## 📁 Structure Simplifiée

```
CommVulnHunter/
├── 📄 email_fetcher_csv.py      # Récupération + classification aléatoire
├── 📄 simple_classifier.py     # Classificateur modulaire
├── 📄 sync_emails_csv.py        # Synchronisation auto
├── src/pages/
│   ├── 📄 index.html           # Interface (mode random)
│   ├── 📄 csvEmailLoader.js    # Chargeur CSV
│   ├── 📄 emails_live.csv      # Données emails
│   └── 📄 email_stats.json     # Stats simples
└── 📄 .env                     # Config Gmail
```

## 🎲 Classification Aléatoire

### Comment ça marche

```python
def classify_email_random(from_addr, subject, body):
    # Seed basé sur le contenu = résultats reproductibles
    content_hash = hash(f"{from_addr}{subject}{body[:100]}")
    random.seed(abs(content_hash) % 1000000)
    
    # Distribution réaliste
    classification = random.choices(
        ['IMPORTANT', 'SPAM'], 
        weights=[70, 30],  # 70% important, 30% spam
        k=1
    )[0]
    
    return classification
```

### Avantages

- **Reproductible** : Même email = même classification
- **Réaliste** : Distribution proche de la réalité
- **Rapide** : Pas de calculs complexes
- **Prévisible** : Pour les tests et démos

## 🔮 Préparation Modèle ML

### Architecture prête

Le système est structuré pour recevoir facilement un modèle ML :

```python
# Dans simple_classifier.py
def _classify_ml(self, from_addr, subject, body):
    """
    TODO: Remplacer par votre modèle ML
    """
    # 1. Préprocessing
    features = self.extract_features(from_addr, subject, body)
    
    # 2. Prédiction
    prediction = self.ml_model.predict(features)
    
    # 3. Post-processing
    return "SPAM" if prediction > 0.5 else "IMPORTANT"
```

### Format de données parfait

Le CSV généré est prêt pour l'entraînement :

| Colonne | Usage ML | Description |
|---------|----------|-------------|
| `from` | Feature | Domaine, format d'adresse |
| `subject` | Feature | Mots-clés, longueur, ponctuation |
| `body` | Feature | Contenu, liens, structure |
| `type` | Label | SPAM/IMPORTANT (target) |
| `date` | Feature | Patterns temporels |

## 📊 Interface Simplifiée

### Dashboard adapté

- **🎲 Bouton "ANALYSER (RANDOM)"** : Indique clairement le mode
- **📈 Statistiques réalistes** : Basées sur la distribution 70/30
- **🔍 Fonctionnalités intactes** : Recherche, filtres, aperçu
- **⚡ Performance** : Chargement instantané

### Utilisation

1. **Ouvrir** `src/pages/index.html`
2. **Charger** les emails CSV automatiquement
3. **Tester** la classification aléatoire
4. **Analyser** les patterns dans les données

## 🛠️ Commandes Rapides

```bash
# Récupération unique
python email_fetcher_csv.py

# Synchronisation continue
python sync_emails_csv.py

# Test du classificateur
python simple_classifier.py

# Interface web
open src/pages/index.html
```

## 🔄 Migration vers ML

Quand vous êtes prêt pour un vrai modèle :

### 1. Entraînement

```python
# Utiliser emails_live.csv comme dataset
import pandas as pd

data = pd.read_csv('src/pages/emails_live.csv')
# ... entraînement de votre modèle
```

### 2. Intégration

```python
# Dans simple_classifier.py
def _classify_ml(self, from_addr, subject, body):
    # Remplacer cette fonction par votre modèle
    return votre_modele.predict(from_addr, subject, body)
```

### 3. Activation

```python
# Changer le mode
set_classification_mode("ml")
```

## 📈 Avantages de cette Approche

### Pour le développement
- **🚀 Démarrage rapide** : Pas besoin de modèle complexe
- **🧪 Tests faciles** : Résultats prévisibles
- **📊 Données propres** : CSV structuré et exploitable

### Pour la production
- **🔄 Migration fluide** : Architecture prête pour ML
- **📈 Évolutif** : Ajout de features simple
- **🛡️ Fiable** : Pas de dépendances ML complexes

## 🎯 Points Clés

1. **Classification aléatoire reproductible** avec seed basé sur le contenu
2. **Interface adaptée** pour indiquer le mode simple
3. **Architecture modulaire** prête pour ML
4. **Données CSV parfaites** pour l'entraînement futur
5. **Performance optimale** sans calculs complexes

---

✨ **Version Simple - Prête pour l'évolution vers un modèle ML avancé** ✨# CommVulnHunter


## commande to build and run the image in local
```bash
docker-compose up --build 
```
## you can deploy these containers on aws with terraform
```bash
terraform init
terraform plan
terraform apply
```