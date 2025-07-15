# 🔐 Nouveau Système d'Authentification CommVulnHunter

## Vue d'ensemble

Ce nouveau système d'authentification offre une sécurité renforcée avec:
- **JWT tokens** sécurisés avec refresh tokens
- **Base de données SQLite** pour la gestion des utilisateurs
- **Hashage bcrypt** des mots de passe
- **API REST complète** avec FastAPI
- **Interface utilisateur moderne** et responsive
- **Audit de sécurité** avec logging complet
- **Protection contre les attaques** (rate limiting, headers sécurisés)

## 🚀 Installation et Démarrage

### 1. Installer les dépendances

```bash
pip install -r auth_system/requirements.txt
```

### 2. Démarrer le serveur d'authentification

```bash
python auth_system/start_auth_server.py
```

Le serveur sera disponible sur: http://localhost:9000

### 3. Accéder à l'interface

- **Page de connexion**: http://localhost:8080/new_login.html
- **Page d'inscription**: http://localhost:8080/new_register.html  
- **Dashboard sécurisé**: http://localhost:8080/new_dashboard.html
- **Documentation API**: http://localhost:9000/docs

## 👥 Comptes par défaut

Le système crée automatiquement des comptes de test:

| Rôle | Email | Mot de passe | Permissions |
|------|-------|-------------|-------------|
| **Admin** | admin@emailfilter.com | admin123 | Toutes les permissions |
| **User** | user@emailfilter.com | user123 | Utilisateur standard |
| **Demo** | demo@emailfilter.com | demo123 | Accès limité |

## 🏗️ Architecture

### Backend (Port 9000)
```
auth_system/
├── database.py          # Modèles de données et connexion SQLite
├── jwt_handler.py       # Gestion des tokens JWT et sécurité
├── auth_api.py          # API REST avec FastAPI
├── start_auth_server.py # Script de démarrage
├── requirements.txt     # Dépendances Python
├── auth.db             # Base de données SQLite (créée automatiquement)
└── audit.log           # Logs d'audit sécurisé
```

### Frontend (Port 8080)
```
src/pages/
├── new_login.html      # Page de connexion moderne
├── new_register.html   # Page d'inscription avec validation
└── new_dashboard.html  # Dashboard sécurisé avec JWT
```

## 🔧 Fonctionnalités

### Authentification
- ✅ **Inscription sécurisée** avec validation des mots de passe
- ✅ **Connexion JWT** avec access et refresh tokens
- ✅ **Session management** automatique avec refresh
- ✅ **Déconnexion sécurisée** avec invalidation des tokens
- ✅ **Protection des routes** côté frontend et backend
- ✅ **Rate limiting** pour prévenir les attaques brute force

### Gestion des utilisateurs
- ✅ **Profils utilisateur** complets avec rôles
- ✅ **Changement de mot de passe** sécurisé
- ✅ **Récupération de mot de passe** (tokens temporaires)
- ✅ **Verrouillage de compte** après tentatives échouées
- ✅ **Administration** des utilisateurs (pour les admins)

### Sécurité
- ✅ **Hashage bcrypt** des mots de passe
- ✅ **Headers de sécurité** HTTP complets
- ✅ **CORS sécurisé** avec origins autorisées
- ✅ **Audit logging** de toutes les actions
- ✅ **Validation stricte** des entrées utilisateur
- ✅ **Protection XSS/CSRF** intégrée

### Interface utilisateur
- ✅ **Design moderne** avec Bootstrap 5
- ✅ **Responsive design** mobile-friendly
- ✅ **Feedback temps réel** sur les formulaires
- ✅ **Validation mot de passe** visuelle
- ✅ **Indicateurs de force** du mot de passe
- ✅ **Comptes de démonstration** intégrés

## 📡 API Endpoints

### Authentification
- `POST /auth/register` - Inscription
- `POST /auth/login` - Connexion
- `POST /auth/refresh` - Renouvellement de token
- `POST /auth/logout` - Déconnexion
- `POST /auth/logout-all` - Déconnexion de tous les appareils

### Gestion du profil
- `GET /auth/me` - Informations utilisateur
- `POST /auth/change-password` - Changer le mot de passe
- `POST /auth/forgot-password` - Demande de réinitialisation
- `POST /auth/reset-password` - Réinitialisation avec token

### Administration (Admin uniquement)
- `GET /auth/admin/users` - Liste des utilisateurs
- `PUT /auth/admin/users/{id}/role` - Modifier le rôle
- `DELETE /auth/admin/users/{id}` - Désactiver un utilisateur

### Monitoring
- `GET /auth/health` - État de santé du service

## 🔒 Modèle de sécurité

### Tokens JWT
- **Access Token**: 30 minutes de validité
- **Refresh Token**: 30 jours de validité
- **Algorithme**: HS256 avec clé secrète
- **Payload**: ID utilisateur, email, nom, rôle

### Protection des mots de passe
- **Longueur minimale**: 8 caractères
- **Complexité requise**: Majuscule, minuscule, chiffre, caractère spécial
- **Hashage**: bcrypt avec salt automatique
- **Vérification**: Mots de passe communs interdits

### Protection des comptes
- **Tentatives échouées**: Max 5 avant verrouillage
- **Durée de verrouillage**: 30 minutes automatique
- **Rate limiting**: 10 requêtes par minute par IP
- **Sessions**: Tracking complet avec IP et User-Agent

## 🧪 Tests et validation

### Tests automatiques
```bash
# Test de l'API d'authentification
curl -X POST http://localhost:9000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@emailfilter.com","password":"admin123"}'

# Test de santé du service
curl http://localhost:9000/auth/health
```

### Tests manuels
1. **Inscription**: Créer un nouveau compte avec validation
2. **Connexion**: Se connecter avec les comptes de test
3. **Protection**: Tenter d'accéder au dashboard sans authentification
4. **Refresh**: Laisser expirer le token et vérifier le renouvellement
5. **Admin**: Tester les fonctions d'administration

## 🔧 Configuration

### Variables d'environnement

```python
# JWT Configuration
JWT_SECRET_KEY = "your-super-secret-jwt-key-change-in-production"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Rate Limiting
MAX_REQUESTS_PER_MINUTE = 10
RATE_LIMIT_WINDOW_SIZE = 60

# Account Security
MAX_FAILED_LOGIN_ATTEMPTS = 5
ACCOUNT_LOCKOUT_DURATION_MINUTES = 30
```

### Personnalisation

Pour modifier la configuration, éditez les fichiers:
- `auth_system/jwt_handler.py` - Configuration JWT et sécurité
- `auth_system/database.py` - Modèles de données
- `auth_system/auth_api.py` - Endpoints et middleware

## 🔄 Intégration avec l'application existante

### Migration depuis l'ancien système
1. Les anciennes pages (`login.html`, `auth.js`) restent fonctionnelles
2. Les nouvelles pages utilisent le préfixe `new_` 
3. Pas de conflit entre les deux systèmes
4. Migration progressive possible

### Utilisation des tokens JWT
```javascript
// Récupérer le token
const token = localStorage.getItem('access_token');

// Utiliser dans les requêtes API
fetch('/api/protected-endpoint', {
    headers: {
        'Authorization': `Bearer ${token}`
    }
});
```

## 📞 Support et dépannage

### Problèmes courants

1. **Erreur de démarrage du serveur**
   ```bash
   # Vérifier les dépendances
   pip install -r auth_system/requirements.txt
   
   # Vérifier le port
   lsof -i :9000
   ```

2. **Base de données corrompue**
   ```bash
   # Supprimer et recréer
   rm auth_system/auth.db
   python auth_system/start_auth_server.py
   ```

3. **Tokens expirés**
   - Les refresh tokens se renouvellent automatiquement
   - En cas de problème, se déconnecter et se reconnecter

### Logs de débogage
- **Serveur**: Console du serveur d'authentification
- **Audit**: Fichier `auth_system/audit.log`
- **Frontend**: Console du navigateur

## 🚀 Déploiement en production

### Modifications requises pour la production

1. **Changer la clé secrète JWT**:
   ```python
   JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'your-production-secret')
   ```

2. **Configurer HTTPS**:
   - Certificats SSL/TLS
   - Headers de sécurité renforcés
   - Cookies sécurisés

3. **Base de données**:
   - PostgreSQL ou MySQL recommandés
   - Sauvegardes automatiques
   - Chiffrement au repos

4. **Monitoring**:
   - Logs centralisés
   - Métriques de performance
   - Alertes de sécurité

## 📊 Métriques et monitoring

Le système génère automatiquement:
- **Logs d'audit** : Toutes les actions de sécurité
- **Métriques d'utilisation** : Connexions, erreurs, performances
- **Indicateurs de sécurité** : Tentatives d'intrusion, comptes verrouillés

---

**🎉 Votre nouveau système d'authentification est maintenant prêt !**

Démarrez avec:
```bash
python auth_system/start_auth_server.py
```

Puis accédez à: http://localhost:8080/new_login.html