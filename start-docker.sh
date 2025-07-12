#!/bin/bash

# Script de démarrage pour CommVulnHunter avec authentification
echo "🚀 Démarrage de CommVulnHunter avec système d'authentification"
echo "================================================="

# Vérifier que Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé. Veuillez installer Docker d'abord."
    exit 1
fi

# Vérifier que Docker Compose est installé
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé. Veuillez installer Docker Compose d'abord."
    exit 1
fi

# Arrêter les conteneurs existants s'ils existent
echo "🛑 Arrêt des conteneurs existants..."
docker-compose down

# Construire et démarrer les conteneurs
echo "🔨 Construction et démarrage des conteneurs..."
docker-compose up --build -d

# Attendre que les services démarrent
echo "⏳ Attente du démarrage des services..."
sleep 10

# Vérifier le statut des conteneurs
echo "📊 Vérification du statut des conteneurs..."
docker-compose ps

# Afficher les informations d'accès
echo ""
echo "✅ CommVulnHunter est maintenant démarré !"
echo "================================================="
echo "🌐 Application Web (avec authentification) : http://localhost:8080"
echo "🔧 API FastAPI : http://localhost:8000"
echo ""
echo "🔐 Comptes de test disponibles :"
echo "   👤 Administrateur : admin@emailfilter.com / admin123"
echo "   👤 Utilisateur    : user@emailfilter.com / user123"
echo "   👤 Démo          : demo@emailfilter.com / demo123"
echo ""
echo "📚 Pages disponibles :"
echo "   🏠 Accueil (redirige vers login) : http://localhost:8080/"
echo "   🔑 Connexion : http://localhost:8080/login.html"
echo "   📊 Dashboard : http://localhost:8080/index.html"
echo "   🧪 Test Auth : http://localhost:8080/test-auth.html"
echo ""
echo "📝 Pour arrêter les services : docker-compose down"
echo "📝 Pour voir les logs : docker-compose logs -f"
echo "================================================="