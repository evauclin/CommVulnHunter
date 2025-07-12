#!/bin/bash

echo "🔐 CommVulnHunter - Démarrage avec Authentification Sécurisée"
echo "============================================================="

# Vérifier que Docker est installé
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

echo "✅ Docker et Docker Compose détectés"
echo ""

# Arrêter les conteneurs existants s'ils existent
echo "🔄 Arrêt des conteneurs existants..."
docker-compose down 2>/dev/null

echo "🏗️ Construction des images Docker..."
docker-compose build

echo "🚀 Démarrage des services..."
docker-compose up -d

echo ""
echo "⏳ Attente du démarrage des services..."
sleep 10

# Vérifier que les services sont en marche
echo ""
echo "🔍 Vérification des services..."

# Vérifier le service web
if curl -s http://localhost:8080 > /dev/null; then
    echo "✅ Service Web (8080) : OPÉRATIONNEL"
else
    echo "❌ Service Web (8080) : ERREUR"
fi

# Vérifier le service ML
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Service ML (8000) : OPÉRATIONNEL" 
else
    echo "❌ Service ML (8000) : ERREUR"
fi

# Vérifier le service d'authentification
if curl -s http://localhost:9000/auth/health > /dev/null; then
    echo "✅ Service Auth (9000) : OPÉRATIONNEL"
else
    echo "❌ Service Auth (9000) : ERREUR"
fi

echo ""
echo "🎉 Démarrage terminé !"
echo ""
echo "📍 URLs d'accès:"
echo "   🏠 Application principale (redirige vers login) : http://localhost:8080"
echo "   🔐 Page de connexion                          : http://localhost:8080/pages/new_login.html"
echo "   ✍️ Page d'inscription                         : http://localhost:8080/pages/new_register.html"
echo "   📊 Dashboard sécurisé (après authentification) : http://localhost:8080/pages/new_dashboard.html"
echo "   🎯 App ML principale (protégée)               : http://localhost:8080/index.html"
echo "   🔧 API ML (protégée)                          : http://localhost:8000"
echo "   🔑 API Authentification                       : http://localhost:9000"
echo "   📚 Documentation API Auth                     : http://localhost:9000/docs"
echo ""
echo "👥 Comptes de test:"
echo "   Admin : admin@emailfilter.com / admin123"
echo "   User  : user@emailfilter.com / user123" 
echo "   Demo  : demo@emailfilter.com / demo123"
echo ""
echo "🔒 FLUX D'AUTHENTIFICATION:"
echo "   1. Accédez à http://localhost:8080 (vous serez redirigé vers login)"
echo "   2. Connectez-vous avec un des comptes de test"
echo "   3. Accédez au dashboard sécurisé ou à l'application ML"
echo "   4. L'API ML est maintenant protégée par authentification"
echo ""
echo "🛑 Pour arrêter : docker-compose down"
echo "📋 Voir les logs : docker-compose logs -f"