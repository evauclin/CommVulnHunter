// Protection d'authentification globale
(function() {
    'use strict';
    
    // Configuration
    const CONFIG = {
        loginPage: 'login.html',
        storageKeys: {
            session: 'emailFilter_session',
            remember: 'emailFilter_remember'
        },
        protectedPages: [
            'index.html',
            'test-auth.html'
        ]
    };
    
    // Fonction de vérification d'authentification
    function checkAuthentication() {
        try {
            const session = localStorage.getItem(CONFIG.storageKeys.session);
            const remember = localStorage.getItem(CONFIG.storageKeys.remember);
            
            // Vérifier la session active
            if (session) {
                const sessionData = JSON.parse(session);
                if (sessionData.expires > Date.now() && sessionData.user) {
                    console.log('[AuthGuard] Session valide trouvée pour:', sessionData.user.email);
                    return true;
                }
            }
            
            // Vérifier le token "remember me"
            if (remember) {
                const rememberData = JSON.parse(remember);
                if (rememberData.expires > Date.now() && rememberData.user) {
                    console.log('[AuthGuard] Token remember valide trouvé pour:', rememberData.user.email);
                    return true;
                }
            }
            
            return false;
        } catch (error) {
            console.error('[AuthGuard] Erreur lors de la vérification d\'authentification:', error);
            return false;
        }
    }
    
    // Fonction de redirection
    function redirectToLogin() {
        console.log('[AuthGuard] Redirection vers la page de connexion');
        
        // Nettoyer les données corrompues
        localStorage.removeItem(CONFIG.storageKeys.session);
        localStorage.removeItem(CONFIG.storageKeys.remember);
        
        // Rediriger
        window.location.href = CONFIG.loginPage;
    }
    
    // Fonction pour vérifier si la page courante est protégée
    function isProtectedPage() {
        const currentPage = window.location.pathname.split('/').pop();
        
        // Exclure explicitement la page de login
        if (currentPage === 'login.html') {
            return false;
        }
        
        return CONFIG.protectedPages.includes(currentPage) || currentPage === '';
    }
    
    // Protection principale
    function protectPage() {
        // Vérifier si la page doit être protégée
        if (!isProtectedPage()) {
            return;
        }
        
        console.log('[AuthGuard] Vérification de l\'authentification...');
        
        // Vérifier l'authentification
        if (!checkAuthentication()) {
            console.log('[AuthGuard] Accès non autorisé détecté');
            redirectToLogin();
            return;
        }
        
        console.log('[AuthGuard] Accès autorisé');
    }
    
    // Fonction de protection avec écran de chargement
    function protectPageWithLoader() {
        // Créer un écran de chargement si inexistant
        if (!document.getElementById('authLoading')) {
            const loader = document.createElement('div');
            loader.id = 'authLoading';
            loader.innerHTML = `
                <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            display: flex; align-items: center; justify-content: center; 
                            z-index: 9999; color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                    <div style="text-align: center;">
                        <div style="width: 40px; height: 40px; border: 4px solid rgba(255,255,255,0.3); 
                                    border-radius: 50%; border-top: 4px solid white; 
                                    animation: spin 1s linear infinite; margin: 0 auto 20px;"></div>
                        <h3>Vérification de l'authentification...</h3>
                        <p>Veuillez patienter</p>
                    </div>
                </div>
                <style>
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
            `;
            document.body.appendChild(loader);
        }
        
        // Masquer le contenu principal
        const mainContent = document.getElementById('mainContent');
        if (mainContent) {
            mainContent.style.display = 'none';
        }
        
        // Masquer aussi le body pour éviter le flash de contenu
        document.body.style.visibility = 'hidden';
        
        // Vérifier l'authentification
        setTimeout(() => {
            if (!checkAuthentication()) {
                redirectToLogin();
                return;
            }
            
            // Afficher le contenu si authentifié
            const authLoading = document.getElementById('authLoading');
            if (authLoading) {
                authLoading.style.display = 'none';
            }
            
            if (mainContent) {
                mainContent.style.display = 'block';
            }
            
            // Rendre le body visible
            document.body.style.visibility = 'visible';
            
            console.log('[AuthGuard] Page chargée avec succès');
        }, 500);
    }
    
    // Surveillance des changements d'état d'authentification
    function watchAuthState() {
        // Surveiller les changements dans le localStorage
        window.addEventListener('storage', function(e) {
            if (e.key === CONFIG.storageKeys.session || e.key === CONFIG.storageKeys.remember) {
                console.log('[AuthGuard] Changement d\'état d\'authentification détecté');
                protectPage();
            }
        });
        
        // Surveiller les changements de focus de l'onglet
        document.addEventListener('visibilitychange', function() {
            if (!document.hidden) {
                console.log('[AuthGuard] Retour sur l\'onglet - revérification');
                protectPage();
            }
        });
    }
    
    // Fonction publique pour forcer la vérification
    window.AuthGuard = {
        check: protectPage,
        checkWithLoader: protectPageWithLoader,
        isAuthenticated: checkAuthentication,
        redirectToLogin: redirectToLogin
    };
    
    // Protection automatique au chargement
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', protectPage);
    } else {
        protectPage();
    }
    
    // Initialiser la surveillance
    watchAuthState();
    
    console.log('[AuthGuard] Module de protection chargé');
})();