// Gestionnaire d'authentification
class AuthenticationManager {
    constructor() {
        this.isLoggedIn = false;
        this.currentUser = null;
        this.sessionTimeout = 30 * 60 * 1000; // 30 minutes
        this.sessionTimer = null;
        this.init();
    }

    // Initialisation
    init() {
        this.checkExistingSession();
        this.setupSessionTimeout();
    }

    // Vérifier une session existante
    checkExistingSession() {
        const session = localStorage.getItem('emailFilter_session');
        const rememberSession = localStorage.getItem('emailFilter_remember');
        
        if (session) {
            try {
                const sessionData = JSON.parse(session);
                if (this.isSessionValid(sessionData)) {
                    this.currentUser = sessionData.user;
                    this.isLoggedIn = true;
                    this.updateSessionExpiry();
                    return true;
                }
            } catch (error) {
                console.error('Erreur lors de la vérification de session:', error);
            }
        }
        
        if (rememberSession) {
            try {
                const rememberData = JSON.parse(rememberSession);
                if (this.isRememberTokenValid(rememberData)) {
                    this.currentUser = rememberData.user;
                    this.isLoggedIn = true;
                    this.createSession(false);
                    return true;
                }
            } catch (error) {
                console.error('Erreur lors de la vérification du token de mémorisation:', error);
            }
        }
        
        return false;
    }

    // Vérifier si la session est valide
    isSessionValid(sessionData) {
        const now = Date.now();
        return sessionData.expires > now && sessionData.user;
    }

    // Vérifier si le token de mémorisation est valide
    isRememberTokenValid(rememberData) {
        const now = Date.now();
        return rememberData.expires > now && rememberData.user;
    }

    // Connexion utilisateur
    async login(email, password, rememberMe = false) {
        try {
            // Simulation d'une authentification (remplacez par votre logique)
            const result = await this.authenticateUser(email, password);
            
            if (result.success) {
                this.currentUser = result.user;
                this.isLoggedIn = true;
                
                // Créer la session
                this.createSession(rememberMe);
                
                // Déclencher l'événement de connexion
                this.dispatchLoginEvent();
                
                return { success: true, user: result.user };
            } else {
                return { success: false, message: result.message };
            }
        } catch (error) {
            console.error('Erreur lors de la connexion:', error);
            return { success: false, message: 'Erreur de connexion' };
        }
    }

    // Authentification utilisateur (simulation)
    async authenticateUser(email, password) {
        // Simulation d'un délai d'authentification
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Utilisateurs de test (remplacez par votre logique d'authentification)
        const testUsers = [
            { email: 'admin@emailfilter.com', password: 'admin123', role: 'admin', name: 'Administrateur' },
            { email: 'user@emailfilter.com', password: 'user123', role: 'user', name: 'Utilisateur' },
            { email: 'demo@emailfilter.com', password: 'demo123', role: 'demo', name: 'Démo' }
        ];
        
        const user = testUsers.find(u => u.email === email && u.password === password);
        
        if (user) {
            return {
                success: true,
                user: {
                    id: user.email,
                    email: user.email,
                    name: user.name,
                    role: user.role,
                    loginTime: new Date().toISOString()
                }
            };
        } else {
            return {
                success: false,
                message: 'Email ou mot de passe incorrect'
            };
        }
    }

    // Créer une session
    createSession(rememberMe = false) {
        const now = Date.now();
        const sessionData = {
            user: this.currentUser,
            created: now,
            expires: now + this.sessionTimeout,
            rememberMe: rememberMe
        };
        
        // Stocker la session
        localStorage.setItem('emailFilter_session', JSON.stringify(sessionData));
        
        // Si "se souvenir de moi" est activé, créer un token de longue durée
        if (rememberMe) {
            const rememberData = {
                user: this.currentUser,
                expires: now + (30 * 24 * 60 * 60 * 1000) // 30 jours
            };
            localStorage.setItem('emailFilter_remember', JSON.stringify(rememberData));
        }
        
        this.setupSessionTimeout();
    }

    // Déconnexion
    logout() {
        this.isLoggedIn = false;
        this.currentUser = null;
        
        // Supprimer les données de session
        localStorage.removeItem('emailFilter_session');
        localStorage.removeItem('emailFilter_remember');
        
        // Annuler le timer de session
        if (this.sessionTimer) {
            clearTimeout(this.sessionTimer);
            this.sessionTimer = null;
        }
        
        // Déclencher l'événement de déconnexion
        this.dispatchLogoutEvent();
        
        // Rediriger vers la page de connexion
        window.location.href = 'login.html';
    }

    // Vérifier si l'utilisateur est authentifié
    isAuthenticated() {
        return this.isLoggedIn && this.currentUser;
    }

    // Obtenir l'utilisateur actuel
    getCurrentUser() {
        return this.currentUser;
    }

    // Mettre à jour l'expiration de la session
    updateSessionExpiry() {
        const session = localStorage.getItem('emailFilter_session');
        if (session) {
            try {
                const sessionData = JSON.parse(session);
                sessionData.expires = Date.now() + this.sessionTimeout;
                localStorage.setItem('emailFilter_session', JSON.stringify(sessionData));
                this.setupSessionTimeout();
            } catch (error) {
                console.error('Erreur lors de la mise à jour de la session:', error);
            }
        }
    }

    // Configurer le timeout de session
    setupSessionTimeout() {
        if (this.sessionTimer) {
            clearTimeout(this.sessionTimer);
        }
        
        this.sessionTimer = setTimeout(() => {
            this.logout();
        }, this.sessionTimeout);
    }

    // Déclencher l'événement de connexion
    dispatchLoginEvent() {
        const event = new CustomEvent('authLogin', {
            detail: { user: this.currentUser }
        });
        document.dispatchEvent(event);
    }

    // Déclencher l'événement de déconnexion
    dispatchLogoutEvent() {
        const event = new CustomEvent('authLogout', {
            detail: { user: this.currentUser }
        });
        document.dispatchEvent(event);
    }

    // Vérifier les permissions utilisateur
    hasPermission(permission) {
        if (!this.isAuthenticated()) {
            return false;
        }
        
        const role = this.currentUser.role;
        const permissions = {
            admin: ['view', 'edit', 'delete', 'manage_users', 'view_logs'],
            user: ['view', 'edit'],
            demo: ['view']
        };
        
        return permissions[role] && permissions[role].includes(permission);
    }

    // Obtenir les informations de session
    getSessionInfo() {
        const session = localStorage.getItem('emailFilter_session');
        if (session) {
            try {
                const sessionData = JSON.parse(session);
                return {
                    user: sessionData.user,
                    created: new Date(sessionData.created).toLocaleString(),
                    expires: new Date(sessionData.expires).toLocaleString(),
                    timeLeft: Math.max(0, sessionData.expires - Date.now())
                };
            } catch (error) {
                console.error('Erreur lors de la récupération des informations de session:', error);
            }
        }
        return null;
    }

    // Rafraîchir la session
    refreshSession() {
        if (this.isAuthenticated()) {
            this.updateSessionExpiry();
        }
    }
}

// Créer une instance globale
const AuthManager = new AuthenticationManager();

// Middleware pour protéger les routes
function requireAuth() {
    if (!AuthManager.isAuthenticated()) {
        window.location.href = 'login.html';
        return false;
    }
    return true;
}

// Middleware pour vérifier les permissions
function requirePermission(permission) {
    if (!AuthManager.hasPermission(permission)) {
        alert('Vous n\'avez pas les permissions nécessaires pour cette action.');
        return false;
    }
    return true;
}

// Fonction utilitaire pour actualiser la session sur les interactions
function refreshSessionOnActivity() {
    // Actualiser la session sur les clics et les mouvements de souris
    let lastActivity = Date.now();
    
    document.addEventListener('click', () => {
        const now = Date.now();
        if (now - lastActivity > 60000) { // 1 minute
            AuthManager.refreshSession();
            lastActivity = now;
        }
    });
    
    document.addEventListener('mousemove', () => {
        const now = Date.now();
        if (now - lastActivity > 60000) { // 1 minute
            AuthManager.refreshSession();
            lastActivity = now;
        }
    });
}

// Initialiser le rafraîchissement automatique de session
document.addEventListener('DOMContentLoaded', refreshSessionOnActivity);

// Exporter pour utilisation dans d'autres fichiers
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AuthManager, requireAuth, requirePermission };
}

// S'assurer que AuthManager est disponible globalement
window.AuthManager = AuthManager;