// Version simplifiée et robuste du système d'authentification
(function() {
    'use strict';

    // Configuration
    const CONFIG = {
        sessionTimeout: 30 * 60 * 1000, // 30 minutes
        rememberDuration: 30 * 24 * 60 * 60 * 1000, // 30 jours
        storageKeys: {
            session: 'emailFilter_session',
            remember: 'emailFilter_remember'
        }
    };

    // Comptes de test et utilisateurs enregistrés
    const TEST_USERS = [
        { email: 'admin@emailfilter.com', password: 'admin123', role: 'admin', name: 'Administrateur' },
        { email: 'user@emailfilter.com', password: 'user123', role: 'user', name: 'Utilisateur' },
        { email: 'demo@emailfilter.com', password: 'demo123', role: 'demo', name: 'Démo' }
    ];

    // Clé pour stocker les utilisateurs enregistrés
    const REGISTERED_USERS_KEY = 'emailFilter_registeredUsers';

    // Permissions par rôle
    const PERMISSIONS = {
        admin: ['view', 'edit', 'delete', 'manage_users', 'view_logs'],
        user: ['view', 'edit'],
        demo: ['view']
    };

    // État global
    let currentUser = null;
    let isLoggedIn = false;
    let sessionTimer = null;

    // Fonctions utilitaires
    function log(message) {
        console.log(`[AuthManager] ${message}`);
    }

    function error(message) {
        console.error(`[AuthManager] ${message}`);
    }

    // Gestion du stockage
    function saveToStorage(key, data) {
        try {
            localStorage.setItem(key, JSON.stringify(data));
        } catch (e) {
            error('Erreur lors de la sauvegarde: ' + e.message);
        }
    }

    function loadFromStorage(key) {
        try {
            const data = localStorage.getItem(key);
            return data ? JSON.parse(data) : null;
        } catch (e) {
            error('Erreur lors du chargement: ' + e.message);
            return null;
        }
    }

    function removeFromStorage(key) {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            error('Erreur lors de la suppression: ' + e.message);
        }
    }

    // Gestion des sessions
    function isSessionValid(sessionData) {
        const now = Date.now();
        return sessionData && sessionData.expires > now && sessionData.user;
    }

    function createSession(user, rememberMe = false) {
        const now = Date.now();
        const sessionData = {
            user: user,
            created: now,
            expires: now + CONFIG.sessionTimeout,
            rememberMe: rememberMe
        };
        
        saveToStorage(CONFIG.storageKeys.session, sessionData);
        
        if (rememberMe) {
            const rememberData = {
                user: user,
                expires: now + CONFIG.rememberDuration
            };
            saveToStorage(CONFIG.storageKeys.remember, rememberData);
        }
        
        setupSessionTimeout();
        log('Session créée pour ' + user.email);
    }

    function updateSessionExpiry() {
        const session = loadFromStorage(CONFIG.storageKeys.session);
        if (session) {
            session.expires = Date.now() + CONFIG.sessionTimeout;
            saveToStorage(CONFIG.storageKeys.session, session);
            setupSessionTimeout();
        }
    }

    function setupSessionTimeout() {
        if (sessionTimer) {
            clearTimeout(sessionTimer);
        }
        
        sessionTimer = setTimeout(() => {
            log('Session expirée');
            logout();
        }, CONFIG.sessionTimeout);
    }

    function checkExistingSession() {
        const session = loadFromStorage(CONFIG.storageKeys.session);
        if (session && isSessionValid(session)) {
            currentUser = session.user;
            isLoggedIn = true;
            updateSessionExpiry();
            log('Session existante trouvée pour ' + currentUser.email);
            return true;
        }
        
        const remember = loadFromStorage(CONFIG.storageKeys.remember);
        if (remember && remember.expires > Date.now()) {
            currentUser = remember.user;
            isLoggedIn = true;
            createSession(currentUser, false);
            log('Session restaurée depuis remember token pour ' + currentUser.email);
            return true;
        }
        
        return false;
    }

    // Gestion des utilisateurs enregistrés
    function getRegisteredUsers() {
        const users = loadFromStorage(REGISTERED_USERS_KEY);
        return users || [];
    }

    function saveRegisteredUsers(users) {
        saveToStorage(REGISTERED_USERS_KEY, users);
    }

    function getAllUsers() {
        const registeredUsers = getRegisteredUsers();
        return [...TEST_USERS, ...registeredUsers];
    }

    function userExists(email) {
        const allUsers = getAllUsers();
        return allUsers.some(user => user.email.toLowerCase() === email.toLowerCase());
    }

    function addUser(userData) {
        const registeredUsers = getRegisteredUsers();
        const newUser = {
            id: userData.email,
            email: userData.email,
            name: userData.name,
            password: userData.password,
            role: userData.role || 'user',
            createdAt: new Date().toISOString(),
            newsletter: userData.newsletter || false
        };
        
        registeredUsers.push(newUser);
        saveRegisteredUsers(registeredUsers);
        log('Nouvel utilisateur enregistré: ' + userData.email);
        return newUser;
    }

    // Authentification
    function authenticateUser(email, password) {
        return new Promise((resolve) => {
            // Simuler un délai d'authentification
            setTimeout(() => {
                const allUsers = getAllUsers();
                const user = allUsers.find(u => u.email === email && u.password === password);
                
                if (user) {
                    resolve({
                        success: true,
                        user: {
                            id: user.email,
                            email: user.email,
                            name: user.name,
                            role: user.role,
                            loginTime: new Date().toISOString()
                        }
                    });
                } else {
                    resolve({
                        success: false,
                        message: 'Email ou mot de passe incorrect'
                    });
                }
            }, 1000);
        });
    }

    // Inscription d'un nouvel utilisateur
    function registerUser(userData) {
        return new Promise((resolve) => {
            // Simuler un délai d'inscription
            setTimeout(() => {
                // Vérifier si l'utilisateur existe déjà
                if (userExists(userData.email)) {
                    resolve({
                        success: false,
                        message: 'Un compte avec cet email existe déjà'
                    });
                    return;
                }

                // Validation des données
                if (!userData.name || userData.name.trim().length < 2) {
                    resolve({
                        success: false,
                        message: 'Le nom doit contenir au moins 2 caractères'
                    });
                    return;
                }

                if (!userData.email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(userData.email)) {
                    resolve({
                        success: false,
                        message: 'Veuillez fournir un email valide'
                    });
                    return;
                }

                if (!userData.password || userData.password.length < 6) {
                    resolve({
                        success: false,
                        message: 'Le mot de passe doit contenir au moins 6 caractères'
                    });
                    return;
                }

                // Créer l'utilisateur
                try {
                    const newUser = addUser(userData);
                    resolve({
                        success: true,
                        user: {
                            id: newUser.email,
                            email: newUser.email,
                            name: newUser.name,
                            role: newUser.role
                        },
                        message: 'Compte créé avec succès'
                    });
                } catch (error) {
                    resolve({
                        success: false,
                        message: 'Erreur lors de la création du compte'
                    });
                }
            }, 1500);
        });
    }

    // API publique
    window.AuthManager = {
        // Connexion
        login: async function(email, password, rememberMe = false) {
            try {
                log('Tentative de connexion pour ' + email);
                const result = await authenticateUser(email, password);
                
                if (result.success) {
                    currentUser = result.user;
                    isLoggedIn = true;
                    createSession(currentUser, rememberMe);
                    
                    // Créer un cookie d'authentification pour nginx
                    const cookieValue = btoa(JSON.stringify({
                        user: currentUser.email,
                        expires: Date.now() + CONFIG.sessionTimeout
                    }));
                    
                    const cookieExpire = new Date(Date.now() + CONFIG.sessionTimeout);
                    document.cookie = `emailFilter_auth=${cookieValue}; expires=${cookieExpire.toUTCString()}; path=/; SameSite=Strict`;
                    
                    // Déclencher l'événement de connexion
                    document.dispatchEvent(new CustomEvent('authLogin', {
                        detail: { user: currentUser }
                    }));
                    
                    log('Connexion réussie pour ' + currentUser.email);
                    return { success: true, user: currentUser };
                } else {
                    log('Connexion échouée pour ' + email);
                    return { success: false, message: result.message };
                }
            } catch (error) {
                error('Erreur lors de la connexion: ' + error.message);
                return { success: false, message: 'Erreur de connexion' };
            }
        },

        // Inscription
        register: async function(userData) {
            try {
                log('Tentative d\'inscription pour ' + userData.email);
                const result = await registerUser(userData);
                
                if (result.success) {
                    log('Inscription réussie pour ' + userData.email);
                    return { success: true, user: result.user, message: result.message };
                } else {
                    log('Inscription échouée pour ' + userData.email + ': ' + result.message);
                    return { success: false, message: result.message };
                }
            } catch (error) {
                error('Erreur lors de l\'inscription: ' + error.message);
                return { success: false, message: 'Erreur lors de l\'inscription' };
            }
        },

        // Déconnexion
        logout: function() {
            log('Déconnexion de ' + (currentUser ? currentUser.email : 'utilisateur inconnu'));
            
            isLoggedIn = false;
            currentUser = null;
            
            removeFromStorage(CONFIG.storageKeys.session);
            removeFromStorage(CONFIG.storageKeys.remember);
            
            // Supprimer le cookie d'authentification
            document.cookie = 'emailFilter_auth=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; SameSite=Strict';
            
            if (sessionTimer) {
                clearTimeout(sessionTimer);
                sessionTimer = null;
            }
            
            // Déclencher l'événement de déconnexion
            document.dispatchEvent(new CustomEvent('authLogout', {
                detail: { user: currentUser }
            }));
            
            // Rediriger vers la page de connexion
            window.location.href = 'login.html';
        },

        // Vérifications
        isAuthenticated: function() {
            return isLoggedIn && currentUser;
        },

        getCurrentUser: function() {
            return currentUser;
        },

        hasPermission: function(permission) {
            if (!this.isAuthenticated()) {
                return false;
            }
            
            const role = currentUser.role;
            return PERMISSIONS[role] && PERMISSIONS[role].includes(permission);
        },

        // Gestion des sessions
        getSessionInfo: function() {
            const session = loadFromStorage(CONFIG.storageKeys.session);
            if (session) {
                return {
                    user: session.user,
                    created: new Date(session.created).toLocaleString(),
                    expires: new Date(session.expires).toLocaleString(),
                    timeLeft: Math.max(0, session.expires - Date.now())
                };
            }
            return null;
        },

        refreshSession: function() {
            if (this.isAuthenticated()) {
                updateSessionExpiry();
                log('Session rafraîchie pour ' + currentUser.email);
            }
        },

        // Initialisation
        init: function() {
            log('Initialisation du système d\'authentification');
            
            if (checkExistingSession()) {
                // Créer le cookie d'authentification si une session existe
                const cookieValue = btoa(JSON.stringify({
                    user: currentUser.email,
                    expires: Date.now() + CONFIG.sessionTimeout
                }));
                
                const cookieExpire = new Date(Date.now() + CONFIG.sessionTimeout);
                document.cookie = `emailFilter_auth=${cookieValue}; expires=${cookieExpire.toUTCString()}; path=/; SameSite=Strict`;
            }
            
            // Rafraîchir la session sur l'activité
            let lastActivity = Date.now();
            
            document.addEventListener('click', () => {
                const now = Date.now();
                if (now - lastActivity > 60000) { // 1 minute
                    this.refreshSession();
                    lastActivity = now;
                }
            });
            
            document.addEventListener('mousemove', () => {
                const now = Date.now();
                if (now - lastActivity > 60000) { // 1 minute
                    this.refreshSession();
                    lastActivity = now;
                }
            });
        }
    };

    // Fonctions utilitaires globales
    window.requireAuth = function() {
        if (!window.AuthManager.isAuthenticated()) {
            window.location.href = 'login.html';
            return false;
        }
        return true;
    };

    window.requirePermission = function(permission) {
        if (!window.AuthManager.hasPermission(permission)) {
            alert('Vous n\'avez pas les permissions nécessaires pour cette action.');
            return false;
        }
        return true;
    };

    // Initialisation automatique seulement si pas sur la page de connexion
    document.addEventListener('DOMContentLoaded', function() {
        const currentPage = window.location.pathname.split('/').pop();
        if (currentPage !== 'login.html' && currentPage !== 'register.html') {
            window.AuthManager.init();
            log('Système d\'authentification initialisé');
        } else {
            log('Page de connexion/inscription détectée - nettoyage des données corrompues');
            // Sur la page de login, nettoyer les données potentiellement corrompues
            const session = loadFromStorage(CONFIG.storageKeys.session);
            const remember = loadFromStorage(CONFIG.storageKeys.remember);
            
            // Si des données existent mais sont invalides, les supprimer
            if (session && !isSessionValid(session)) {
                removeFromStorage(CONFIG.storageKeys.session);
                log('Session expirée supprimée');
            }
            if (remember && remember.expires <= Date.now()) {
                removeFromStorage(CONFIG.storageKeys.remember);
                log('Token remember expiré supprimé');
            }
            
            // Supprimer le cookie d'authentification expiré
            document.cookie = 'emailFilter_auth=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; SameSite=Strict';
        }
    });

    // Alias pour compatibilité
    window.logout = window.AuthManager.logout;

    log('Module d\'authentification chargé');
})();