// interceptor.js - Capture complète des données de notification
(() => {
    console.log('[INTERCEPTOR] 🚀 Démarrage de l\'interceptor complet...');

    if (window.Notification.isHijackedByGuardian) {
        console.log('[INTERCEPTOR] ⚠️ Déjà hijacké, abandon');
        return;
    }

    const OriginalNotification = window.Notification;
    console.log('[INTERCEPTOR] 📦 Notification originale sauvegardée');

    const HijackedNotification = function(title, options = {}) {
        console.log('[INTERCEPTOR] 🎯 Notification interceptée!');
        console.log('[INTERCEPTOR] 📧 Titre:', title);
        console.log('[INTERCEPTOR] ⚙️ Options:', options);

        // Extraire toutes les données possibles
        const notificationData = {
            title: title || 'Sans titre',
            body: options.body || '',
            icon: options.icon || null,
            image: options.image || null,
            badge: options.badge || null,
            tag: options.tag || null,
            data: options.data || null,
            timestamp: Date.now(),
            dir: options.dir || 'auto',
            lang: options.lang || navigator.language || 'fr',
            vibrate: options.vibrate || null,
            silent: options.silent || false,
            requireInteraction: options.requireInteraction || false,
            sticky: options.sticky || false,
            renotify: options.renotify || false,
            actions: options.actions || [],

            // Métadonnées supplémentaires
            url: window.location.href,
            origin: window.location.origin,
            userAgent: navigator.userAgent,
            timestamp_iso: new Date().toISOString(),

            // Informations sur la page
            pageTitle: document.title,
            pageDescription: getPageDescription(),
            service: detectService()
        };

        console.log('[INTERCEPTOR] 📊 Données complètes extraites:', notificationData);

        // Envoyer via toutes les méthodes disponibles
        sendNotificationData(notificationData);

        // Créer un objet notification factice mais fonctionnel
        const fakeNotification = createFakeNotification(title, options);

        console.log('[INTERCEPTOR] ✅ Notification traitée et transmise');
        return fakeNotification;
    };

    // === FONCTIONS UTILITAIRES ===
    function detectService() {
        const hostname = window.location.hostname;
        if (hostname.includes('google.com')) return 'Gmail';
        if (hostname.includes('yahoo.com')) return 'Yahoo Mail';
        return 'Unknown';
    }

    function getPageDescription() {
        const metaDesc = document.querySelector('meta[name="description"]');
        return metaDesc ? metaDesc.content : '';
    }

    function sendNotificationData(data) {
        console.log('[INTERCEPTOR] 📤 Envoi des données via multiple canaux...');

        // Méthode 1 : Événement window personnalisé
        try {
            window.dispatchEvent(new CustomEvent('Guardian_Intercepted', {
                detail: data,
                bubbles: true,
                cancelable: true
            }));
            console.log('[INTERCEPTOR] ✅ Envoyé via window event');
        } catch (e) {
            console.error('[INTERCEPTOR] ❌ Erreur window event:', e);
        }

        // Méthode 2 : Événement document
        try {
            document.dispatchEvent(new CustomEvent('Guardian_DOM_Event', {
                detail: data,
                bubbles: true,
                cancelable: true
            }));
            console.log('[INTERCEPTOR] ✅ Envoyé via document event');
        } catch (e) {
            console.error('[INTERCEPTOR] ❌ Erreur document event:', e);
        }

        // Méthode 3 : window.postMessage
        try {
            window.postMessage({
                type: 'GUARDIAN_NOTIFICATION',
                data: data,
                timestamp: Date.now()
            }, '*');
            console.log('[INTERCEPTOR] ✅ Envoyé via postMessage');
        } catch (e) {
            console.error('[INTERCEPTOR] ❌ Erreur postMessage:', e);
        }

        // Méthode 4 : Custom property sur window
        try {
            if (!window.GuardianNotifications) {
                window.GuardianNotifications = [];
            }
            window.GuardianNotifications.push(data);

            // Garder seulement les 10 dernières
            if (window.GuardianNotifications.length > 10) {
                window.GuardianNotifications = window.GuardianNotifications.slice(-10);
            }
            console.log('[INTERCEPTOR] ✅ Sauvegardé dans window.GuardianNotifications');
        } catch (e) {
            console.error('[INTERCEPTOR] ❌ Erreur window property:', e);
        }

        // Méthode 5 : localStorage de secours (si disponible)
        try {
            if (typeof localStorage !== 'undefined') {
                const key = 'guardian_last_notification';
                localStorage.setItem(key, JSON.stringify(data));
                console.log('[INTERCEPTOR] ✅ Sauvegardé dans localStorage');
            }
        } catch (e) {
            console.log('[INTERCEPTOR] ℹ️ localStorage non disponible ou restreint');
        }
    }

    function createFakeNotification(title, options) {
        // Créer un objet qui simule une vraie notification
        const fakeNotif = {
            title: title,
            body: options.body || '',
            icon: options.icon || '',
            tag: options.tag || '',
            data: options.data || null,

            // Méthodes simulées
            close: function() {
                console.log('[INTERCEPTOR] 🔒 Fake notification fermée');
                if (this.onclose) this.onclose();
            },

            addEventListener: function(type, listener, options) {
                console.log(`[INTERCEPTOR] 👂 EventListener ajouté: ${type}`);
                // Simuler l'ajout d'event listener
                if (!this._listeners) this._listeners = {};
                if (!this._listeners[type]) this._listeners[type] = [];
                this._listeners[type].push(listener);
            },

            removeEventListener: function(type, listener) {
                console.log(`[INTERCEPTOR] 🗑️ EventListener retiré: ${type}`);
                if (this._listeners && this._listeners[type]) {
                    const index = this._listeners[type].indexOf(listener);
                    if (index > -1) {
                        this._listeners[type].splice(index, 1);
                    }
                }
            },

            dispatchEvent: function(event) {
                console.log(`[INTERCEPTOR] 📢 Event dispatché:`, event.type);
                // Simuler la dispatch d'event
                return true;
            },

            // Propriétés d'événement
            onclick: null,
            onshow: null,
            onclose: null,
            onerror: null
        };

        // Simuler l'événement 'show' après un délai
        setTimeout(() => {
            if (fakeNotif.onshow) {
                console.log('[INTERCEPTOR] 📢 Événement show simulé');
                fakeNotif.onshow();
            }
        }, 100);

        return fakeNotif;
    }

    // === COPIE DES PROPRIÉTÉS STATIQUES ===
    HijackedNotification.permission = OriginalNotification.permission;
    HijackedNotification.requestPermission = function(callback) {
        console.log('[INTERCEPTOR] 🔐 requestPermission appelé');
        return OriginalNotification.requestPermission.call(this, callback);
    };

    // Propriétés personnalisées pour le debugging
    HijackedNotification.isHijackedByGuardian = true;
    HijackedNotification.originalNotification = OriginalNotification;
    HijackedNotification.guardianVersion = '2.0';

    // === REMPLACEMENT DE window.Notification ===
    try {
        Object.defineProperty(window, 'Notification', {
            value: HijackedNotification,
            writable: false,
            configurable: false
        });
        console.log('[INTERCEPTOR] 🔄 window.Notification remplacé (non-configurable)');
    } catch (e) {
        // Fallback si defineProperty échoue
        window.Notification = HijackedNotification;
        console.log('[INTERCEPTOR] 🔄 window.Notification remplacé (fallback)');
    }

    // === TESTS ET VALIDATION ===
    function runDiagnostics() {
        console.log('[INTERCEPTOR] 🔍 Exécution des diagnostics...');

        // Test 1: Vérifier le remplacement
        console.log('[INTERCEPTOR] Test 1 - Hijack status:', window.Notification.isHijackedByGuardian);

        // Test 2: Vérifier les permissions
        console.log('[INTERCEPTOR] Test 2 - Permission:', window.Notification.permission);

        // Test 3: Test fonctionnel après un délai
        setTimeout(() => {
            console.log('[INTERCEPTOR] 🧪 Test fonctionnel...');
            try {
                const testNotif = new window.Notification('🧪 Test Guardian', {
                    body: 'Test de l\'interceptor complet',
                    tag: 'guardian-test',
                    icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><circle cx="8" cy="8" r="8" fill="blue"/></svg>'
                });
                console.log('[INTERCEPTOR] ✅ Test réussi, notification créée:', testNotif);
            } catch (e) {
                console.error('[INTERCEPTOR] ❌ Test échoué:', e);
            }
        }, 2000);

        // Test 4: Vérifier la disponibilité des canaux de communication
        console.log('[INTERCEPTOR] Test 4 - Canaux de communication:');
        console.log('  - window events:', typeof window.dispatchEvent === 'function');
        console.log('  - document events:', typeof document.dispatchEvent === 'function');
        console.log('  - postMessage:', typeof window.postMessage === 'function');
        console.log('  - localStorage:', typeof localStorage !== 'undefined');
    }

    // Exécuter les diagnostics après initialisation
    setTimeout(runDiagnostics, 1000);

    console.log('[INTERCEPTOR] 🎉 Interceptor complet installé avec succès!');
    console.log('[INTERCEPTOR] 📋 Fonctionnalités activées:');
    console.log('  ✅ Capture complète des données de notification');
    console.log('  ✅ Multiple canaux de communication');
    console.log('  ✅ Simulation de notification fonctionnelle');
    console.log('  ✅ Diagnostics et tests automatiques');
    console.log('  ✅ Gestion des erreurs robuste');
})();