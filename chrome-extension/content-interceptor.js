// content-interceptor.js

console.log('🛡️ [Interceptor] Script d\'interception des notifications injecté.');

// Sauvegarder l'implémentation originale de Notification
const OriginalNotification = window.Notification;

// Redéfinir la classe Notification
window.Notification = function(title, options) {
    console.log(`[Interceptor] Notification interceptée ! Titre: "${title}"`);
    console.log('[Interceptor] Options:', options);

    // Empêcher la notification originale de s'afficher
    // En ne faisant rien ici, on la bloque.

    // Envoyer les données au background script pour une analyse et une recréation.
    try {
        chrome.runtime.sendMessage({
            type: 'INTERCEPTED_NOTIFICATION',
            data: {
                title: title,
                body: options?.body || '',
                icon: options?.icon || '',
                origin: window.location.origin
            }
        });
        console.log('[Interceptor] Données envoyées au background script.');
    } catch(e) {
        console.error("[Interceptor] Impossible de communiquer avec le background script. L'extension a peut-être été mise à jour.", e);
    }

    // Pour que le code de la page ne plante pas, on doit retourner un objet qui ressemble à une notification.
    // On ne peut pas retourner l'instance originale, car cela afficherait la notification.
    // On retourne un "dummy object".
    return {
        // Simuler les propriétés et méthodes communes pour éviter les erreurs sur la page.
        title: title,
        body: options?.body || '',
        icon: options?.icon || '',
        close: () => {},
        onclick: null,
        onerror: null
    };
};

// On s'assure que notre fausse notification a les mêmes propriétés statiques que la vraie
window.Notification.permission = OriginalNotification.permission;
window.Notification.requestPermission = OriginalNotification.requestPermission;