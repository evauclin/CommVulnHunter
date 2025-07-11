// content.js - Injecte le code d'interception directement dans la page.

// Le code de notre intercepteur, sous forme de chaîne de caractères.
const interceptorCode = `
    (() => {
        // Pour éviter d'injecter plusieurs fois si le content script se recharge.
        if (window.Notification.isHijackedByGuardian) {
            return;
        }

        console.log("🚀 [Interceptor] Tentative d'activation de l'intercepteur...");

        const OriginalNotification = window.Notification;

        const HijackedNotification = function(title, options) {
            console.log('🎉🎉🎉 NOTIFICATION INTERCEPTÉE ET BLOQUÉE 🎉🎉🎉');
            console.log('Titre intercepté:', title);
            
            // On envoie un événement personnalisé pour que le content script puisse le récupérer.
            window.dispatchEvent(new CustomEvent('GuardianEvent_Intercepted', {
                detail: {
                    type: 'ANALYZE_NOTIFICATION',
                    data: {
                        title: title,
                        body: options?.body || ''
                    }
                }
            }));

            // On retourne un objet vide pour bloquer la notification originale.
            return {};
        };
        
        // On recopie les propriétés statiques pour ne pas casser les sites.
        HijackedNotification.permission = OriginalNotification.permission;
        HijackedNotification.requestPermission = OriginalNotification.requestPermission;
        HijackedNotification.isHijackedByGuardian = true;

        // On remplace la fonction Notification du navigateur par la nôtre.
        window.Notification = HijackedNotification;

        console.log('✅ [Interceptor] Intercepteur est maintenant actif.');
    })();
`;

// On injecte ce code directement dans une balise script.
try {
    const scriptElement = document.createElement('script');
    scriptElement.textContent = interceptorCode;
    // On l'ajoute à la balise <head> ou au <html> s'il n'y a pas de head.
    (document.head || document.documentElement).appendChild(scriptElement);
    // On retire la balise du DOM après son exécution pour garder le code propre.
    scriptElement.remove();
    console.log('[Content Script] L\'injecteur direct a terminé son travail.');
} catch (e) {
    console.error('[Content Script] Erreur d\'injection directe:', e);
}


// Ce listener reste dans le content script pour relayer le message au background.
window.addEventListener('GuardianEvent_Intercepted', (event) => {
    console.log('[Content Script] Événement intercepté reçu, relais vers le background...');
    try {
        chrome.runtime.sendMessage(event.detail);
    } catch(e) {
        console.error("Erreur lors de l'envoi du message au background:", e);
    }
});