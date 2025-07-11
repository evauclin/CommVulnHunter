// interceptor.js - S'exécute dans le contexte de la page pour intercepter les notifications.

(() => {
    if (window.Notification.isHijackedByAIGuardian) return;

    const OriginalNotification = window.Notification;

    const HijackedNotification = function(title, options) {
        console.log(`[Interceptor] Notification interceptée et bloquée: "${title}"`);

        // Envoie un événement personnalisé que le content-loader peut attraper
        window.dispatchEvent(new CustomEvent('AI_Guardian_InterceptedNotification', {
            detail: {
                type: 'ANALYZE_NOTIFICATION',
                data: {
                    title: title,
                    body: options?.body || '',
                }
            }
        }));

        return {}; // Retourne un objet vide pour bloquer la notif originale
    };

    // Copie des propriétés statiques pour la compatibilité
    HijackedNotification.permission = OriginalNotification.permission;
    HijackedNotification.requestPermission = OriginalNotification.requestPermission;
    HijackedNotification.isHijackedByAIGuardian = true;

    window.Notification = HijackedNotification;
})();