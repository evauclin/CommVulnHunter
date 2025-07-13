// interceptor.js - Capture les notifications natives de la page.
(() => {
    'use strict';
    // Évite les injections multiples
    if (window.Notification.isGuardianHijacked) {
        return;
    }

    console.log('[GUARDIAN INTERCEPTOR] 🚀 Injection de l\'intercepteur de notifications.');

    const OriginalNotification = window.Notification;

    // Remplacement de la fonction Notification
    const HijackedNotification = function(title, options) {
        console.log(`[GUARDIAN INTERCEPTOR] 🎯 Notification interceptée: "${title}"`);

        // Extraire les données du mail. Souvent, Gmail met l'expéditeur dans le titre et le sujet dans le corps.
        // Exemple: Title="john.doe@example.com", Body="Sujet: Votre facture est prête"
        const mailData = {
            sender: title, // Le titre est souvent l'expéditeur
            subject: options.body || '', // Le corps est souvent le sujet
            preview: options.body || '', // On utilise le corps comme aperçu
            source: 'Gmail - Notification Intercept'
        };

        console.log('[GUARDIAN INTERCEPTOR] 📨 Données extraites pour analyse:', mailData);

        // Envoyer les données au background script via un événement personnalisé
        // C'est le pont entre le "monde de la page" et le "monde de l'extension"
        window.dispatchEvent(new CustomEvent('Guardian_NotificationIntercepted', {
            detail: {
                type: 'NEW_MAIL_DETECTED',
                data: mailData
            }
        }));

        // Important : On ne crée PAS la notification originale.
        // On retourne un objet factice pour que le code de Gmail ne plante pas.
        return {
            close: () => {},
            onclick: null,
            onerror: null,
            onshow: null
        };
    };

    // Copier les propriétés statiques de l'original (ex: Notification.permission)
    HijackedNotification.permission = OriginalNotification.permission;
    HijackedNotification.requestPermission = OriginalNotification.requestPermission.bind(OriginalNotification);
    HijackedNotification.isGuardianHijacked = true; // Marqueur pour éviter la double injection

    // Remplacer définitivement window.Notification
    window.Notification = HijackedNotification;

    console.log('[GUARDIAN INTERCEPTOR] ✅ Hijacking de window.Notification terminé.');
})();