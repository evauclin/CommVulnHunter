// content.js - Pont entre le détecteur de mails et le background
console.log('[GUARDIAN CONTENT] 🌉 Content script démarré');

// === 1. ÉCOUTE DES MAILS DÉTECTÉS ===
// Le mail-detector.js (MAIN world) envoie des événements que nous capturons ici (ISOLATED world)

window.addEventListener('message', (event) => {
    // Vérifier que le message vient de notre détecteur
    if (event.source !== window || !event.data) return;

    if (event.data.type === 'GUARDIAN_NEW_MAIL') {
        console.log('[GUARDIAN CONTENT] 📧 Nouveau mail reçu:', event.data.mailData);

        // Transférer vers le background pour analyse
        chrome.runtime.sendMessage({
            type: 'NEW_MAIL_DETECTED',
            data: event.data.mailData
        }).then(response => {
            console.log('[GUARDIAN CONTENT] ✅ Mail envoyé au background:', response);
        }).catch(error => {
            console.error('[GUARDIAN CONTENT] ❌ Erreur envoi background:', error);
        });
    }

    else if (event.data.type === 'GUARDIAN_STATUS') {
        console.log('[GUARDIAN CONTENT] 📊 Statut du détecteur:', event.data.status);
    }
});

// === 2. TEST DE CONNECTIVITÉ ===
// Vérifier que la communication avec le background fonctionne
setTimeout(() => {
    console.log('[GUARDIAN CONTENT] 🧪 Test de connectivité...');

    chrome.runtime.sendMessage({
        type: 'PING',
        data: {
            url: window.location.href,
            timestamp: Date.now()
        }
    }).then(response => {
        console.log('[GUARDIAN CONTENT] 🏓 Pong reçu:', response);
    }).catch(error => {
        console.error('[GUARDIAN CONTENT] ❌ Pas de connexion background:', error);
    });
}, 2000);

// === 3. INJECTION DU DÉTECTEUR ===
// S'assurer que le détecteur est bien injecté
function ensureDetectorInjected() {
    // Envoyer un message au détecteur pour vérifier s'il est actif
    window.postMessage({
        type: 'GUARDIAN_PING',
        from: 'content-script'
    }, '*');

    // Si pas de réponse dans 3 secondes, alerter
    setTimeout(() => {
        window.postMessage({
            type: 'GUARDIAN_STATUS_REQUEST'
        }, '*');
    }, 3000);
}

// Démarrer la vérification après chargement de la page
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensureDetectorInjected);
} else {
    ensureDetectorInjected();
}

// === 4. COMMUNICATION BIDIRECTIONNELLE ===
// Permet au background de communiquer avec le détecteur via le content script

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('[GUARDIAN CONTENT] 📨 Message du background:', message.type);

    if (message.type === 'DETECTOR_COMMAND') {
        // Transférer une commande au détecteur
        window.postMessage({
            type: 'GUARDIAN_COMMAND',
            command: message.command,
            data: message.data
        }, '*');

        sendResponse({ success: true });
    }

    else if (message.type === 'GET_PAGE_INFO') {
        // Envoyer des infos sur la page actuelle
        sendResponse({
            url: window.location.href,
            title: document.title,
            isGmail: window.location.hostname.includes('google.com'),
            isYahoo: window.location.hostname.includes('yahoo.com')
        });
    }

    return true;
});

// === 5. MONITORING DE LA PAGE ===
// Observer les changements importants de la page

let lastUrl = window.location.href;
let lastTitle = document.title;

// Observer les changements d'URL (navigation SPA)
const urlObserver = new MutationObserver(() => {
    if (window.location.href !== lastUrl) {
        console.log('[GUARDIAN CONTENT] 🔄 Navigation détectée:', window.location.href);
        lastUrl = window.location.href;

        // Informer le détecteur du changement de page
        window.postMessage({
            type: 'GUARDIAN_PAGE_CHANGED',
            url: window.location.href
        }, '*');
    }
});

// Observer les changements de titre (souvent liés aux nouveaux mails)
const titleObserver = new MutationObserver(() => {
    if (document.title !== lastTitle) {
        console.log('[GUARDIAN CONTENT] 📰 Titre changé:', document.title);

        // Informer le détecteur du changement de titre
        window.postMessage({
            type: 'GUARDIAN_TITLE_CHANGED',
            title: document.title,
            oldTitle: lastTitle
        }, '*');

        lastTitle = document.title;
    }
});

// Démarrer les observateurs
urlObserver.observe(document.body, { childList: true, subtree: true });
if (document.querySelector('title')) {
    titleObserver.observe(document.querySelector('title'), {
        childList: true,
        characterData: true
    });
}

console.log('[GUARDIAN CONTENT] ✅ Content script initialisé et prêt');