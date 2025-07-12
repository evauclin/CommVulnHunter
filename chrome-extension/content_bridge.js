// content_bridge.js

// Écoute l'événement envoyé par l'intercepteur injecté
window.addEventListener('Guardian_NotificationIntercepted', (event) => {
    // Relais le message au service worker (background.js)
    chrome.runtime.sendMessage(event.detail);
});