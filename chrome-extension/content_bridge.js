// content_bridge.js - Pont entre l'intercepteur et le service worker

// Écoute l'événement envoyé par l'intercepteur injecté dans le monde MAIN
window.addEventListener('Guardian_NotificationIntercepted', (event) => {
    console.log('[GUARDIAN BRIDGE] 🌉 Événement intercepté, relais vers le background...');
    // Relaye le message au service worker (background.js)
    chrome.runtime.sendMessage(event.detail);
});

console.log('[GUARDIAN BRIDGE] ✅ Pont de communication activé.');