// content.js - Version simple qui fonctionne
console.log('[GUARDIAN] 🌉 Content script simple');

// Juste relayer les messages si besoin
window.addEventListener('message', (event) => {
    if (event.source === window && event.data?.type === 'GUARDIAN_EMAIL') {
        try {
            chrome.runtime.sendMessage(event.data);
        } catch (error) {
            console.log('[GUARDIAN] Erreur relay:', error);
        }
    }
});

console.log('[GUARDIAN] ✅ Content prêt');