// content-loader.js - Injecte l'intercepteur dans la page.

try {
    const script = document.createElement('script');
    script.src = chrome.runtime.getURL('interceptor.js');
    (document.head || document.documentElement).appendChild(script);
    script.onload = () => script.remove();
    console.log('[Loader] Script d\'interception injecté.');

    // Écoute les événements de l'intercepteur et les relaie au background
    window.addEventListener('AI_Guardian_InterceptedNotification', (event) => {
        chrome.runtime.sendMessage(event.detail);
    });

} catch (e) {
    console.error('[Loader] Erreur d\'injection :', e);
}