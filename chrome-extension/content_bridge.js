// content_bridge.js - Version Robuste
(() => {
  'use strict';
  console.log('[GUARDIAN BRIDGE] 🌉 Pont robuste activé.');

  // Écoute des notifications de l'intercepteur
  window.addEventListener('Guardian_NotificationIntercepted', (event) => {
    // Essayer d'envoyer le message au background
    sendMessageToBackground(event.detail);
  });

  // Fonction d'envoi qui gère les erreurs
  function sendMessageToBackground(message) {
    try {
      chrome.runtime.sendMessage(message, (response) => {
        // chrome.runtime.lastError est la manière officielle de détecter ce bug
        if (chrome.runtime.lastError) {
          console.warn(`[GUARDIAN BRIDGE] ⚠️ Contexte invalide. Le message n'a pas pu être envoyé. Erreur: ${chrome.runtime.lastError.message}`);
          // Pas besoin de réessayer ici, l'utilisateur doit recharger la page ou l'extension se réveillera.
        }
      });
    } catch (error) {
      console.error('[GUARDIAN BRIDGE] ❌ Exception lors de l\'envoi. Le contexte est probablement invalide.', error);
    }
  }

  // Surveillance de la navigation pour les Single Page Apps (comme Gmail)
  let currentUrl = location.href;
  setInterval(() => {
    if (location.href !== currentUrl) {
      console.log('[GUARDIAN BRIDGE] 🔄 Navigation détectée.');
      currentUrl = location.href;
      // Demander une ré-injection de l'intercepteur car la page a changé
      setTimeout(() => sendMessageToBackground({ type: 'REQUEST_REINJECTION' }), 1000);
    }
  }, 2000);

})();