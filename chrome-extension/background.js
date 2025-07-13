// background.js - Version 5.4 avec meilleure gestion d'erreurs
console.log('[GUARDIAN BG] 🚀 Démarré (Version corrigée)');

// Gestion des messages
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[GUARDIAN BG] Message reçu:', message.type);

  switch (message.type) {
    case 'NEW_GMAIL_EMAIL_DETECTED':
      console.log('[GUARDIAN BG] 📧 Email détecté:', message.data);
      analyzeMail(message.data)
        .then(() => sendResponse({ received: true }))
        .catch(error => {
          console.error('[GUARDIAN BG] Erreur analyse:', error);
          sendResponse({ received: false, error: error.message });
        });
      return true; // Réponse asynchrone

    case 'STATUS_REQUEST':
      chrome.storage.local.get(['scanHistory', 'statistics'])
        .then(sendResponse)
        .catch(error => sendResponse({ error: error.message }));
      return true;

    case 'SUBMIT_FEEDBACK':
      submitFeedback(message.data)
        .then(() => sendResponse({ success: true }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true;

    default:
      console.warn('[GUARDIAN BG] Type de message inconnu:', message.type);
      sendResponse({ error: 'Type de message non reconnu' });
  }
});

// Analyse d'email avec timeout
async function analyzeMail(mailData) {
  const scanId = `scan_${Date.now()}`;
  const timestamp = Date.now();
  const textToAnalyze = `From: ${mailData.sender}\nSubject: ${mailData.subject}`;

  let scanResult = {
    id: scanId,
    timestamp,
    ...mailData,
    status: 'analyzing'
  };

  try {
    console.log('[GUARDIAN BG] 🔍 Analyse en cours pour:', mailData.subject);

    // Timeout de 10 secondes pour l'API
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: textToAnalyze }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Erreur API ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    const isPhishing = result.prediction === 'phishing';
    const confidence = result.probability || 0;

    scanResult = {
      ...scanResult,
      isPhishing,
      confidence,
      status: 'completed'
    };

    console.log('[GUARDIAN BG] ✅ Analyse terminée:', {
      subject: mailData.subject,
      isPhishing,
      confidence
    });

    if (isPhishing) {
      showPhishingAlert(scanResult);
    }

  } catch (error) {
    console.error('[GUARDIAN BG] ❌ Erreur analyse:', error);
    scanResult.error = error.message;
    scanResult.status = 'error';

    if (error.name === 'AbortError') {
      scanResult.error = 'Timeout de l\'API (10s)';
    }

    showErrorNotification(scanResult.error);
  }

  await updateHistoryAndStats(scanResult);
  return scanResult;
}

// Reste du code background.js inchangé...

// Reste du code background.js inchangé...


// --- GESTION DU FEEDBACK UTILISATEUR ---
async function submitFeedback(feedbackData) {
  const apiPayload = {
    email_text: `Subject: ${feedbackData.subject}\nFrom: ${feedbackData.sender}`,
    predicted_class: feedbackData.original_prediction,
    predicted_probability: 0.5,
    user_satisfaction: feedbackData.user_feedback === 'correct' ? 'yes' : 'no',
    language_detected: 'en'
  };

  try {
    const response = await fetch("http://localhost:8000/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(apiPayload)
    });

    if (response.ok) {
      const { scanHistory = [] } = await chrome.storage.local.get('scanHistory');
      const updatedHistory = scanHistory.map(item =>
        item.id === feedbackData.scan_id ? { ...item, feedbackSent: true } : item
      );
      await chrome.storage.local.set({ scanHistory: updatedHistory });
    }
  } catch (error) {
    console.error('[GUARDIAN BG] ❌ Erreur feedback:', error);
  }
}


// --- SAUVEGARDE HISTORIQUE + STATISTIQUES ---
async function updateHistoryAndStats(scanResult) {
  const data = await chrome.storage.local.get(['scanHistory', 'statistics']);
  const scanHistory = data.scanHistory || [];
  const statistics = data.statistics || { totalScanned: 0, phishingDetected: 0, safeEmails: 0 };

  const newHistory = [scanResult, ...scanHistory].slice(0, 20);

  if (!scanResult.error) {
    statistics.totalScanned++;
    scanResult.isPhishing ? statistics.phishingDetected++ : statistics.safeEmails++;
  }

  await chrome.storage.local.set({ scanHistory: newHistory, statistics });
}


// --- NOTIFICATIONS ---
function showPhishingAlert(res) {
  chrome.notifications.create({
    type: "basic",
    iconUrl: "icons/icon48.png",
    title: "🚨 ALERTE PHISHING !",
    message: `Mail suspect de: ${res.sender}`,
    priority: 2,
    requireInteraction: true
  });
}

function showErrorNotification(msg) {
  chrome.notifications.create({
    type: "basic",
    iconUrl: "icons/icon48.png",
    title: "⚠️ Erreur Guardian",
    message: `Détail: ${msg}`
  });
}


// --- INITIALISATION STORAGE À L'INSTALLATION ---
chrome.runtime.onInstalled.addListener(async (details) => {
  if (details.reason === 'install') {
    chrome.storage.local.set({
      statistics: { totalScanned: 0, phishingDetected: 0, safeEmails: 0 },
      scanHistory: []
    });
  }
});
