// background.js - Service Worker pour Phishing Guardian (v3.1 - Interception)
console.log('[GUARDIAN BG] 🚀 Service Worker démarré');

// === 1. INJECTION DE L'INTERCEPTEUR DE NOTIFICATIONS ===
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
    // S'exécute quand un onglet Gmail ou Yahoo est chargé ou mis à jour
    if (changeInfo.status === 'complete' && tab.url &&
        (tab.url.startsWith("https://mail.google.com/") || tab.url.includes(".mail.yahoo.com/"))) {

        console.log(`[GUARDIAN BG] 📧 Page mail détectée: ${tab.url}`);

        try {
            // Injecter le script intercepteur dans le "monde" de la page principale
            await chrome.scripting.executeScript({
                target: { tabId: tabId, allFrames: true },
                files: ['interceptor.js'],
                world: 'MAIN' // Essentiel pour accéder à window.Notification
            });
            console.log(`[GUARDIAN BG] ✅ Intercepteur injecté avec succès sur tab ${tabId}`);

            // Notifier l'utilisateur que la protection est active
            await showNotification({
                type: "basic",
                iconUrl: "icons/icon48.png",
                title: "🛡️ Protection Activée",
                message: "Guardian intercepte maintenant les notifications de nouveaux mails.",
                priority: 1
            });

        } catch (error) {
            console.error(`[GUARDIAN BG] ❌ Erreur injection sur tab ${tabId}:`, error);
        }
    }
});

// === 2. ÉCOUTE DES MESSAGES (DU PONT) ===
// ... (le reste de ce fichier est identique à la version précédente avec historique et feedback)
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log(`[GUARDIAN BG] 📨 Message reçu:`, message.type);

    if (message.type === 'NEW_MAIL_DETECTED') {
        analyzeMail(message.data);
        sendResponse({ success: true, message: 'Mail en cours d\'analyse' });
    } else if (message.type === 'STATUS_REQUEST') {
        chrome.storage.local.get(['scanHistory', 'statistics'], (result) => {
            sendResponse(result);
        });
        return true;
    } else if (message.type === 'SUBMIT_FEEDBACK') {
        submitFeedback(message.data);
        sendResponse({ success: true, message: 'Feedback en cours d\'envoi' });
    }
    return true;
});


// === 3. ANALYSE DES MAILS AVEC L'API IA ===
async function analyzeMail(mailData) {
    const textToAnalyze = `${mailData.subject || ''}\n${mailData.sender || ''}\n${mailData.preview || ''}`;
    console.log(`[GUARDIAN BG] 🤖 Analyse IA en cours...`, textToAnalyze);

    const scanId = `scan_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;

    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "application/json" },
            body: JSON.stringify({ text: textToAnalyze.trim() })
        });

        if (!response.ok) throw new Error(`Erreur API: ${response.status} ${response.statusText}`);

        const result = await response.json();
        const isPhishing = result.prediction === 'phishing';

        const scanResult = {
            id: scanId,
            timestamp: Date.now(),
            subject: mailData.subject || 'Sujet non disponible',
            sender: mailData.sender || 'Expéditeur inconnu',
            isPhishing: isPhishing,
            confidence: result.confidence || 0,
            feedbackSent: false
        };

        await updateHistory(scanResult);
        await updateStatistics(isPhishing);

        if (isPhishing) {
            await showPhishingAlert(scanResult);
        }

    } catch (error) {
        console.error(`[GUARDIAN BG] ❌ Erreur lors de l'analyse:`, error);
        const errorResult = {
            id: scanId,
            timestamp: Date.now(),
            subject: mailData.subject || 'Sujet non disponible',
            sender: mailData.sender || 'Expéditeur inconnu',
            isPhishing: false,
            error: error.message
        };
        await updateHistory(errorResult);
        await showErrorNotification(error.message);
    }
}

// ... (le reste du fichier background.js - feedback, notifications, stats, init - reste identique)
async function submitFeedback(feedbackData) {
    console.log('[GUARDIAN BG] 💬 Envoi du feedback à l\'API...', feedbackData);
    try {
        const response = await fetch("http://localhost:8000/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "application/json" },
            body: JSON.stringify(feedbackData)
        });
        if (!response.ok) throw new Error(`Erreur API feedback: ${response.status}`);
        const result = await response.json();
        console.log('[GUARDIAN BG] ✅ Feedback envoyé avec succès:', result);

        const { scanHistory = [] } = await chrome.storage.local.get('scanHistory');
        const updatedHistory = scanHistory.map(item => {
            if (item.id === feedbackData.scan_id) {
                return { ...item, feedbackSent: true };
            }
            return item;
        });
        await chrome.storage.local.set({ scanHistory: updatedHistory });
    } catch (error) {
        console.error('[GUARDIAN BG] ❌ Erreur lors de l\'envoi du feedback:', error);
    }
}
async function showPhishingAlert(scanResult) {
    await showNotification({
        type: "basic", iconUrl: "icons/icon48.png",
        title: "🚨 ALERTE PHISHING !",
        message: `Mail suspect de: ${scanResult.sender}\nSujet: ${scanResult.subject.substring(0, 50)}...`,
        priority: 2, requireInteraction: true
    });
}
async function showErrorNotification(errorMessage) {
    await showNotification({
        type: "basic", iconUrl: "icons/icon48.png",
        title: "⚠️ Erreur d'Analyse",
        message: `Impossible d'analyser le mail: ${errorMessage}`,
        priority: 1
    });
}
async function showNotification(options) {
    try {
        await chrome.notifications.create(options);
    } catch (error) {
        console.error(`[GUARDIAN BG] ❌ Erreur création notification:`, error);
    }
}
async function updateHistory(scanResult) {
    const { scanHistory = [] } = await chrome.storage.local.get(['scanHistory']);
    const newHistory = [scanResult, ...scanHistory];
    const limitedHistory = newHistory.slice(0, 20);
    await chrome.storage.local.set({ scanHistory: limitedHistory });
    console.log(`[GUARDIAN BG] 📜 Historique mis à jour. Total: ${limitedHistory.length}`);
}
async function updateStatistics(isPhishing) {
    const { statistics = { totalScanned: 0, phishingDetected: 0, safeEmails: 0 } } =
        await chrome.storage.local.get(['statistics']);
    statistics.totalScanned++;
    if (isPhishing) statistics.phishingDetected++;
    else statistics.safeEmails++;
    await chrome.storage.local.set({ statistics });
    console.log(`[GUARDIAN BG] 📈 Stats mises à jour:`, statistics);
}
chrome.runtime.onInstalled.addListener(async () => {
    console.log(`[GUARDIAN BG] 🎯 Extension installée/mise à jour`);
    await chrome.storage.local.set({
        statistics: { totalScanned: 0, phishingDetected: 0, safeEmails: 0 },
        scanHistory: [],
        settings: { autoScan: true, notifications: true }
    });
});