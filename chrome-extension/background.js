// background.js - Service Worker pour Phishing Guardian
console.log('[GUARDIAN BG] 🚀 Service Worker démarré');

// === 1. INJECTION DU DÉTECTEUR DE MAILS ===
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
    // Vérifier que la page est complètement chargée et sur Gmail/Yahoo
    if (changeInfo.status === 'complete' && tab.url &&
        (tab.url.startsWith("https://mail.google.com/") || tab.url.includes(".mail.yahoo.com/"))) {

        console.log(`[GUARDIAN BG] 📧 Page mail détectée: ${tab.url}`);

        try {
            // Attendre que la page soit vraiment prête
            await new Promise(resolve => setTimeout(resolve, 3000));

            // Injecter le détecteur de mails
            await chrome.scripting.executeScript({
                target: { tabId: tabId },
                files: ['mail-detector.js'],
                world: 'MAIN'
            });

            console.log(`[GUARDIAN BG] ✅ Détecteur injecté avec succès sur tab ${tabId}`);

            // Notifier l'utilisateur que la protection est active
            await showNotification({
                type: "basic",
                iconUrl: "icons/icon48.png",
                title: "🛡️ Protection Activée",
                message: "Guardian surveille maintenant vos nouveaux mails",
                priority: 1
            });

        } catch (error) {
            console.error(`[GUARDIAN BG] ❌ Erreur injection sur tab ${tabId}:`, error);
        }
    }
});

// === 2. ÉCOUTE DES NOTIFICATIONS DE NOUVEAUX MAILS ===
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log(`[GUARDIAN BG] 📨 Message reçu:`, message.type, `de tab:`, sender.tab?.id);

    if (message.type === 'NEW_MAIL_DETECTED') {
        console.log(`[GUARDIAN BG] 🔍 Nouveau mail détecté:`, message.data);
        analyzeMail(message.data);
        sendResponse({ success: true, message: 'Mail en cours d\'analyse' });

    } else if (message.type === 'PING') {
        console.log(`[GUARDIAN BG] 🏓 Ping reçu du content script`);
        sendResponse({ pong: true, timestamp: Date.now() });

    } else if (message.type === 'STATUS_REQUEST') {
        // Demande de statut pour le popup
        chrome.storage.local.get(['lastScan', 'statistics'], (result) => {
            sendResponse(result);
        });
        return true; // Réponse asynchrone
    }

    return true;
});

// === 3. ANALYSE DES MAILS AVEC L'API IA ===
async function analyzeMail(mailData) {
    const textToAnalyze = `${mailData.subject || ''}\n${mailData.sender || ''}\n${mailData.preview || ''}`;

    console.log(`[GUARDIAN BG] 🤖 Analyse IA en cours...`);
    console.log(`[GUARDIAN BG] 📝 Texte à analyser:`, textToAnalyze);

    try {
        // Appel à votre API de détection de phishing
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({
                text: textToAnalyze.trim()
            })
        });

        if (!response.ok) {
            throw new Error(`Erreur API: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        console.log(`[GUARDIAN BG] ✅ Réponse API:`, result);

        const isPhishing = result.prediction === 'phishing';
        const confidence = result.confidence || 0;

        // Sauvegarder le résultat
        const scanResult = {
            timestamp: Date.now(),
            subject: mailData.subject || 'Sujet non disponible',
            sender: mailData.sender || 'Expéditeur inconnu',
            isPhishing: isPhishing,
            confidence: confidence,
            source: mailData.source || 'Détection auto'
        };

        await chrome.storage.local.set({
            lastScan: scanResult
        });

        // Mettre à jour les statistiques
        await updateStatistics(isPhishing);

        // Créer la notification appropriée
        if (isPhishing) {
            await showPhishingAlert(scanResult);
        } else {
            await showSafeNotification(scanResult);
        }

        console.log(`[GUARDIAN BG] 📊 Analyse terminée: ${isPhishing ? 'PHISHING DÉTECTÉ' : 'MAIL SÛR'}`);

    } catch (error) {
        console.error(`[GUARDIAN BG] ❌ Erreur lors de l'analyse:`, error);

        // Sauvegarder l'erreur
        const errorResult = {
            timestamp: Date.now(),
            subject: mailData.subject || 'Sujet non disponible',
            sender: mailData.sender || 'Expéditeur inconnu',
            isPhishing: false,
            error: error.message,
            source: mailData.source || 'Détection auto'
        };

        await chrome.storage.local.set({ lastScan: errorResult });
        await showErrorNotification(error.message);
    }
}

// === 4. NOTIFICATIONS ===
async function showPhishingAlert(scanResult) {
    await showNotification({
        type: "basic",
        iconUrl: "icons/icon48.png",
        title: "🚨 ALERTE PHISHING !",
        message: `Mail suspect de: ${scanResult.sender}\nSujet: ${scanResult.subject.substring(0, 50)}...`,
        priority: 2,
        requireInteraction: true
    });
}

async function showSafeNotification(scanResult) {
    await showNotification({
        type: "basic",
        iconUrl: "icons/icon48.png",
        title: "✅ Mail Analysé - Sûr",
        message: `De: ${scanResult.sender}\nConfiance: ${Math.round(scanResult.confidence * 100)}%`,
        priority: 1
    });
}

async function showErrorNotification(errorMessage) {
    await showNotification({
        type: "basic",
        iconUrl: "icons/icon48.png",
        title: "⚠️ Erreur d'Analyse",
        message: `Impossible d'analyser le mail: ${errorMessage}`,
        priority: 1
    });
}

// Fonction helper pour créer des notifications
async function showNotification(options) {
    try {
        const notificationId = await chrome.notifications.create(options);
        console.log(`[GUARDIAN BG] 🔔 Notification créée: ${notificationId}`);
        return notificationId;
    } catch (error) {
        console.error(`[GUARDIAN BG] ❌ Erreur création notification:`, error);
    }
}

// === 5. STATISTIQUES ===
async function updateStatistics(isPhishing) {
    const { statistics = { totalScanned: 0, phishingDetected: 0, safeEmails: 0 } } =
        await chrome.storage.local.get(['statistics']);

    statistics.totalScanned++;
    if (isPhishing) {
        statistics.phishingDetected++;
    } else {
        statistics.safeEmails++;
    }

    await chrome.storage.local.set({ statistics });
    console.log(`[GUARDIAN BG] 📈 Stats mises à jour:`, statistics);
}

// === 6. INITIALISATION ===
chrome.runtime.onInstalled.addListener(async () => {
    console.log(`[GUARDIAN BG] 🎯 Extension installée/mise à jour`);

    // Initialiser les statistiques
    await chrome.storage.local.set({
        statistics: { totalScanned: 0, phishingDetected: 0, safeEmails: 0 },
        settings: { autoScan: true, notifications: true }
    });
});

chrome.runtime.onStartup.addListener(() => {
    console.log(`[GUARDIAN BG] 🌅 Extension démarrée`);
});