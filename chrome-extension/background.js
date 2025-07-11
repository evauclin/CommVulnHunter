// background.js - Analyse, notifie et sauvegarde l'historique.

console.log('🛡️ Guardian Background Service Actif.');

const API_URL = 'http://localhost:8000';
const pendingFeedbacks = new Map();

// --- GESTION DE L'HISTORIQUE via chrome.storage ---
async function getHistory() {
    try {
        const result = await chrome.storage.local.get(['notificationHistory']);
        return result.notificationHistory || [];
    } catch (e) { return []; }
}

async function addToHistory(item) {
    const history = await getHistory();
    const newHistory = [item, ...history].slice(0, 100);
    await chrome.storage.local.set({ notificationHistory: newHistory });
}

async function clearHistory() {
    await chrome.storage.local.set({ notificationHistory: [] });
}

// --- GESTION DES MESSAGES ---
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    switch (request.type) {
        case 'ANALYZE_NOTIFICATION':
            processNotification(request.data, sender.origin);
            break;
        case 'GET_HISTORY':
            getHistory().then(sendResponse);
            return true;
        case 'CLEAR_HISTORY':
            clearHistory().then(() => sendResponse({ status: 'ok' }));
            return true;
    }
});

// --- LOGIQUE D'ANALYSE ET NOTIFICATION ---
async function processNotification(notifData, origin) {
    try {
        const textToAnalyze = `${notifData.title}\n${notifData.body}`;
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textToAnalyze })
        });
        if (!response.ok) throw new Error(`API Error ${response.status}`);

        const result = await response.json();

        await addToHistory({
            title: notifData.title,
            body: notifData.body,
            origin: new URL(origin).hostname,
            isPhishing: result.prediction === 'phishing',
            timestamp: new Date().toISOString()
        });

        createReplacementNotification(notifData, origin, result);
    } catch (error) {
        console.error("❌ Erreur d'analyse:", error);
        createReplacementNotification(notifData, origin, null, true); // Crée une notification d'erreur
    }
}

function createReplacementNotification(notifData, origin, analysisResult, isError = false) {
    const notificationId = `guardian-${Date.now()}`;
    // --- CORRECTION ---
    const isPhishing = !isError && analysisResult?.prediction === 'phishing';

    let title = "✅ Notification Analysée";
    if (isPhishing) title = "⚠️ ALERTE - Contenu Suspect";
    if (isError) title = "❓ Erreur d'Analyse";

    if (!isError) {
        pendingFeedbacks.set(notificationId, { notifData, analysisResult });
    }

    chrome.notifications.create(notificationId, {
        type: 'basic',
        iconUrl: 'icons/icon128.png',
        title: title,
        message: `${notifData.title}${notifData.body ? `\n${notifData.body}` : ''}`,
        contextMessage: `Origine: ${new URL(origin).hostname}`,
        buttons: isError ? [] : [{ title: '👍 Correct' }, { title: '👎 Incorrect' }]
    });
}

// --- GESTION DU FEEDBACK ---
chrome.notifications.onButtonClicked.addListener(async (notificationId, buttonIndex) => {
    const feedbackInfo = pendingFeedbacks.get(notificationId);
    if (!feedbackInfo || !feedbackInfo.analysisResult) return;

    const userSatisfaction = (buttonIndex === 0) ? 'yes' : 'no';

    try {
        const payload = {
            email_text: `${feedbackInfo.notifData.title}\n${feedbackInfo.notifData.body}`,
            predicted_class: feedbackInfo.analysisResult.prediction,
            predicted_probability: feedbackInfo.analysisResult.probability,
            user_satisfaction: userSatisfaction
        };
        await fetch(`${API_URL}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
    } catch (error) {
        console.error("❌ Échec de l'envoi du feedback:", error);
    }

    chrome.notifications.clear(notificationId);
    pendingFeedbacks.delete(notificationId);
});