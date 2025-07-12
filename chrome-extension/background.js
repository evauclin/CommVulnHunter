// background.js

// --- 1. INJECTION DE L'ESPION (POUR GMAIL ET YAHOO) ---
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    // On vérifie que la page est complètement chargée et que l'URL correspond à l'un des deux services
    if (changeInfo.status === 'complete' && tab.url &&
        (tab.url.startsWith("https://mail.google.com/") || tab.url.includes(".mail.yahoo.com/"))
    ) {
        console.log(`[BG] Injection de l'espion sur : ${tab.url}`);
        chrome.scripting.executeScript({
            target: { tabId: tabId },
            files: ['interceptor.js'],
            world: 'MAIN',
        }).catch(err => console.error("Échec de l'injection :", err));
    }
});

// --- 2. ÉCOUTE DES NOTIFICATIONS INTERCEPTÉES (NE CHANGE PAS) ---
chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'ANALYZE_NOTIFICATION') {
        console.log("✅ [BG] Notification interceptée, analyse en cours...", message.data);
        analyzeWithAPI(message.data);
    }
});

// --- 3. LOGIQUE D'ANALYSE ET DE NOTIFICATION (NE CHANGE PAS) ---
async function analyzeWithAPI(notifData) {
    const textToAnalyze = `${notifData.title}\n${notifData.body}`;
    console.log("▶️ [BG] Envoi à l'API :", textToAnalyze);

    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: textToAnalyze })
        });

        if (!response.ok) throw new Error(`Erreur API ${response.status}`);

        const result = await response.json();
        console.log("✅ [BG] Réponse de l'API :", result);

        const isPhishing = result.prediction === 'phishing';

        await chrome.storage.local.set({ last_scan: { content: notifData.title, is_phishing: isPhishing } });

        const notificationOptions = {
            type: "basic",
            iconUrl: "icons/icon48.png",
            title: isPhishing ? "⚠️ ALERTE PHISHING !" : "✅ Message Analysé (Sûr)",
            message: `Le message original "${notifData.title}" a été analysé.`,
            priority: 2
        };
        chrome.notifications.create(notificationOptions);

    } catch (error) {
        console.error("❌ [BG] ERREUR lors de l'appel API :", error);

        const errorOptions = {
            type: "basic",
            iconUrl: "icons/icon48.png",
            title: "Erreur d'Analyse",
            message: "Impossible de contacter l'API. Vérifiez qu'elle est lancée.",
            priority: 1
        };
        // On crée la notification d'erreur. Si ça échoue, on logue l'erreur de l'icône.
        chrome.notifications.create('error_notif', errorOptions, () => {
            if (chrome.runtime.lastError) {
                console.error("Impossible de créer la notification d'erreur :", chrome.runtime.lastError.message);
            }
        });
    }
}