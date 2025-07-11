// background.js - Service Worker pour l'extension AI Anti-Phishing Guardian

console.log('🛡️ AI Anti-Phishing Guardian - Background Script activé');

class BackgroundManager {
    constructor() {
        this.analysisHistory = new Map();
        this.apiUrl = 'http://localhost:8000'; // Votre API FastAPI
        this.settings = {
            enableNotifications: true,
            enableAutoAnalysis: true,
            confidenceThreshold: 70,
            enableRealTimeProtection: true,
            showSafeIndicator: true
        };

        this.stats = {
            sitesAnalyzed: 0,
            threatsBlocked: 0,
            lastAnalysis: null,
            apiStatus: 'unknown'
        };

        this.init();
    }

    async init() {
        // Charger les paramètres sauvegardés
        await this.loadSettings();

        // Vérifier le statut de l'API
        await this.checkApiStatus();

        // Configurer les listeners
        this.setupMessageListeners();
        this.setupNotificationListeners();
        this.setupTabListeners();
        this.setupAlarmListeners();

        // Démarrer les tâches périodiques
        this.startPeriodicTasks();

        console.log('✅ Background Manager initialisé');
        console.log('📊 API Status:', this.stats.apiStatus);
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.sync.get('phishingDetectorSettings');
            if (result.phishingDetectorSettings) {
                this.settings = { ...this.settings, ...result.phishingDetectorSettings };
            }
            console.log('⚙️ Paramètres chargés:', this.settings);
        } catch (error) {
            console.warn('⚠️ Impossible de charger les paramètres:', error);
        }
    }

    async saveSettings() {
        try {
            await chrome.storage.sync.set({
                phishingDetectorSettings: this.settings
            });
            console.log('💾 Paramètres sauvegardés');
        } catch (error) {
            console.error('❌ Erreur sauvegarde paramètres:', error);
        }
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiUrl}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(5000)
            });

            if (response.ok) {
                const data = await response.json();
                this.stats.apiStatus = 'online';
                console.log('✅ API ML en ligne:', data.status);

                // Mettre à jour les informations de fine-tuning si disponibles
                if (data.is_finetuning_running !== undefined) {
                    this.stats.isFinetuningRunning = data.is_finetuning_running;
                }
            } else {
                this.stats.apiStatus = 'error';
                console.warn('⚠️ API ML répond avec erreur:', response.status);
            }
        } catch (error) {
            this.stats.apiStatus = 'offline';
            console.warn('❌ API ML non accessible:', error.message);
        }
    }

    setupMessageListeners() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Permet les réponses asynchrones
        });
    }

    async handleMessage(request, sender, sendResponse) {
        try {
            switch (request.type) {
                case 'ANALYSIS_COMPLETE':
                    await this.handleAnalysisComplete(request.data, sender.tab);
                    sendResponse({ status: 'handled' });
                    break;

                case 'SHOW_NOTIFICATION':
                    if (this.settings.enableNotifications) {
                        await this.showNotification(request.title, request.message, request.type);
                    }
                    sendResponse({ status: 'notification_shown' });
                    break;

                case 'GET_SETTINGS':
                    sendResponse({ settings: this.settings });
                    break;

                case 'UPDATE_SETTINGS':
                    this.settings = { ...this.settings, ...request.settings };
                    await this.saveSettings();
                    sendResponse({ status: 'settings_updated' });
                    break;

                case 'GET_HISTORY':
                    const history = Array.from(this.analysisHistory.entries()).map(([url, data]) => ({
                        url,
                        ...data
                    }));
                    sendResponse({ history: history.slice(-100) }); // Dernières 100 analyses
                    break;

                case 'CLEAR_HISTORY':
                    this.analysisHistory.clear();
                    this.stats.sitesAnalyzed = 0;
                    this.stats.threatsBlocked = 0;
                    sendResponse({ status: 'history_cleared' });
                    break;

                case 'GET_STATS':
                    await this.updateStats();
                    sendResponse({ stats: this.stats });
                    break;

                case 'CHECK_API_STATUS':
                    await this.checkApiStatus();
                    sendResponse({ apiStatus: this.stats.apiStatus });
                    break;

                case 'FORCE_RECHECK_TAB':
                    if (sender.tab) {
                        await this.recheckTab(sender.tab.id);
                    }
                    sendResponse({ status: 'recheck_initiated' });
                    break;

                default:
                    sendResponse({ error: 'Unknown message type' });
            }
        } catch (error) {
            console.error('❌ Erreur handling message:', error);
            sendResponse({ error: error.message });
        }
    }

    async handleAnalysisComplete(data, tab) {
        // Sauvegarder dans l'historique
        this.analysisHistory.set(data.url, {
            ...data,
            timestamp: new Date().toISOString(),
            tabId: tab?.id,
            tabTitle: tab?.title,
            domain: data.domain || new URL(data.url).hostname
        });

        // Mettre à jour les statistiques
        this.stats.sitesAnalyzed++;
        this.stats.lastAnalysis = new Date().toISOString();

        if (data.isPhishing) {
            this.stats.threatsBlocked++;
        }

        // Mettre à jour l'icône de l'extension
        await this.updateExtensionIcon(data, tab?.id);

        // Gérer les menaces détectées
        if (data.isPhishing && data.confidence >= this.settings.confidenceThreshold) {
            await this.handleThreatDetected(data, tab);
        }

        // Log pour debugging
        console.log(`📊 Analyse: ${data.domain} - ${data.isPhishing ? 'MENACE' : 'SÛR'} (${data.confidence}%)`);
    }

    async updateExtensionIcon(data, tabId) {
        if (!tabId) return;

        try {
            let iconSuffix = '';
            let badgeText = '';
            let badgeColor = '#666';

            if (data.isPhishing) {
                iconSuffix = '_danger';
                badgeText = '⚠️';
                badgeColor = '#f44336';
            } else {
                iconSuffix = '_safe';
                badgeText = '✓';
                badgeColor = '#4caf50';
            }

            // Mettre à jour l'icône (si vous avez les variantes)
            await chrome.action.setIcon({
                path: {
                    "16": `icons/icon16${iconSuffix}.png`,
                    "32": `icons/icon32${iconSuffix}.png`,
                    "48": `icons/icon48${iconSuffix}.png`,
                    "128": `icons/icon128${iconSuffix}.png`
                },
                tabId
            }).catch(() => {
                // Fallback vers l'icône normale si les variantes n'existent pas
                console.log('⚠️ Icônes variantes non trouvées, utilisation icône par défaut');
            });

            await chrome.action.setBadgeText({ text: badgeText, tabId });
            await chrome.action.setBadgeBackgroundColor({ color: badgeColor, tabId });

        } catch (error) {
            console.warn('⚠️ Impossible de mettre à jour l\'icône:', error);
        }
    }

    async handleThreatDetected(data, tab) {
        // Notification système prioritaire
        if (this.settings.enableNotifications) {
            await this.showNotification(
                '🚨 Menace détectée !',
                `${data.domain} pourrait être dangereux (Confiance: ${data.confidence}%)`,
                'danger'
            );
        }

        // Log de sécurité
        console.warn(`🚨 MENACE DÉTECTÉE: ${data.url}`);
        console.warn(`   Domain: ${data.domain}`);
        console.warn(`   Confiance: ${data.confidence}%`);
        console.warn(`   Titre: ${tab?.title || 'N/A'}`);

        // Optionnel: Envoyer des analytics anonymes vers votre API
        this.sendThreatAnalytics(data, tab);
    }

    async sendThreatAnalytics(data, tab) {
        try {
            // Envoyer des données anonymisées pour améliorer le modèle
            const analyticsData = {
                domain: data.domain,
                isPhishing: data.isPhishing,
                confidence: data.confidence,
                hasLoginForm: data.pageData?.hasLoginForm || false,
                hasPaymentForm: data.pageData?.hasPaymentForm || false,
                suspiciousCount: data.pageData?.suspiciousCount || 0,
                timestamp: new Date().toISOString(),
                userAgent: navigator.userAgent.substring(0, 100), // Limité pour la vie privée
                language: navigator.language
            };

            await fetch(`${this.apiUrl}/analytics/threat-detection`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(analyticsData)
            }).catch(() => {
                // Ignorer silencieusement les erreurs d'analytics
            });

        } catch (error) {
            // Ignorer les erreurs d'analytics
        }
    }

    async showNotification(title, message, type = 'info') {
        try {
            const iconUrl = type === 'danger' ? 'icons/icon48.png' : 'icons/icon48.png';

            await chrome.notifications.create({
                type: 'basic',
                iconUrl: iconUrl,
                title: title,
                message: message,
                priority: type === 'danger' ? 2 : 1,
                requireInteraction: type === 'danger' // Les menaces nécessitent une interaction
            });
        } catch (error) {
            console.warn('⚠️ Impossible d\'afficher la notification:', error);
        }
    }

    setupNotificationListeners() {
        chrome.notifications.onClicked.addListener((notificationId) => {
            // Ouvrir le popup ou le dashboard
            chrome.tabs.create({
                url: `${this.apiUrl}` // Rediriger vers votre dashboard ML
            });
        });

        chrome.notifications.onButtonClicked.addListener((notificationId, buttonIndex) => {
            // Gérer les boutons de notification si nécessaire
            console.log('Notification button clicked:', buttonIndex);
        });
    }

    setupTabListeners() {
        // Réinitialiser l'icône quand on change d'onglet
        chrome.tabs.onActivated.addListener(async (activeInfo) => {
            await this.resetIconForTab(activeInfo.tabId);
        });

        // Nettoyer l'historique quand un onglet est fermé
        chrome.tabs.onRemoved.addListener((tabId) => {
            // Supprimer les analyses de cet onglet après délai
            setTimeout(() => {
                for (const [url, data] of this.analysisHistory.entries()) {
                    if (data.tabId === tabId) {
                        this.analysisHistory.delete(url);
                    }
                }
            }, 300000); // 5 minutes
        });

        // Surveiller les changements d'URL
        chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
            if (changeInfo.status === 'complete' && tab.url) {
                await this.resetIconForTab(tabId);
            }
        });
    }

    setupAlarmListeners() {
        chrome.alarms.onAlarm.addListener((alarm) => {
            switch (alarm.name) {
                case 'api-health-check':
                    this.checkApiStatus();
                    break;
                case 'cleanup-history':
                    this.cleanupOldHistory();
                    break;
            }
        });
    }

    startPeriodicTasks() {
        // Vérifier l'API toutes les 5 minutes
        chrome.alarms.create('api-health-check', { periodInMinutes: 5 });

        // Nettoyer l'historique tous les jours
        chrome.alarms.create('cleanup-history', { periodInMinutes: 1440 });
    }

    async resetIconForTab(tabId) {
        try {
            await chrome.action.setIcon({
                path: {
                    "16": "icons/icon16.png",
                    "32": "icons/icon32.png",
                    "48": "icons/icon48.png",
                    "128": "icons/icon128.png"
                },
                tabId
            });
            await chrome.action.setBadgeText({ text: '', tabId });
        } catch (error) {
            console.warn('⚠️ Impossible de réinitialiser l\'icône:', error);
        }
    }

    async recheckTab(tabId) {
        try {
            await chrome.tabs.sendMessage(tabId, { type: 'REANALYZE_PAGE' });
        } catch (error) {
            console.warn('⚠️ Impossible de relancer l\'analyse:', error);
        }
    }

    cleanupOldHistory() {
        const now = new Date();
        const maxAge = 7 * 24 * 60 * 60 * 1000; // 7 jours

        for (const [url, data] of this.analysisHistory.entries()) {
            try {
                const analysisDate = new Date(data.timestamp);
                if (now - analysisDate > maxAge) {
                    this.analysisHistory.delete(url);
                }
            } catch (error) {
                // Supprimer les entrées avec des timestamps invalides
                this.analysisHistory.delete(url);
            }
        }

        console.log(`🧹 Historique nettoyé: ${this.analysisHistory.size} entrées restantes`);
    }

    async updateStats() {
        const history = Array.from(this.analysisHistory.values());

        this.stats.sitesAnalyzed = history.length;
        this.stats.threatsBlocked = history.filter(item => item.isPhishing).length;

        if (history.length > 0) {
            this.stats.lastAnalysis = Math.max(...history.map(item => new Date(item.timestamp))).toISOString();
        }

        // Calculer le taux de réussite
        this.stats.successRate = this.stats.sitesAnalyzed > 0
            ? Math.round(((this.stats.sitesAnalyzed - this.stats.threatsBlocked) / this.stats.sitesAnalyzed) * 100)
            : 100;
    }

    // Méthode publique pour obtenir les statistiques
    getStats() {
        return this.stats;
    }
}

// Initialiser le gestionnaire
const backgroundManager = new BackgroundManager();

// Listener pour l'installation de l'extension
chrome.runtime.onInstalled.addListener(async (details) => {
    console.log('🎉 Extension installée/mise à jour:', details.reason);

    if (details.reason === 'install') {
        console.log('🆕 Première installation');

        // Créer les alarmes
        await backgroundManager.startPeriodicTasks();

        // Ouvrir la page de bienvenue
        chrome.tabs.create({
            url: chrome.runtime.getURL('popup.html')
        });

        // Notification de bienvenue
        if (backgroundManager.settings.enableNotifications) {
            await backgroundManager.showNotification(
                '🛡️ Extension installée !',
                'AI Anti-Phishing Guardian vous protège maintenant',
                'info'
            );
        }

    } else if (details.reason === 'update') {
        console.log('🔄 Extension mise à jour vers', chrome.runtime.getManifest().version);
    }
});

// Listener pour le démarrage de Chrome
chrome.runtime.onStartup.addListener(() => {
    console.log('🚀 Chrome démarré, extension prête');
    backgroundManager.checkApiStatus();
});

// Gérer les erreurs non capturées
self.addEventListener('error', (event) => {
    console.error('❌ Erreur background script:', event.error);
});

self.addEventListener('unhandledrejection', (event) => {
    console.error('❌ Promise rejetée background script:', event.reason);
});

console.log('✅ Background script initialisé - Service Worker actif');