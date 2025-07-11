// popup.js - Logic de l'interface utilisateur du popup

class PopupManager {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.currentTab = 'status';
        this.settings = {
            enableNotifications: true,
            enableAutoAnalysis: true,
            confidenceThreshold: 70,
            showSafeIndicator: true
        };
        this.stats = {
            sitesAnalyzed: 0,
            threatsBlocked: 0,
            successRate: 100,
            apiStatus: 'checking'
        };
        this.currentSiteData = null;

        this.init();
    }

    async init() {
        console.log('🚀 Initialisation du popup...');

        // Setup des event listeners
        this.setupEventListeners();

        // Charger les données initiales
        await this.loadInitialData();

        console.log('✅ Popup initialisé');
    }

    setupEventListeners() {
        // Navigation par onglets
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.currentTarget.dataset.tab;
                this.switchTab(tabName);
            });
        });

        // Boutons d'action
        document.getElementById('reanalyzeBtn').addEventListener('click', () => this.reanalyzePage());
        document.getElementById('openDashboardBtn').addEventListener('click', () => this.openDashboard());
        document.getElementById('reportFalsePositiveBtn').addEventListener('click', () => this.reportFalsePositive());
        document.getElementById('clearHistoryBtn').addEventListener('click', () => this.clearHistory());
        document.getElementById('checkApiBtn').addEventListener('click', () => this.checkApiStatus());
        document.getElementById('saveSettingsBtn').addEventListener('click', () => this.saveSettings());

        // Paramètres - changements automatiques
        document.querySelectorAll('#tab-settings input, #tab-settings select').forEach(element => {
            element.addEventListener('change', () => {
                if (element.type !== 'checkbox' || element.checked !== undefined) {
                    this.onSettingChange();
                }
            });
        });

        // Auto-refresh périodique
        setInterval(() => {
            if (this.currentTab === 'status') {
                this.refreshCurrentSiteStatus();
            } else if (this.currentTab === 'history') {
                this.loadHistory();
            }
        }, 10000); // Refresh toutes les 10 secondes
    }

    async loadInitialData() {
        try {
            // Charger les paramètres
            await this.loadSettings();

            // Charger le statut du site actuel
            await this.loadCurrentSiteStatus();

            // Charger les statistiques
            await this.loadStats();

            // Vérifier le statut de l'API
            await this.checkApiStatus();

        } catch (error) {
            console.error('❌ Erreur chargement données initiales:', error);
            this.showToast('Erreur de chargement des données', 'error');
        }
    }

    async loadCurrentSiteStatus() {
        try {
            // Obtenir l'onglet actuel
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab) {
                this.updateSiteDisplay({ error: 'Aucun onglet actif' });
                return;
            }

            // Mettre à jour l'affichage de l'URL
            this.updateUrlDisplay(tab.url);

            // Demander le statut au content script
            try {
                const response = await chrome.tabs.sendMessage(tab.id, { type: 'GET_PAGE_STATUS' });

                if (response && !response.error) {
                    this.currentSiteData = response;
                    this.updateSiteStatusDisplay(response);
                } else {
                    this.updateSiteDisplay({
                        error: response?.error || 'Extension non active sur cette page',
                        url: tab.url
                    });
                }
            } catch (error) {
                // Content script probablement pas chargé (page système, etc.)
                this.updateSiteDisplay({
                    error: 'Page non analysable (page système)',
                    url: tab.url
                });
            }

        } catch (error) {
            console.error('❌ Erreur chargement statut site:', error);
            this.updateSiteDisplay({ error: 'Erreur de communication' });
        }
    }

    updateUrlDisplay(url) {
        try {
            const urlObj = new URL(url);
            document.getElementById('currentUrl').textContent = urlObj.hostname + urlObj.pathname;
            document.getElementById('currentDomain').textContent = urlObj.hostname;
        } catch {
            document.getElementById('currentUrl').textContent = 'URL invalide';
            document.getElementById('currentDomain').textContent = '';
        }
    }

    updateSiteStatusDisplay(data) {
        const statusCard = document.getElementById('siteStatusCard');
        const statusIcon = document.getElementById('statusIcon');
        const statusText = document.getElementById('statusText');
        const statusConfidence = document.getElementById('statusConfidence');
        const confidenceBar = document.getElementById('confidenceBar');
        const confidenceFill = document.getElementById('confidenceFill');
        const reportBtn = document.getElementById('reportFalsePositiveBtn');

        // Reset classes
        statusCard.className = 'site-status-card';

        if (data.error) {
            statusCard.classList.add('unknown');
            statusIcon.textContent = '❓';
            statusText.textContent = data.error;
            statusConfidence.textContent = '';
            confidenceBar.style.display = 'none';
            reportBtn.style.display = 'none';
            return;
        }

        if (data.isAnalyzing) {
            statusCard.classList.add('analyzing');
            statusIcon.textContent = '🔍';
            statusText.textContent = 'Analyse en cours...';
            statusConfidence.textContent = 'Vérification par IA...';
            confidenceBar.style.display = 'block';
            confidenceFill.style.width = '60%';
            reportBtn.style.display = 'none';

            // Désactiver le bouton re-analyser
            document.getElementById('reanalyzeBtn').disabled = true;
        } else if (data.lastResult) {
            const result = data.lastResult;
            const isPhishing = result.prediction === 'phishing';
            const confidence = this.getConfidenceScore(result);

            if (isPhishing) {
                statusCard.classList.add('danger');
                statusIcon.textContent = '⚠️';
                statusText.textContent = 'Site suspect détecté';
                statusConfidence.textContent = `Confiance: ${confidence}%`;
                reportBtn.style.display = 'inline-flex';
            } else {
                statusCard.classList.add('safe');
                statusIcon.textContent = '✅';
                statusText.textContent = 'Site vérifié comme sûr';
                statusConfidence.textContent = `Confiance: ${100 - confidence}%`;
                reportBtn.style.display = 'none';
            }

            confidenceBar.style.display = 'block';
            confidenceFill.style.width = confidence + '%';
            document.getElementById('reanalyzeBtn').disabled = false;
        } else {
            statusCard.classList.add('unknown');
            statusIcon.textContent = '❓';
            statusText.textContent = 'Pas encore analysé';
            statusConfidence.textContent = 'Cliquez sur Re-analyser';
            confidenceBar.style.display = 'none';
            reportBtn.style.display = 'none';
            document.getElementById('reanalyzeBtn').disabled = false;
        }
    }

    updateSiteDisplay(data) {
        // Version simplifiée pour les erreurs
        const statusCard = document.getElementById('siteStatusCard');
        const statusIcon = document.getElementById('statusIcon');
        const statusText = document.getElementById('statusText');
        const statusConfidence = document.getElementById('statusConfidence');

        statusCard.className = 'site-status-card unknown';
        statusIcon.textContent = '❓';
        statusText.textContent = data.error || 'Erreur inconnue';
        statusConfidence.textContent = '';
        document.getElementById('confidenceBar').style.display = 'none';
        document.getElementById('reportFalsePositiveBtn').style.display = 'none';
    }

    getConfidenceScore(result) {
        if (typeof result.probability === 'number') {
            return Math.round(result.probability * 100);
        }

        // Fallback basé sur le niveau de confiance
        switch (result.confidence) {
            case 'HIGH': return 85;
            case 'MEDIUM': return 65;
            case 'LOW': return 35;
            default: return 50;
        }
    }

    async refreshCurrentSiteStatus() {
        // Refresh silencieux du statut
        if (this.currentTab === 'status') {
            await this.loadCurrentSiteStatus();
        }
    }

    async loadStats() {
        try {
            const response = await chrome.runtime.sendMessage({ type: 'GET_STATS' });
            if (response && response.stats) {
                this.stats = response.stats;
                this.updateStatsDisplay();
            }
        } catch (error) {
            console.error('❌ Erreur chargement stats:', error);
        }
    }

    updateStatsDisplay() {
        document.getElementById('sitesAnalyzed').textContent = this.stats.sitesAnalyzed || 0;
        document.getElementById('threatsBlocked').textContent = this.stats.threatsBlocked || 0;
        document.getElementById('successRate').textContent = (this.stats.successRate || 100) + '%';

        // Statut API
        const apiStatusText = document.getElementById('apiStatusText');
        switch (this.stats.apiStatus) {
            case 'online':
                apiStatusText.textContent = '🟢';
                break;
            case 'offline':
                apiStatusText.textContent = '🔴';
                break;
            default:
                apiStatusText.textContent = '🔄';
        }
    }

    async loadHistory() {
        try {
            const response = await chrome.runtime.sendMessage({ type: 'GET_HISTORY' });
            const historyList = document.getElementById('historyList');

            if (response && response.history && response.history.length > 0) {
                historyList.innerHTML = response.history.slice(-10).reverse().map(item => {
                    const domain = this.extractDomain(item.url);
                    const time = this.formatTime(item.timestamp);
                    const badgeClass = item.isPhishing ? 'danger' : 'safe';
                    const badgeText = item.isPhishing ? 'SUSPECT' : 'SÛR';

                    return `
                        <div class="history-item ${badgeClass}">
                            <div class="history-info">
                                <div class="history-domain">${domain}</div>
                                <div class="history-time">${time}</div>
                            </div>
                            <div class="history-badge ${badgeClass}">${badgeText}</div>
                        </div>
                    `;
                }).join('');
            } else {
                historyList.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">📊</div>
                        <div class="empty-state-text">Aucun historique disponible</div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('❌ Erreur chargement historique:', error);
        }
    }

    async loadSettings() {
        try {
            const response = await chrome.runtime.sendMessage({ type: 'GET_SETTINGS' });
            if (response && response.settings) {
                this.settings = response.settings;
                this.updateSettingsDisplay();
            }
        } catch (error) {
            console.error('❌ Erreur chargement paramètres:', error);
        }
    }

    updateSettingsDisplay() {
        document.getElementById('enableNotifications').checked = this.settings.enableNotifications;
        document.getElementById('enableAutoAnalysis').checked = this.settings.enableAutoAnalysis;
        document.getElementById('showSafeIndicator').checked = this.settings.showSafeIndicator;
        document.getElementById('confidenceThreshold').value = this.settings.confidenceThreshold;
    }

    async checkApiStatus() {
        try {
            const apiStatus = document.getElementById('apiStatus');
            const apiConnectionStatus = document.getElementById('apiConnectionStatus');
            const checkBtn = document.getElementById('checkApiBtn');

            // Mettre à jour l'interface pour indiquer la vérification
            apiStatus.innerHTML = `
                <div class="status-dot checking"></div>
                <span>Vérification...</span>
            `;
            if (apiConnectionStatus) {
                apiConnectionStatus.textContent = 'Vérification en cours...';
            }
            checkBtn.disabled = true;

            const response = await chrome.runtime.sendMessage({ type: 'CHECK_API_STATUS' });

            if (response && response.apiStatus === 'online') {
                apiStatus.innerHTML = `
                    <div class="status-dot online"></div>
                    <span>API en ligne</span>
                `;
                if (apiConnectionStatus) {
                    apiConnectionStatus.textContent = 'Connexion établie avec l\'API ML';
                }
                this.stats.apiStatus = 'online';
            } else {
                apiStatus.innerHTML = `
                    <div class="status-dot offline"></div>
                    <span>API hors ligne</span>
                `;
                if (apiConnectionStatus) {
                    apiConnectionStatus.textContent = 'Impossible de joindre l\'API ML';
                }
                this.stats.apiStatus = 'offline';
            }

            this.updateStatsDisplay();
            checkBtn.disabled = false;

        } catch (error) {
            console.error('❌ Erreur vérification API:', error);

            const apiStatus = document.getElementById('apiStatus');
            apiStatus.innerHTML = `
                <div class="status-dot offline"></div>
                <span>Erreur</span>
            `;

            document.getElementById('checkApiBtn').disabled = false;
        }
    }

    // Actions utilisateur
    async reanalyzePage() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab) return;

            const response = await chrome.tabs.sendMessage(tab.id, { type: 'REANALYZE_PAGE' });

            if (response && response.status === 'reanalysis_started') {
                this.showToast('Re-analyse démarrée', 'success');

                // Mettre à jour l'interface immédiatement
                this.updateSiteStatusDisplay({ isAnalyzing: true });

                // Rafraîchir après un délai
                setTimeout(() => {
                    this.loadCurrentSiteStatus();
                }, 3000);
            }
        } catch (error) {
            console.error('❌ Erreur re-analyse:', error);
            this.showToast('Erreur lors de la re-analyse', 'error');
        }
    }

    openDashboard() {
        chrome.tabs.create({ url: this.apiUrl });
    }

    async reportFalsePositive() {
        if (!this.currentSiteData || !this.currentSiteData.lastResult) {
            this.showToast('Aucune analyse à signaler', 'warning');
            return;
        }

        try {
            // Envoyer un feedback négatif via l'API
            const feedbackData = {
                email_text: `Site: ${this.currentSiteData.url}`,
                predicted_class: this.currentSiteData.lastResult.prediction,
                predicted_probability: this.currentSiteData.lastResult.probability || 0.5,
                user_satisfaction: 'no',
                language_detected: this.currentSiteData.lastResult.language_detected || 'en'
            };

            const response = await fetch(`${this.apiUrl}/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(feedbackData)
            });

            if (response.ok) {
                const result = await response.json();

                if (result.auto_finetuning_triggered) {
                    this.showToast('Signalement envoyé ! 🧠 IA en cours d\'amélioration', 'success');
                } else {
                    this.showToast('Signalement envoyé, merci !', 'success');
                }

                // Masquer le bouton après signalement
                document.getElementById('reportFalsePositiveBtn').style.display = 'none';
            } else {
                throw new Error('Erreur serveur');
            }

        } catch (error) {
            console.error('❌ Erreur signalement:', error);
            this.showToast('Erreur lors du signalement', 'error');
        }
    }

    async clearHistory() {
        if (confirm('Êtes-vous sûr de vouloir effacer l\'historique ?')) {
            try {
                await chrome.runtime.sendMessage({ type: 'CLEAR_HISTORY' });
                this.loadHistory();
                this.loadStats();
                this.showToast('Historique effacé', 'success');
            } catch (error) {
                console.error('❌ Erreur effacement historique:', error);
                this.showToast('Erreur lors de l\'effacement', 'error');
            }
        }
    }

    onSettingChange() {
        // Auto-save des paramètres quand ils changent
        setTimeout(() => this.saveSettings(), 500);
    }

    async saveSettings() {
        try {
            const newSettings = {
                enableNotifications: document.getElementById('enableNotifications').checked,
                enableAutoAnalysis: document.getElementById('enableAutoAnalysis').checked,
                showSafeIndicator: document.getElementById('showSafeIndicator').checked,
                confidenceThreshold: parseInt(document.getElementById('confidenceThreshold').value)
            };

            await chrome.runtime.sendMessage({
                type: 'UPDATE_SETTINGS',
                settings: newSettings
            });

            this.settings = newSettings;

            // Feedback visuel
            const btn = document.getElementById('saveSettingsBtn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<span class="btn-icon">✅</span>Sauvegardé';

            setTimeout(() => {
                btn.innerHTML = originalText;
            }, 1500);

        } catch (error) {
            console.error('❌ Erreur sauvegarde paramètres:', error);
            this.showToast('Erreur de sauvegarde', 'error');
        }
    }

    // Navigation et UI
    switchTab(tabName) {
        // Mettre à jour les boutons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });

        // Mettre à jour le contenu
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `tab-${tabName}`);
        });

        this.currentTab = tabName;

        // Charger les données spécifiques à l'onglet
        if (tabName === 'history') {
            this.loadHistory();
        } else if (tabName === 'settings') {
            this.loadSettings();
        } else if (tabName === 'status') {
            this.loadCurrentSiteStatus();
            this.loadStats();
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        // Auto-remove après 3 secondes
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 3000);
    }

    // Utilitaires
    extractDomain(url) {
        try {
            return new URL(url).hostname;
        } catch {
            return url.substring(0, 30) + '...';
        }
    }

    formatTime(timestamp) {
        try {
            const date = new Date(timestamp);
            const now = new Date();
            const diff = now - date;

            if (diff < 60000) { // < 1 minute
                return 'À l\'instant';
            } else if (diff < 3600000) { // < 1 heure
                return Math.floor(diff / 60000) + ' min';
            } else if (diff < 86400000) { // < 1 jour
                return Math.floor(diff / 3600000) + ' h';
            } else {
                return date.toLocaleDateString();
            }
        } catch {
            return 'Date inconnue';
        }
    }
}

// Initialiser le gestionnaire quand le DOM est prêt
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 DOM chargé, initialisation du popup...');
    window.popupManager = new PopupManager();
});

// Gérer les erreurs non capturées
window.addEventListener('error', (event) => {
    console.error('❌ Erreur popup:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('❌ Promise rejetée popup:', event.reason);
});