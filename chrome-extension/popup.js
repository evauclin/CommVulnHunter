// popup.js - Interface utilisateur pour Phishing Guardian
document.addEventListener('DOMContentLoaded', () => {
    console.log('[GUARDIAN POPUP] 🎨 Interface démarrée');

    const historyList = document.getElementById('history-list');
    const totalScannedEl = document.getElementById('total-scanned');
    const safeEmailsEl = document.getElementById('safe-emails');
    const phishingDetectedEl = document.getElementById('phishing-detected');
    const testBtn = document.getElementById('test-notification');
    const clearBtn = document.getElementById('clear-data');
    const apiIndicator = document.getElementById('api-indicator');
    const apiText = document.getElementById('api-text');

    // === NOUVEAU : Affichage de l'historique ===
    function renderHistory(history = []) {
        historyList.innerHTML = ''; // Vider la liste

        if (history.length === 0) {
            historyList.innerHTML = `
                <div class="history-item-placeholder">
                    <p>Aucune analyse récente.</p>
                    <p>La surveillance est active sur Gmail et Yahoo.</p>
                </div>`;
            return;
        }

        history.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'history-item';
            itemDiv.dataset.scanId = item.id;

            let statusClass = 'unknown';
            let statusIcon = '❓';
            if (item.error) {
                statusClass = 'error';
                statusIcon = '⚠️';
            } else if (item.isPhishing) {
                statusClass = 'phishing';
                statusIcon = '🚨';
            } else {
                statusClass = 'safe';
                statusIcon = '✅';
            }
            itemDiv.classList.add(statusClass);

            const timeAgo = formatTimeAgo(item.timestamp);

            itemDiv.innerHTML = `
                <div class="history-item-header">
                    <h4 class="scan-subject">${item.subject.substring(0, 40)}${item.subject.length > 40 ? '...' : ''}</h4>
                    <span class="scan-status-icon">${statusIcon}</span>
                </div>
                <p class="scan-sender">De: ${item.sender}</p>
                <p class="scan-time">${timeAgo}</p>
                <div class="feedback-actions">
                    ${item.feedbackSent ? 
                        `<p class="feedback-thanks">🙏 Merci pour votre retour !</p>` :
                        item.error ? 
                        `<p><i>Erreur d'analyse.</i></p>` :
                        `
                        <p>Cette prédiction était-elle correcte ?</p>
                        <button class="feedback-btn correct" data-feedback="correct">Oui</button>
                        <button class="feedback-btn incorrect" data-feedback="incorrect">Non</button>
                        `
                    }
                </div>
            `;
            historyList.appendChild(itemDiv);
        });

        // Attacher les écouteurs d'événements pour le feedback
        attachFeedbackListeners();
    }

    function attachFeedbackListeners() {
        document.querySelectorAll('.feedback-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const button = e.target;
                const itemDiv = button.closest('.history-item');
                const scanId = itemDiv.dataset.scanId;
                const feedbackType = button.dataset.feedback;

                // Trouver l'item dans les données locales pour envoyer toutes les infos
                chrome.storage.local.get('scanHistory', (result) => {
                    const scanItem = result.scanHistory.find(item => item.id === scanId);
                    if (!scanItem) return;

                    const originalPrediction = scanItem.isPhishing ? 'phishing' : 'safe';

                    // L'utilisateur est d'accord ("oui") si le feedback est "correct".
                    // L'utilisateur n'est pas d'accord ("non") si le feedback est "incorrect".
                    // On ne change pas la prédiction, on enregistre juste si l'utilisateur était d'accord.
                    const userFeedback = feedbackType; // 'correct' ou 'incorrect'

                    const feedbackPayload = {
                        scan_id: scanId,
                        user_feedback: userFeedback,
                        original_prediction: originalPrediction,
                        subject: scanItem.subject,
                        sender: scanItem.sender
                    };

                    // Envoyer le feedback via le background script
                    chrome.runtime.sendMessage({ type: 'SUBMIT_FEEDBACK', data: feedbackPayload });

                    // Mettre à jour l'UI instantanément
                    const feedbackContainer = button.closest('.feedback-actions');
                    feedbackContainer.innerHTML = `<p class="feedback-thanks">🙏 Merci pour votre retour !</p>`;
                });
            });
        });
    }


    // === GESTION DES STATISTIQUES ===
    function updateStatistics(stats) {
        if (!stats) stats = { totalScanned: 0, safeEmails: 0, phishingDetected: 0 };
        totalScannedEl.textContent = stats.totalScanned || 0;
        safeEmailsEl.textContent = stats.safeEmails || 0;
        phishingDetectedEl.textContent = stats.phishingDetected || 0;
    }

    // === CHARGEMENT INITIAL DES DONNÉES ===
    async function loadData() {
        const data = await chrome.runtime.sendMessage({ type: 'STATUS_REQUEST' });
        if (data) {
            renderHistory(data.scanHistory);
            updateStatistics(data.statistics);
        }
    }

    // === ÉVÉNEMENTS ===
    testBtn.addEventListener('click', () => {
        chrome.runtime.sendMessage({
            type: 'NEW_MAIL_DETECTED',
            data: {
                subject: `[TEST] ${Math.random() > 0.5 ? 'Gagnez un iPhone !' : 'Réunion importante'}`,
                sender: 'test@guardian-demo.com',
                preview: 'Ceci est un test du système de détection Guardian.'
            }
        });
    });

    clearBtn.addEventListener('click', () => {
        if (confirm('Effacer toutes les statistiques et l\'historique ?')) {
            chrome.storage.local.clear(() => {
                // Réinitialiser avec des valeurs par défaut
                chrome.storage.local.set({
                    statistics: { totalScanned: 0, phishingDetected: 0, safeEmails: 0 },
                    scanHistory: []
                }, () => {
                    loadData();
                    console.log('Données effacées et réinitialisées.');
                });
            });
        }
    });

    // === ACTUALISATION AUTOMATIQUE ===
    chrome.storage.onChanged.addListener((changes, namespace) => {
        if (namespace === 'local' && (changes.scanHistory || changes.statistics)) {
            console.log('[GUARDIAN POPUP] 🔄 Données mises à jour, rechargement...');
            loadData();
        }
    });

    // === Fonctions utilitaires ===
    function formatTimeAgo(timestamp) {
        const now = new Date();
        const past = new Date(timestamp);
        const diffInSeconds = Math.floor((now - past) / 1000);

        const minutes = Math.floor(diffInSeconds / 60);
        if (minutes < 1) return "À l'instant";
        if (minutes < 60) return `Il y a ${minutes} min`;

        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `Il y a ${hours}h`;

        const days = Math.floor(hours / 24);
        return `Il y a ${days}j`;
    }

    async function testAPI() {
        apiText.textContent = 'API: Test en cours...';
        try {
            const response = await fetch('http://localhost:8000/health');
            if (response.ok) {
                apiIndicator.classList.add('online');
                apiText.textContent = 'API: ✅ Connectée';
            } else { throw new Error(); }
        } catch (error) {
            apiIndicator.classList.add('offline');
            apiText.textContent = 'API: ❌ Déconnectée';
        }
    }

    // Initialisation
    loadData();
    testAPI();
});