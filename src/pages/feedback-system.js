class FeedbackSystem {
    constructor(apiUrl = 'http://localhost:8000') {
        this.apiUrl = apiUrl;
        this.lastPrediction = null;
        this.currentEmailData = null;
        this.feedbackStats = {
            total: 0,
            accuracy: 100
        };

        this.init();
    }

    init() {
        this.loadFeedbackStats();
        this.setupEventListeners();
        console.log('💬 Système de feedback initialisé');
    }

    setupEventListeners() {
        // Écouter les changements sur les boutons de satisfaction
        document.querySelectorAll('input[name="satisfaction"]').forEach(input => {
            input.addEventListener('change', (e) => {
                this.handleFeedback(e.target.value);
            });
        });

        // Bouton pour voir les statistiques de feedback
        this.addFeedbackStatsButton();
    }

    // 🆕 Méthode appelée après chaque prédiction ML
    storePredictionResult(emailData, predictionResult) {
        this.lastPrediction = predictionResult;
        this.currentEmailData = {
            text: `From: ${emailData.from} Subject: ${emailData.subject} Body: ${emailData.body}`,
            from: emailData.from,
            subject: emailData.subject,
            body: emailData.body,
            id: emailData.id
        };

        console.log('📊 Prédiction stockée pour feedback:', this.lastPrediction);

        // Réactiver les boutons de feedback
        this.enableFeedbackButtons();
    }

    enableFeedbackButtons() {
        document.querySelectorAll('input[name="satisfaction"]').forEach(input => {
            input.disabled = false;
            input.checked = false;
        });

        // Ajouter un indicateur visuel
        const feedbackSection = document.querySelector('.analyze-section .mt-3.p-3');
        if (feedbackSection && !feedbackSection.querySelector('.feedback-indicator')) {
            const indicator = document.createElement('div');
            indicator.className = 'feedback-indicator text-center mb-2';
            indicator.innerHTML = '<small class="text-light"><i class="bi bi-arrow-down"></i> Votre avis nous aide à améliorer le modèle</small>';
            feedbackSection.insertBefore(indicator, feedbackSection.firstChild);
        }
    }

    // 🚀 Gérer le feedback utilisateur
    async handleFeedback(satisfaction) {
        if (!this.lastPrediction || !this.currentEmailData) {
            console.warn('⚠️ Aucune prédiction disponible pour le feedback');
            this.showFeedbackMessage('Veuillez d\'abord analyser un email', 'warning');
            return;
        }

        // Désactiver temporairement les boutons
        document.querySelectorAll('input[name="satisfaction"]').forEach(input => {
            input.disabled = true;
        });

        try {
            // Préparer les données de feedback
            const feedbackData = {
                email_text: this.currentEmailData.text,
                predicted_label: this.lastPrediction.prediction,
                user_satisfaction: satisfaction,
                confidence_score: this.lastPrediction.probability,
                email_id: this.currentEmailData.id
            };

            console.log('📤 Envoi du feedback:', feedbackData);

            // Envoyer le feedback à l'API
            const response = await fetch(`${this.apiUrl}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedbackData)
            });

            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }

            const result = await response.json();
            console.log('✅ Feedback envoyé avec succès:', result);

            // Afficher le message de confirmation
            this.showFeedbackMessage(
                `Merci pour votre retour ! ${satisfaction === 'yes' ? '👍' : '👎'} (Total: ${result.feedback_id || '?'} feedbacks)`,
                satisfaction === 'yes' ? 'success' : 'info'
            );

            // Mettre à jour les statistiques
            await this.loadFeedbackStats();

            // Marquer le feedback comme envoyé
            this.markFeedbackSent(satisfaction);

        } catch (error) {
            console.error('❌ Erreur envoi feedback:', error);
            this.showFeedbackMessage('Erreur lors de l\'envoi du feedback', 'danger');

            // Réactiver les boutons en cas d'erreur
            document.querySelectorAll('input[name="satisfaction"]').forEach(input => {
                input.disabled = false;
            });
        }
    }

    markFeedbackSent(satisfaction) {
        // Marquer visuellement que le feedback a été envoyé
        const feedbackSection = document.querySelector('.analyze-section .mt-3.p-3');
        if (feedbackSection) {
            feedbackSection.style.opacity = '0.7';

            // Ajouter une icône de confirmation
            const selectedInput = document.querySelector(`input[name="satisfaction"][value="${satisfaction}"]`);
            if (selectedInput) {
                const label = selectedInput.nextElementSibling;
                if (label && !label.querySelector('.feedback-sent')) {
                    const checkmark = document.createElement('span');
                    checkmark.className = 'feedback-sent ms-1';
                    checkmark.innerHTML = '<i class="bi bi-check-circle-fill text-success"></i>';
                    label.appendChild(checkmark);
                }
            }
        }

        // Désactiver définitivement les boutons pour cette prédiction
        setTimeout(() => {
            document.querySelectorAll('input[name="satisfaction"]').forEach(input => {
                input.disabled = true;
            });
        }, 2000);
    }

    showFeedbackMessage(message, type = 'info') {
        // Afficher le message dans la section d'analyse
        const analysisResult = document.getElementById('analysisResult');
        if (analysisResult) {
            const alertClass = {
                'success': 'text-success',
                'info': 'text-info',
                'warning': 'text-warning',
                'danger': 'text-danger'
            };

            const feedbackHtml = `
                <div class="feedback-message mt-2 p-2" style="background: rgba(255,255,255,0.1); border-radius: 5px;">
                    <small class="${alertClass[type] || 'text-info'}">
                        <i class="bi bi-chat-dots me-1"></i>
                        ${message}
                    </small>
                </div>
            `;

            // Ajouter le message ou remplacer l'existant
            const existingMessage = analysisResult.querySelector('.feedback-message');
            if (existingMessage) {
                existingMessage.outerHTML = feedbackHtml;
            } else {
                analysisResult.innerHTML += feedbackHtml;
            }

            // Supprimer le message après 5 secondes
            setTimeout(() => {
                const messageElement = analysisResult.querySelector('.feedback-message');
                if (messageElement) {
                    messageElement.remove();
                }
            }, 5000);
        }
    }

    // 📊 Charger les statistiques de feedback
    async loadFeedbackStats() {
        try {
            const response = await fetch(`${this.apiUrl}/feedback/stats`);
            if (response.ok) {
                const data = await response.json();
                this.feedbackStats = data.feedback_statistics;
                this.updateStatsDisplay(data);
                console.log('📈 Stats feedback chargées:', this.feedbackStats);
            }
        } catch (error) {
            console.warn('⚠️ Impossible de charger les stats feedback:', error);
        }
    }

    updateStatsDisplay(data) {
        // Mettre à jour l'affichage des stats dans l'interface
        const stats = data.feedback_statistics;

        // Ajouter les stats de feedback dans le dashboard si pas déjà présent
        this.addFeedbackStatsToUI(stats);

        // Mettre à jour le badge de précision
        const successRate = document.getElementById('successRate');
        if (successRate && stats.total_feedbacks > 0) {
            successRate.textContent = `${stats.accuracy_rate}%`;

            // Changer la couleur selon la précision
            const card = successRate.closest('.stats-card');
            if (card) {
                if (stats.accuracy_rate >= 90) {
                    card.className = 'stats-card stats-success';
                } else if (stats.accuracy_rate >= 80) {
                    card.className = 'stats-card stats-warning';
                } else {
                    card.className = 'stats-card stats-danger';
                }
            }
        }
    }

    addFeedbackStatsToUI(stats) {
        // Vérifier si les stats feedback sont déjà affichées
        if (document.getElementById('feedbackStats')) return;

        // Trouver l'emplacement pour ajouter les stats
        const statsRow = document.querySelector('.row.mb-4');
        if (statsRow && stats.total_feedbacks > 0) {
            const feedbackStatsHTML = `
                <div class="col-md-3" id="feedbackStats">
                    <div class="stats-card" style="background: linear-gradient(135deg, #667eea, #764ba2);">
                        <div class="display-6">${stats.total_feedbacks}</div>
                        <h6>Feedbacks reçus</h6>
                        <small>Précision: ${stats.accuracy_rate}%</small>
                    </div>
                </div>
            `;

            // Ajouter après les stats existantes
            statsRow.insertAdjacentHTML('beforeend', feedbackStatsHTML);
        }
    }

    addFeedbackStatsButton() {
        // Ajouter un bouton pour voir les détails des feedbacks
        const btnGroup = document.querySelector('.btn-group[role="group"]');
        if (btnGroup && !document.getElementById('feedbackStatsBtn')) {
            const feedbackBtn = document.createElement('button');
            feedbackBtn.id = 'feedbackStatsBtn';
            feedbackBtn.type = 'button';
            feedbackBtn.className = 'btn btn-outline-info';
            feedbackBtn.innerHTML = '<i class="bi bi-graph-up"></i> Feedback';
            feedbackBtn.onclick = () => this.showFeedbackDashboard();

            btnGroup.appendChild(feedbackBtn);
        }
    }

    async showFeedbackDashboard() {
        try {
            const response = await fetch(`${this.apiUrl}/feedback/stats`);
            const data = await response.json();

            this.displayFeedbackModal(data);

        } catch (error) {
            console.error('Erreur chargement dashboard feedback:', error);
            alert('Impossible de charger les statistiques de feedback');
        }
    }

    displayFeedbackModal(data) {
        const stats = data.feedback_statistics;
        const history = data.retraining_history;

        // Créer une modale pour afficher les stats détaillées
        const modalHTML = `
            <div class="modal fade" id="feedbackModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">
                                <i class="bi bi-graph-up text-primary"></i>
                                Dashboard Feedback & Réentraînement
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row mb-4">
                                <div class="col-md-3 text-center">
                                    <div class="badge bg-primary fs-4">${stats.total_feedbacks}</div>
                                    <div>Total Feedbacks</div>
                                </div>
                                <div class="col-md-3 text-center">
                                    <div class="badge bg-success fs-4">${stats.accuracy_rate}%</div>
                                    <div>Précision Modèle</div>
                                </div>
                                <div class="col-md-3 text-center">
                                    <div class="badge bg-warning fs-4">${stats.negative_feedbacks}</div>
                                    <div>Feedbacks Négatifs</div>
                                </div>
                                <div class="col-md-3 text-center">
                                    <div class="badge bg-info fs-4">${stats.unprocessed_feedbacks}</div>
                                    <div>Non Traités</div>
                                </div>
                            </div>
                            
                            <h6><i class="bi bi-arrow-repeat"></i> Historique des Réentraînements</h6>
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th>Date</th>
                                            <th>Feedbacks</th>
                                            <th>Précision Avant</th>
                                            <th>Précision Après</th>
                                            <th>Statut</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${history.map(h => `
                                            <tr>
                                                <td>${new Date(h.timestamp).toLocaleDateString()}</td>
                                                <td>${h.num_feedbacks}</td>
                                                <td>${(h.old_accuracy * 100).toFixed(1)}%</td>
                                                <td>${(h.new_accuracy * 100).toFixed(1)}%</td>
                                                <td>
                                                    <span class="badge ${h.status === 'SUCCESS' ? 'bg-success' : 'bg-danger'}">
                                                        ${h.status}
                                                    </span>
                                                </td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                            
                            <div class="alert alert-info mt-3">
                                <i class="bi bi-info-circle"></i>
                                <strong>Prochain réentraînement:</strong> ${data.next_retraining_at}
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Fermer</button>
                            <button type="button" class="btn btn-primary" onclick="feedbackSystem.triggerRetraining()">
                                <i class="bi bi-arrow-repeat"></i> Forcer Réentraînement
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Supprimer l'ancienne modale si elle existe
        const existingModal = document.getElementById('feedbackModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Ajouter la nouvelle modale
        document.body.insertAdjacentHTML('beforeend', modalHTML);

        // Afficher la modale
        const modal = new bootstrap.Modal(document.getElementById('feedbackModal'));
        modal.show();
    }

    async triggerRetraining() {
        if (!confirm('Êtes-vous sûr de vouloir déclencher un réentraînement du modèle ?')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiUrl}/feedback/trigger-retraining`, {
                method: 'POST'
            });

            if (response.ok) {
                const result = await response.json();
                alert('Réentraînement déclenché avec succès ! Le processus se déroule en arrière-plan.');

                // Fermer la modale
                const modal = bootstrap.Modal.getInstance(document.getElementById('feedbackModal'));
                if (modal) modal.hide();

            } else {
                throw new Error('Erreur lors du déclenchement');
            }

        } catch (error) {
            console.error('Erreur déclenchement réentraînement:', error);
            alert('Erreur lors du déclenchement du réentraînement');
        }
    }

    // 🧹 Reset pour nouvelle analyse
    resetForNewAnalysis() {
        // Reset l'état pour une nouvelle analyse
        this.lastPrediction = null;
        this.currentEmailData = null;

        // Reset les boutons de feedback
        document.querySelectorAll('input[name="satisfaction"]').forEach(input => {
            input.disabled = true;
            input.checked = false;
        });

        // Supprimer les indicateurs visuels
        const feedbackIndicator = document.querySelector('.feedback-indicator');
        if (feedbackIndicator) {
            feedbackIndicator.remove();
        }

        const feedbackSent = document.querySelectorAll('.feedback-sent');
        feedbackSent.forEach(element => element.remove());

        // Reset l'opacité
        const feedbackSection = document.querySelector('.analyze-section .mt-3.p-3');
        if (feedbackSection) {
            feedbackSection.style.opacity = '1';
        }

        console.log('🧹 Feedback system reset for new analysis');
    }
}

// 🚀 Initialisation globale
let feedbackSystem;

// Modifications des fonctions existantes pour intégrer le feedback

// Modifier la fonction analyzeEmail existante
const originalAnalyzeEmail = window.analyzeEmail;
window.analyzeEmail = async function() {
    // Reset le système de feedback avant nouvelle analyse
    if (feedbackSystem) {
        feedbackSystem.resetForNewAnalysis();
    }

    // Appeler la fonction d'analyse originale (si elle existe)
    if (originalAnalyzeEmail) {
        return await originalAnalyzeEmail();
    }

    // Sinon, utiliser la logique d'analyse standard
    if (!mlStatusOnline) {
        showAnalysisError('Le modèle ML n\'est pas disponible.');
        return;
    }

    const resultDiv = document.getElementById('analysisResult');
    const analyzeBtn = document.getElementById('analyzeBtn');

    const emailFrom = document.getElementById('emailFrom').textContent;
    const emailSubject = document.getElementById('emailSubject').textContent;
    const emailBody = document.getElementById('emailBody').textContent;

    if (emailFrom === 'Sélectionnez un email') {
        showAnalysisError('Veuillez sélectionner un email complet.');
        return;
    }

    const rawEmailText = `From: ${emailFrom} Subject: ${emailSubject} Body: ${emailBody}`;

    analyzeBtn.disabled = true;
    resultDiv.innerHTML = `<div class="text-info"><div class="spinner-border spinner-border-sm me-2"></div><strong>Analyse IA en cours...</strong></div>`;

    try {
        const response = await fetch(`${mlApiUrl}/predict`, {
            method: 'POST',
            mode: 'cors',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: cleanText(rawEmailText) }),
            signal: AbortSignal.timeout(30000)
        });

        if (!response.ok) throw new Error(`API Error ${response.status}: ${await response.text()}`);

        const mlResult = await response.json();
        displayMLResult(mlResult);

        // 🆕 Stocker le résultat pour le système de feedback
        if (feedbackSystem) {
            const emailData = {
                from: emailFrom,
                subject: emailSubject,
                body: emailBody,
                id: document.getElementById('emailId').textContent
            };

            feedbackSystem.storePredictionResult(emailData, mlResult);
        }

    } catch (error) {
        showAnalysisError(`Erreur ML: ${error.message}`);
        checkMLStatus();
    } finally {
        analyzeBtn.disabled = !mlStatusOnline;
    }
};

// Modifier la fonction selectEmail pour reset le feedback
const originalSelectEmail = window.selectEmail;
window.selectEmail = function(index) {
    // Reset le feedback system
    if (feedbackSystem) {
        feedbackSystem.resetForNewAnalysis();
    }

    // Appeler la fonction originale
    if (originalSelectEmail) {
        return originalSelectEmail(index);
    }

    // Logique selectEmail par défaut si nécessaire...
};

// 🎯 Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    // Attendre que les autres systèmes soient initialisés
    setTimeout(() => {
        feedbackSystem = new FeedbackSystem(mlApiUrl || 'http://localhost:8000');
        console.log('✅ Système de feedback intégré au dashboard');
    }, 1000);
});

// Export pour utilisation dans d'autres scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FeedbackSystem };
}