// Affiche les données dès que le popup est ouvert
document.addEventListener('DOMContentLoaded', refreshUI);

// Met à jour l'interface quand le stockage change
chrome.storage.onChanged.addListener(refreshUI);

function refreshUI() {
    chrome.runtime.sendMessage({ type: 'STATUS_REQUEST' })
        .then(data => {
            if (data) {
                renderStats(data.statistics);
                renderHistory(data.scanHistory);
            }
        });
}

function renderStats(stats = { totalScanned: 0, phishingDetected: 0, safeEmails: 0 }) {
    document.getElementById('total-scanned').textContent = stats.totalScanned;
    document.getElementById('safe-emails').textContent = stats.safeEmails;
    document.getElementById('phishing-detected').textContent = stats.phishingDetected;
}

function renderHistory(history = []) {
    const list = document.getElementById('history-list');
    if (history.length === 0) {
        list.innerHTML = `<p class="empty-state">Aucune analyse récente.</p>`;
        return;
    }
    list.innerHTML = ''; // Vider la liste

    history.forEach(item => {
        const div = document.createElement('div');
        const statusClass = item.error ? 'error' : item.isPhishing ? 'phishing' : 'safe';
        div.className = `history-item ${statusClass}`;

        let feedbackHTML = `<div class="feedback-thanks">Merci pour votre retour !</div>`;
        if (!item.feedbackSent && !item.error) {
            feedbackHTML = `
                <div class="feedback" data-id="${item.id}" data-prediction="${item.isPhishing ? 'phishing' : 'safe'}">
                    <span>L'analyse est correcte ?</span>
                    <button class="feedback-btn" data-feedback="correct">Oui</button>
                    <button class="feedback-btn" data-feedback="incorrect">Non</button>
                </div>`;
        }

        div.innerHTML = `
            <div class="mail-subject">${item.subject.substring(0, 50)}</div>
            <div class="mail-sender">De: ${item.sender}</div>
            ${feedbackHTML}
        `;
        list.appendChild(div);
    });

    // Attacher les gestionnaires d'événements
    list.querySelectorAll('.feedback-btn').forEach(button => {
        button.addEventListener('click', handleFeedbackClick);
    });
}

function handleFeedbackClick(e) {
    const feedbackDiv = e.target.parentElement;
    const scanId = feedbackDiv.dataset.id;
    const feedback = e.target.dataset.feedback;

    chrome.runtime.sendMessage({
        type: 'SUBMIT_FEEDBACK',
        data: {
            scan_id: scanId,
            user_feedback: feedback,
            original_prediction: feedbackDiv.dataset.prediction,
            subject: e.target.closest('.history-item').querySelector('.mail-subject').textContent,
            sender: e.target.closest('.history-item').querySelector('.mail-sender').textContent.replace('De: ', '')
        }
    });

    feedbackDiv.innerHTML = `<div class="feedback-thanks">Merci !</div>`;
}