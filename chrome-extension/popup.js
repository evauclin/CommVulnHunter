// popup.js - Gère l'affichage du popup.

async function sendMessage(message) {
    try {
        if (!chrome.runtime?.id) return null;
        return await chrome.runtime.sendMessage(message);
    } catch (e) {
        console.error(`Erreur de communication: ${e.message}`);
        return null;
    }
}

function displayHistory(history) {
    const listElement = document.getElementById('historyList');
    if (!history || history.length === 0) {
        listElement.innerHTML = '<p class="empty-state">Aucune notification enregistrée.</p>';
        return;
    }

    // --- CORRECTION DE LA FONCTION ---
    const escapeHTML = (str = '') =>
        str.replace(/[&<>"']/g, m => ({
            '&': '&',
            '<': '<',
            '>': '>',
            '"': '"',
            "'": "'"
        }[m]));
    // ---------------------------------

    listElement.innerHTML = history.map(item => `
        <div class="history-item ${item.isPhishing ? 'phishing' : 'safe'}">
            <div class="item-header">
                <span class="item-origin">${escapeHTML(item.origin)}</span>
                <span class="item-time">${new Date(item.timestamp).toLocaleTimeString()}</span>
            </div>
            <p class="item-title">${escapeHTML(item.title)}</p>
            <p class="item-body">${escapeHTML(item.body)}</p>
        </div>
    `).join('');
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-content, .tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    if (tabName === 'history') {
        sendMessage({ type: 'GET_HISTORY' }).then(response => {
            if (response) displayHistory(response);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
    document.getElementById('clearHistoryBtn').addEventListener('click', async () => {
        if (confirm("Voulez-vous vraiment effacer tout l'historique ?")) {
            await sendMessage({ type: 'CLEAR_HISTORY' });
            displayHistory([]);
        }
    });
    switchTab('status');
});