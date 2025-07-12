// popup.js
document.addEventListener('DOMContentLoaded', () => {
    const statusCard = document.getElementById('status-card');
    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');
    const notificationContent = document.getElementById('notification-content');

    function updatePopup(scanResult) {
        if (!scanResult || !scanResult.last_scan) {
            statusCard.className = 'waiting';
            statusIcon.textContent = '⏱️';
            statusText.textContent = 'En attente de la prochaine notification...';
            notificationContent.style.display = 'none';
            return;
        }

        const lastScan = scanResult.last_scan;
        notificationContent.textContent = `Titre : "${lastScan.content}"`;
        notificationContent.style.display = 'block';

        if (lastScan.is_phishing) {
            statusCard.className = 'phishing';
            statusIcon.textContent = '⚠️';
            statusText.textContent = 'Phishing Détecté !';
        } else {
            statusCard.className = 'safe';
            statusIcon.textContent = '✅';
            statusText.textContent = 'Message Analysé et Sûr';
        }
    }

    chrome.storage.local.get(['last_scan'], updatePopup);
    chrome.storage.onChanged.addListener((changes, namespace) => {
        if (namespace === 'local' && changes.last_scan) {
            chrome.storage.local.get(['last_scan'], updatePopup);
        }
    });
});