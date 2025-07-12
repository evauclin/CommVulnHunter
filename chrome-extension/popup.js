// popup.js - Interface utilisateur pour Phishing Guardian
console.log('[GUARDIAN POPUP] 🎨 Interface démarrée');

// === ÉLÉMENTS DOM ===
const statusCard = document.getElementById('status-card');
const statusIcon = document.getElementById('status-icon');
const statusText = document.getElementById('status-text');
const statusDetails = document.getElementById('status-details');

const lastScanCard = document.getElementById('last-scan');
const scanSubject = lastScanCard.querySelector('.scan-subject');
const scanSender = lastScanCard.querySelector('.scan-sender');
const scanTime = lastScanCard.querySelector('.scan-time');
const scanStatus = document.getElementById('scan-status');

const totalScanned = document.getElementById('total-scanned');
const safeEmails = document.getElementById('safe-emails');
const phishingDetected = document.getElementById('phishing-detected');

const testBtn = document.getElementById('test-notification');
const clearBtn = document.getElementById('clear-data');

const apiIndicator = document.getElementById('api-indicator');
const apiText = document.getElementById('api-text');

// === GESTION DU STATUT ===
function updateStatus(type, title, details) {
    // Réinitialiser les classes
    statusCard.className = 'status-card';

    // Ajouter la nouvelle classe
    statusCard.classList.add(type);

    // Mettre à jour le contenu
    switch (type) {
        case 'waiting':
            statusIcon.textContent = '⏱️';
            break;
        case 'safe':
            statusIcon.textContent = '✅';
            break;
        case 'phishing':
            statusIcon.textContent = '🚨';
            break;
        case 'error':
            statusIcon.textContent = '⚠️';
            break;
        default:
            statusIcon.textContent = '❓';
    }

    statusText.textContent = title;
    statusDetails.textContent = details;

    console.log(`[GUARDIAN POPUP] 📊 Statut mis à jour: ${type} - ${title}`);
}

// === GESTION DU DERNIER SCAN ===
function updateLastScan(scanData) {
    if (!scanData) {
        scanSubject.textContent = 'Aucune analyse récente';
        scanSender.textContent = '-';
        scanTime.textContent = '-';
        scanStatus.className = 'scan-status unknown';
        scanStatus.textContent = '?';
        return;
    }

    // Contenu
    scanSubject.textContent = scanData.subject || 'Sujet non disponible';
    scanSender.textContent = `De: ${scanData.sender || 'Expéditeur inconnu'}`;

    // Temps
    if (scanData.timestamp) {
        const date = new Date(scanData.timestamp);
        const now = new Date();
        const diffMinutes = Math.floor((now - date) / (1000 * 60));

        if (diffMinutes < 1) {
            scanTime.textContent = 'À l\'instant';
        } else if (diffMinutes < 60) {
            scanTime.textContent = `Il y a ${diffMinutes} min`;
        } else {
            const diffHours = Math.floor(diffMinutes / 60);
            scanTime.textContent = `Il y a ${diffHours}h`;
        }
    } else {
        scanTime.textContent = 'Heure inconnue';
    }

    // Statut de sécurité
    if (scanData.error) {
        scanStatus.className = 'scan-status unknown';
        scanStatus.textContent = '!';
        updateStatus('error', 'Erreur d\'Analyse', scanData.error);
    } else if (scanData.isPhishing) {
        scanStatus.className = 'scan-status danger';
        scanStatus.textContent = '⚠';
        updateStatus('phishing', 'Phishing Détecté !', `Mail suspect analysé`);
    } else {
        scanStatus.className = 'scan-status safe';
        scanStatus.textContent = '✓';
        updateStatus('safe', 'Mail Analysé - Sûr', `Confiance: ${Math.round((scanData.confidence || 0) * 100)}%`);
    }

    console.log('[GUARDIAN POPUP] 📧 Dernier scan mis à jour:', scanData.subject);
}

// === GESTION DES STATISTIQUES ===
function updateStatistics(stats) {
    if (!stats) {
        stats = { totalScanned: 0, safeEmails: 0, phishingDetected: 0 };
    }

    // Animation des compteurs
    animateCounter(totalScanned, parseInt(totalScanned.textContent) || 0, stats.totalScanned || 0);
    animateCounter(safeEmails, parseInt(safeEmails.textContent) || 0, stats.safeEmails || 0);
    animateCounter(phishingDetected, parseInt(phishingDetected.textContent) || 0, stats.phishingDetected || 0);

    console.log('[GUARDIAN POPUP] 📈 Statistiques mises à jour:', stats);
}

// Animation des compteurs
function animateCounter(element, start, end) {
    const duration = 500;
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current);
    }, 16);
}

// === TEST DE L'API ===
async function testAPI() {
    console.log('[GUARDIAN POPUP] 🧪 Test de l\'API...');

    apiIndicator.className = 'api-indicator';
    apiText.textContent = 'API: Test en cours...';

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: 'Test de connectivité' })
        });

        if (response.ok) {
            apiIndicator.classList.add('online');
            apiText.textContent = 'API: ✅ Connectée';
            console.log('[GUARDIAN POPUP] ✅ API accessible');
        } else {
            throw new Error(`Status ${response.status}`);
        }

    } catch (error) {
        apiIndicator.classList.add('offline');
        apiText.textContent = 'API: ❌ Déconnectée';
        console.log('[GUARDIAN POPUP] ❌ API inaccessible:', error.message);
    }
}

// === CHARGEMENT DES DONNÉES ===
async function loadData() {
    console.log('[GUARDIAN POPUP] 📂 Chargement des données...');

    try {
        // Demander les données au background
        const response = await chrome.runtime.sendMessage({ type: 'STATUS_REQUEST' });

        console.log('[GUARDIAN POPUP] 📦 Données reçues:', response);

        // Mettre à jour l'interface
        if (response.lastScan) {
            updateLastScan(response.lastScan);
        } else {
            updateStatus('waiting', 'En Attente', 'Aucun mail analysé récemment');
        }

        if (response.statistics) {
            updateStatistics(response.statistics);
        }

    } catch (error) {
        console.error('[GUARDIAN POPUP] ❌ Erreur chargement:', error);
        updateStatus('error', 'Erreur de Communication', 'Impossible de contacter le service');
    }
}

// === ÉVÉNEMENTS ===

// Test de notification
testBtn.addEventListener('click', async () => {
    console.log('[GUARDIAN POPUP] 🧪 Test de notification demandé');

    testBtn.textContent = '⏳ Test...';
    testBtn.disabled = true;

    try {
        // Envoyer un mail de test pour analyse
        await chrome.runtime.sendMessage({
            type: 'NEW_MAIL_DETECTED',
            data: {
                subject: 'Test Guardian - Mail de démonstration',
                sender: 'test@guardian-demo.com',
                preview: 'Ceci est un test du système de détection Guardian',
                source: 'Test manuel'
            }
        });

        // Recharger les données après 2 secondes
        setTimeout(() => {
            loadData();
            testBtn.textContent = '🧪 Tester Notification';
            testBtn.disabled = false;
        }, 2000);

    } catch (error) {
        console.error('[GUARDIAN POPUP] ❌ Erreur test:', error);
        testBtn.textContent = '❌ Erreur';
        setTimeout(() => {
            testBtn.textContent = '🧪 Tester Notification';
            testBtn.disabled = false;
        }, 2000);
    }
});

// Effacer les données
clearBtn.addEventListener('click', async () => {
    if (!confirm('Effacer toutes les données de Guardian ?')) return;

    console.log('[GUARDIAN POPUP] 🗑️ Effacement des données...');

    clearBtn.textContent = '⏳ Effacement...';
    clearBtn.disabled = true;

    try {
        await chrome.storage.local.clear();

        // Réinitialiser l'interface
        updateStatus('waiting', 'Données Effacées', 'En attente de nouveaux mails');
        updateLastScan(null);
        updateStatistics(null);

        clearBtn.textContent = '✅ Effacé';
        setTimeout(() => {
            clearBtn.textContent = '🗑️ Effacer Données';
            clearBtn.disabled = false;
        }, 2000);

    } catch (error) {
        console.error('[GUARDIAN POPUP] ❌ Erreur effacement:', error);
        clearBtn.textContent = '❌ Erreur';
        setTimeout(() => {
            clearBtn.textContent = '🗑️ Effacer Données';
            clearBtn.disabled = false;
        }, 2000);
    }
});

// === ACTUALISATION AUTOMATIQUE ===
// Écouter les changements de stockage
chrome.storage.onChanged.addListener((changes, namespace) => {
    if (namespace === 'local') {
        console.log('[GUARDIAN POPUP] 🔄 Données mises à jour, rechargement...');

        if (changes.lastScan) {
            updateLastScan(changes.lastScan.newValue);
        }

        if (changes.statistics) {
            updateStatistics(changes.statistics.newValue);
        }
    }
});

// === INITIALISATION ===
document.addEventListener('DOMContentLoaded', async () => {
    console.log('[GUARDIAN POPUP] 🚀 Initialisation de l\'interface...');

    // Afficher l'état initial
    updateStatus('waiting', 'Chargement...', 'Récupération des données en cours');

    // Charger les données
    await loadData();

    // Tester l'API
    await testAPI();

    console.log('[GUARDIAN POPUP] ✅ Interface initialisée');
});

// === GESTION DES ONGLETS ===
// Détecter si on est sur Gmail/Yahoo
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs[0];
    if (tab && tab.url) {
        const isMailSite = tab.url.includes('mail.google.com') || tab.url.includes('mail.yahoo.com');

        if (isMailSite) {
            statusDetails.textContent += ' - Surveillance active sur cette page';
        } else {
            statusDetails.textContent += ' - Ouvrez Gmail ou Yahoo pour activer la surveillance';
        }
    }
});