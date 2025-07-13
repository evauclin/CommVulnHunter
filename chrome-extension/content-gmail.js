// content-gmail.js - VERSION UNIQUE ET FINALE
console.log('🛡️ GUARDIAN UNIQUE - Démarrage sécurisé');
console.log('📄 Fichier:', 'content-gmail-unique.js');
console.log('⏰ Heure:', new Date().toLocaleTimeString());

// Vérifier qu'on est le seul script Guardian
if (window.guardianLoaded) {
    console.warn('⚠️ Guardian déjà chargé - arrêt pour éviter les doublons');
    throw new Error('Guardian already loaded');
}

window.guardianLoaded = true;

// Fonction de vérification ultra-simple
function isExtensionOK() {
    try {
        return !!(chrome && chrome.runtime && chrome.runtime.id);
    } catch (e) {
        return false;
    }
}

// Envoi de message ultra-sécurisé - JAMAIS de crash
function sendSafeMessage(message) {
    if (!isExtensionOK()) {
        console.log('📦 Message stocké (contexte invalide)');
        return;
    }

    try {
        chrome.runtime.sendMessage(message, function(response) {
            // Ignorer toutes les erreurs - pas de crash possible
            if (chrome.runtime.lastError) {
                console.log('📡 Message envoyé (avec erreur ignorée)');
            } else {
                console.log('✅ Message envoyé avec succès');
            }
        });
    } catch (error) {
        console.log('📡 Exception sendMessage ignorée');
    }
}

class UniqueGmailScanner {
    constructor() {
        this.scannedEmails = new Set();
        this.isActive = true;
        this.scanCount = 0;
    }

    init() {
        console.log("🔄 Init scanner unique...");

        // Attendre que Gmail soit chargé
        this.waitForGmail(() => {
            console.log("✅ Gmail OK - Démarrage scan");
            this.startSimpleScan();
        });
    }

    waitForGmail(callback) {
        let attempts = 0;

        const check = () => {
            attempts++;
            console.log(`🔍 Tentative ${attempts} - Recherche Gmail...`);

            if (attempts > 20) {
                console.log('❌ Gmail non détecté après 20 tentatives');
                return;
            }

            try {
                if (document.querySelector('.nH, [role="navigation"], [gh="tl"]')) {
                    console.log('✅ Gmail détecté !');
                    callback();
                } else {
                    setTimeout(check, 2000);
                }
            } catch (e) {
                setTimeout(check, 2000);
            }
        };

        setTimeout(check, 3000); // Attendre 3s au démarrage
    }

    startSimpleScan() {
        console.log("🚀 Scan simple démarré");

        // Scan immédiat
        this.performScan();

        // Scan périodique espacé - pas d'observer pour éviter les problèmes
        setInterval(() => {
            if (this.isActive && isExtensionOK()) {
                this.performScan();
            } else {
                console.log('🛑 Scanner arrêté (contexte invalide)');
                this.isActive = false;
            }
        }, 15000); // 15 secondes entre chaque scan
    }

    performScan() {
        if (!this.isActive) return;

        this.scanCount++;
        console.log(`📧 Scan ${this.scanCount} - Recherche emails...`);

        try {
            // Sélecteurs Gmail simples et fiables
            const emailRows = document.querySelectorAll('tr.zA, div[role="listitem"]');
            console.log(`📬 ${emailRows.length} lignes d'emails trouvées`);

            let newEmails = 0;

            emailRows.forEach((row, index) => {
                if (index > 10) return; // Limiter à 10 emails max

                try {
                    const emailData = this.extractSimpleData(row);
                    if (emailData) {
                        const key = `${emailData.sender}_${emailData.subject}`;

                        if (!this.scannedEmails.has(key)) {
                            this.scannedEmails.add(key);
                            newEmails++;

                            console.log(`📨 Nouvel email ${newEmails}:`, emailData.sender);

                            sendSafeMessage({
                                type: 'NEW_GMAIL_EMAIL_DETECTED',
                                data: emailData
                            });
                        }
                    }
                } catch (e) {
                    // Ignorer les erreurs d'extraction
                }
            });

            if (newEmails === 0) {
                console.log('📭 Aucun nouvel email');
            } else {
                console.log(`📧 ${newEmails} nouveaux emails traités`);
            }

        } catch (scanError) {
            console.log('⚠️ Erreur scan (ignorée):', scanError.message);
        }
    }

    extractSimpleData(row) {
        try {
            // Extraction simple et robuste
            let sender = 'Inconnu';
            let subject = 'Sans sujet';

            // Chercher l'expéditeur
            const senderEl = row.querySelector('.yW span, .go span, [email]');
            if (senderEl) {
                sender = senderEl.getAttribute('email') ||
                        senderEl.getAttribute('name') ||
                        senderEl.textContent?.trim() ||
                        'Inconnu';
            }

            // Chercher le sujet
            const subjectEl = row.querySelector('.bog, .bqe, .y6 span');
            if (subjectEl && subjectEl.textContent) {
                subject = subjectEl.textContent.trim() || 'Sans sujet';
            }

            // ID simple
            const id = row.getAttribute('data-thread-id') ||
                      `email_${Date.now()}_${Math.random().toString(36).substr(2, 3)}`;

            return {
                id: id,
                sender: sender,
                subject: subject,
                timestamp: Date.now()
            };

        } catch (e) {
            return null;
        }
    }
}

// Démarrage unique et sécurisé
let scanner = null;

function startUniqueScanner() {
    if (scanner) {
        console.log('🔄 Scanner déjà actif');
        return;
    }

    try {
        scanner = new UniqueGmailScanner();
        scanner.init();
    } catch (error) {
        console.error('❌ Erreur init scanner:', error);
    }
}

// Démarrage après chargement complet du DOM
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(startUniqueScanner, 2000);
    });
} else {
    setTimeout(startUniqueScanner, 2000);
}

// Nettoyage à la fermeture
window.addEventListener('beforeunload', () => {
    if (scanner) {
        scanner.isActive = false;
    }
    window.guardianLoaded = false;
});

console.log("✅ Guardian UNIQUE chargé - Anti-doublon garanti");