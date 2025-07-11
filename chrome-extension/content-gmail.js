console.log('🛡️ AI Guardian (Gmail Scanner) activé.');

class GmailScanner {
    constructor() {
        this.scannedThreadIds = new Set();
    }

    init() {
        console.log("Initialisation du scanner pour Gmail (v2 - sélecteurs robustes)...");
        // Utiliser un MutationObserver est plus efficace que setInterval pour les changements d'UI
        const observer = new MutationObserver(() => this.scanForNewEmails());

        // Attendre que la boîte de réception principale soit chargée
        this.waitForElement('div[role="main"]', (main) => {
             observer.observe(main, { childList: true, subtree: true });
             // Lancer un premier scan au cas où les emails sont déjà là
             this.scanForNewEmails();
        });
    }

    /**
     * Attend qu'un élément correspondant au sélecteur apparaisse dans le DOM.
     */
    waitForElement(selector, callback) {
        const element = document.querySelector(selector);
        if (element) {
            callback(element);
        } else {
            const observer = new MutationObserver((mutations, obs) => {
                const foundElement = document.querySelector(selector);
                if (foundElement) {
                    obs.disconnect(); // Arrêter d'observer une fois l'élément trouvé
                    callback(foundElement);
                }
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }
    }

    /**
     * Scanne la boîte de réception pour trouver des emails non encore analysés.
     */
    scanForNewEmails() {
        // NOUVEAU SÉLECTEUR : Cible les lignes d'email basées sur le rôle ARIA "row"
        const emailRows = document.querySelectorAll('div[role="main"] tr[role="row"]');

        if (emailRows.length === 0) return;

        emailRows.forEach(row => {
            // NOUVEAU IDENTIFIANT : Utilise l'attribut data-thread-id qui est stable
            const threadId = row.dataset.threadId;

            if (threadId && !this.scannedThreadIds.has(threadId)) {
                this.scannedThreadIds.add(threadId);

                // NOUVEAUX SÉLECTEURS :
                const senderEl = row.querySelector('td.yX .yW span[email]');
                const subjectEl = row.querySelector('td.xY .xS .xT .bog');

                // Si les sélecteurs ci-dessus ne fonctionnent pas, essayons des alternatives
                const altSenderEl = row.querySelector('span[email]');
                const altSubjectEl = row.querySelector('span.bqe');

                const finalSenderEl = senderEl || altSenderEl;
                const finalSubjectEl = subjectEl || altSubjectEl;


                if (finalSenderEl && finalSubjectEl) {
                    const emailData = {
                        id: threadId,
                        sender: finalSenderEl.getAttribute('email') || finalSenderEl.innerText,
                        subject: finalSubjectEl.innerText,
                        body: '' // Le corps n'est pas disponible dans la vue liste
                    };

                    console.log('📧 Nouvel email détecté sur Gmail, envoi pour analyse:', emailData);

                    chrome.runtime.sendMessage({ type: 'PROCESS_NEW_EMAIL', data: emailData });
                } else {
                    // Log pour le débogage si un sélecteur échoue
                    if (!finalSenderEl) console.warn("Sélecteur d'expéditeur non trouvé pour la ligne :", row);
                    if (!finalSubjectEl) console.warn("Sélecteur de sujet non trouvé pour la ligne :", row);
                }
            }
        });
    }
}

new GmailScanner().init();