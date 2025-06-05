// csvEmailLoader.js - VERSION AVEC AFFICHAGE AMÉLIORÉ
// Gestionnaire d'emails avec support CSV et rendu amélioré

class EmailCSVLoader {
    constructor() {
        this.emails = [];
        this.stats = {};
        this.currentSource = 'unknown';
    }

    /**
     * Parse un fichier CSV d'emails
     */
    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        if (lines.length < 2) return [];

        // Extraire les en-têtes
        const headers = this.parseCSVLine(lines[0]);
        const emails = [];

        // Parser chaque ligne
        for (let i = 1; i < lines.length; i++) {
            const values = this.parseCSVLine(lines[i]);
            if (values.length === headers.length) {
                const email = {};
                headers.forEach((header, index) => {
                    email[header] = values[index] || '';
                });
                emails.push(email);
            }
        }

        return emails;
    }

    /**
     * Parse une ligne CSV en gérant les guillemets
     */
    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        let i = 0;

        while (i < line.length) {
            const char = line[i];
            const nextChar = line[i + 1];

            if (char === '"') {
                if (inQuotes && nextChar === '"') {
                    current += '"';
                    i += 2;
                } else {
                    inQuotes = !inQuotes;
                    i++;
                }
            } else if (char === ',' && !inQuotes) {
                result.push(current);
                current = '';
                i++;
            } else {
                current += char;
                i++;
            }
        }

        result.push(current);
        return result;
    }

    /**
     * Charge les emails depuis un fichier CSV
     */
    async loadEmailsFromCSV(filename = './emails_live.csv') {
        try {
            console.log(`📂 Chargement du fichier CSV: ${filename}`);

            const possiblePaths = [
                filename,
                `src/pages/${filename}`,
                `./${filename}`,
                `../${filename}`,
            ];

            let csvText = null;
            let successPath = null;

            for (const path of possiblePaths) {
                try {
                    console.log(`🔍 Tentative de chargement: ${path}`);
                    const response = await fetch(path);
                    if (response.ok) {
                        csvText = await response.text();
                        successPath = path;
                        break;
                    }
                } catch (e) {
                    console.log(`⚠️ Échec pour: ${path}`);
                }
            }

            if (!csvText) {
                throw new Error(`Fichier ${filename} non trouvé dans les chemins: ${possiblePaths.join(', ')}`);
            }

            this.emails = this.parseCSV(csvText);
            this.currentSource = 'csv_live';

            console.log(`✅ ${this.emails.length} emails chargés depuis: ${successPath}`);
            this.updateSourceIndicator(`📊 Emails CSV chargés (${this.emails.length}) depuis ${successPath}`, 'success');

            return this.emails;
        } catch (error) {
            console.warn(`⚠️ Erreur chargement CSV: ${error.message}`);
            throw error;
        }
    }

    /**
     * Charge les statistiques depuis le fichier JSON
     */
    async loadStats() {
        try {
            const possiblePaths = [
                'email_stats.json',
                'src/pages/email_stats.json',
                './email_stats.json',
                '../email_stats.json'
            ];

            for (const path of possiblePaths) {
                try {
                    const response = await fetch(path);
                    if (response.ok) {
                        this.stats = await response.json();
                        console.log(`📊 Statistiques chargées depuis: ${path}`, this.stats);
                        return;
                    }
                } catch (e) {
                    // Continuer avec le chemin suivant
                }
            }

            console.warn('⚠️ Impossible de charger les statistiques depuis tous les chemins');
        } catch (error) {
            console.warn('⚠️ Impossible de charger les statistiques:', error);
        }
    }

    /**
     * ✅ AMÉLIORÉ : Nettoie le contenu d'un email pour l'affichage
     */
    cleanEmailContent(content) {
        if (!content) return '';

        let cleaned = content;

        // Supprimer le HTML de base
        cleaned = cleaned.replace(/<[^>]*>/g, ' ');

        // Décoder les entités HTML communes
        cleaned = cleaned.replace(/&nbsp;/g, ' ');
        cleaned = cleaned.replace(/&amp;/g, '&');
        cleaned = cleaned.replace(/&lt;/g, '<');
        cleaned = cleaned.replace(/&gt;/g, '>');
        cleaned = cleaned.replace(/&quot;/g, '"');
        cleaned = cleaned.replace(/&#39;/g, "'");

        // Supprimer les caractères de contrôle
        cleaned = cleaned.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]/g, '');

        // Normaliser les espaces
        cleaned = cleaned.replace(/\s+/g, ' ');

        // Supprimer les répétitions de caractères étranges
        cleaned = cleaned.replace(/[^\w\s\-.,!?@()[\]{}:;'"]/g, '');

        return cleaned.trim();
    }

    /**
     * ✅ AMÉLIORÉ : Nettoie et formate l'adresse email
     */
    cleanEmailAddress(emailAddr) {
        if (!emailAddr) return 'Expéditeur inconnu';

        // Extraire juste l'email si format "Nom <email@domain.com>"
        const emailMatch = emailAddr.match(/<(.+?)>/);
        if (emailMatch) {
            const email = emailMatch[1];
            const namePart = emailAddr.replace(/<.+?>/, '').trim();
            return namePart ? `${namePart}` : email;
        }

        // Nettoyer les caractères étranges
        let cleaned = emailAddr.replace(/[^\w\s@.-]/g, '');

        // Limiter la longueur
        if (cleaned.length > 40) {
            cleaned = cleaned.substring(0, 37) + '...';
        }

        return cleaned || 'Expéditeur inconnu';
    }

    /**
     * ✅ AMÉLIORÉ : Nettoie et formate le sujet
     */
    cleanSubject(subject) {
        if (!subject) return 'Sans objet';

        let cleaned = subject;

        // Supprimer les encodages étranges
        cleaned = cleaned.replace(/=\?[^?]*\?[BQ]\?[^?]*\?=/gi, '');

        // Supprimer les préfixes répétés
        cleaned = cleaned.replace(/^(Re:|Fw:|Fwd:)\s*/gi, '');

        // Nettoyer les caractères de contrôle
        cleaned = cleaned.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]/g, '');

        // Supprimer les emojis cassés et caractères étranges
        cleaned = cleaned.replace(/[^\w\s\-.,!?()[\]{}:;'"@#$%&*+=\/\\|~`^]/g, ' ');

        // Normaliser les espaces
        cleaned = cleaned.replace(/\s+/g, ' ').trim();

        return cleaned || 'Sans objet';
    }

    /**
     * ✅ AMÉLIORÉ : Formate un email pour l'affichage
     */
    formatEmailForDisplay(email) {
        const isSpam = email.type === 'SPAM';
        const date = this.formatDate(email.date);

        // Nettoyage amélioré
        const cleanFrom = this.cleanEmailAddress(email.from);
        const cleanSubject = this.cleanSubject(email.subject);
        const cleanBody = this.cleanEmailContent(email.body);

        return {
            ...email,
            isSpam,
            formattedDate: date,
            shortDate: this.formatShortDate(email.date),
            cleanFrom,
            cleanSubject,
            cleanBody,
            truncatedFrom: this.truncate(cleanFrom, 35),
            truncatedSubject: this.truncate(cleanSubject, 60),
            truncatedBody: this.truncate(cleanBody, 100)
        };
    }

    /**
     * Formate une date
     */
    formatDate(dateStr) {
        try {
            const date = new Date(dateStr);
            return date.toLocaleDateString('fr-FR', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return dateStr || 'Date inconnue';
        }
    }

    /**
     * Formate une date courte
     */
    formatShortDate(dateStr) {
        try {
            const date = new Date(dateStr);
            return date.toLocaleDateString('fr-FR', {
                day: '2-digit',
                month: '2-digit'
            });
        } catch {
            return '';
        }
    }

    /**
     * Tronque un texte intelligemment
     */
    truncate(text, length) {
        if (!text) return '';
        if (text.length <= length) return text;

        // Essayer de couper à un mot complet
        const truncated = text.substring(0, length);
        const lastSpace = truncated.lastIndexOf(' ');

        if (lastSpace > length * 0.8) {
            return truncated.substring(0, lastSpace) + '...';
        }

        return truncated + '...';
    }

    /**
     * Met à jour l'indicateur de source
     */
    updateSourceIndicator(message, type = 'info') {
        const indicator = document.getElementById('sourceIndicator');
        const text = document.getElementById('sourceText');

        if (indicator && text) {
            text.textContent = message;
            text.className = `text-${type}`;
            indicator.style.display = 'block';

            if (type === 'success') {
                setTimeout(() => {
                    indicator.style.display = 'none';
                }, 5000);
            }
        }
    }

    /**
     * Calcule les statistiques des emails chargés
     */
    calculateStats() {
        const total = this.emails.length;
        const important = this.emails.filter(e => e.type === 'IMPORTANT').length;
        const spam = this.emails.filter(e => e.type === 'SPAM').length;
        const successRate = total > 0 ? Math.round((important / total) * 100) : 0;

        return {
            total,
            important,
            spam,
            successRate
        };
    }

    /**
     * ✅ AMÉLIORÉ : Affiche les emails dans l'interface avec un meilleur rendu
     */
    displayEmails(container, emails = null) {
        const emailsToDisplay = emails || this.emails;

        if (!container) {
            console.error('Container not found for displaying emails');
            return;
        }

        if (emailsToDisplay.length === 0) {
            container.innerHTML = `
                <div class="text-center p-4 text-muted">
                    <i class="bi bi-inbox" style="font-size: 3rem; opacity: 0.3;"></i>
                    <div class="mt-2">Aucun email trouvé</div>
                    <small>Essayez de charger des emails ou vérifiez vos filtres</small>
                </div>
            `;
            return;
        }

        let html = '';
        emailsToDisplay.forEach((email, index) => {
            const formatted = this.formatEmailForDisplay(email);

            html += `
                <div class="email-item ${formatted.isSpam ? 'spam' : ''}" 
                     onclick="selectEmail(${index})" 
                     data-index="${index}"
                     title="Cliquez pour voir les détails">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <div class="d-flex align-items-center mb-2">
                                <strong class="me-2" style="color: ${formatted.isSpam ? '#dc3545' : '#198754'};">
                                    ${this.escapeHtml(formatted.truncatedFrom)}
                                </strong>
                                <span class="email-type-badge ${formatted.isSpam ? 'bg-danger' : 'bg-success'} text-white">
                                    ${email.type}
                                </span>
                            </div>
                            <div class="text-dark fw-semibold mb-1" style="line-height: 1.3;">
                                ${this.escapeHtml(formatted.truncatedSubject)}
                            </div>
                            <div class="text-muted small" style="line-height: 1.4;">
                                ${this.escapeHtml(formatted.truncatedBody)}
                            </div>
                        </div>
                        <div class="text-end ms-3 flex-shrink-0">
                            <small class="text-muted">${formatted.shortDate}</small>
                            <div class="mt-1">
                                <i class="bi bi-${formatted.isSpam ? 'shield-x text-danger' : 'shield-check text-success'}" 
                                   style="font-size: 0.8rem;"></i>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    }

    /**
     * ✅ NOUVEAU : Échappe le HTML pour éviter l'injection
     */
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Met à jour les statistiques dans l'interface
     */
    updateStatsDisplay() {
        const stats = this.calculateStats();

        const totalElement = document.getElementById('totalEmails');
        const successElement = document.getElementById('successRate');
        const spamElement = document.getElementById('spamCount');
        const counterElement = document.getElementById('emailCounter');

        if (totalElement) totalElement.textContent = stats.total;
        if (successElement) successElement.textContent = stats.successRate + '%';
        if (spamElement) spamElement.textContent = stats.spam;
        if (counterElement) counterElement.textContent = `${stats.total} emails`;
    }

    /**
     * Filtre les emails par type
     */
    filterEmails(type = null) {
        if (!type) return this.emails;
        return this.emails.filter(email =>
            email.type && email.type.toUpperCase() === type.toUpperCase()
        );
    }

    /**
     * Recherche dans les emails
     */
    searchEmails(query) {
        if (!query) return this.emails;

        const searchTerm = query.toLowerCase();
        return this.emails.filter(email => {
            const formatted = this.formatEmailForDisplay(email);
            return (
                formatted.cleanFrom.toLowerCase().includes(searchTerm) ||
                formatted.cleanSubject.toLowerCase().includes(searchTerm) ||
                formatted.cleanBody.toLowerCase().includes(searchTerm)
            );
        });
    }
}

// Instance globale
const emailLoader = new EmailCSVLoader();

// Fonctions globales pour compatibilité
async function loadLiveEmails() {
    try {
        console.log('🔄 Chargement des emails live...');

        await emailLoader.loadEmailsFromCSV('./emails_live.csv');
        await emailLoader.loadStats();

        const container = document.getElementById('emailList');
        emailLoader.displayEmails(container);
        emailLoader.updateStatsDisplay();

        console.log('📧 Emails live chargés avec succès');

        if (emailLoader.emails.length > 0) {
            selectEmail(0);
        }

    } catch (error) {
        console.warn('⚠️ Erreur chargement emails live:', error);
        console.warn('🔄 Fallback vers emails de démo');
        loadDemoEmails();
    }
}

async function loadDemoEmails() {
    try {
        try {
            await emailLoader.loadEmailsFromCSV('emails_demo.csv');
        } catch {
            emailLoader.emails = createDemoEmails();
            emailLoader.currentSource = 'demo_generated';
        }

        const container = document.getElementById('emailList');
        emailLoader.displayEmails(container);
        emailLoader.updateStatsDisplay();
        emailLoader.updateSourceIndicator('📁 Emails de démonstration chargés', 'info');

        if (emailLoader.emails.length > 0) {
            selectEmail(0);
        }

    } catch (error) {
        console.error('❌ Erreur chargement emails démo:', error);
        emailLoader.updateSourceIndicator('❌ Erreur de chargement', 'danger');
    }
}

function createDemoEmails() {
    return [
        {
            id: 'demo_1',
            type: 'IMPORTANT',
            from: 'Alice Martin <alice@example.com>',
            to: 'moi@monemail.com',
            date: '2025-06-01 10:00:00',
            subject: 'Réunion importante demain - Projet Q2',
            body: 'Bonjour, nous avons une réunion importante demain à 10h00 pour finaliser le projet du Q2. Merci de préparer les documents nécessaires.',
            message_id: '',
            processed_at: new Date().toISOString()
        },
        {
            id: 'demo_2',
            type: 'SPAM',
            from: 'promo@fake-deals.com',
            to: 'moi@monemail.com',
            date: '2025-06-01 11:00:00',
            subject: 'URGENT: Gagnez 1000€ maintenant !!!',
            body: 'Félicitations ! Vous avez été sélectionné pour gagner 1000€. Cliquez ici maintenant pour réclamer votre prix avant qu\'il soit trop tard !',
            message_id: '',
            processed_at: new Date().toISOString()
        },
        {
            id: 'demo_3',
            type: 'IMPORTANT',
            from: 'support@entreprise.fr',
            to: 'moi@monemail.com',
            date: '2025-06-01 14:30:00',
            subject: 'Mise à jour de sécurité requise',
            body: 'Une mise à jour de sécurité importante est disponible pour votre compte. Veuillez vous connecter pour l\'installer.',
            message_id: '',
            processed_at: new Date().toISOString()
        }
    ];
}

function selectEmail(index) {
    document.querySelectorAll('.email-item').forEach(item => {
        item.classList.remove('active');
    });

    const emailElement = document.querySelector(`[data-index="${index}"]`);
    if (emailElement) {
        emailElement.classList.add('active');
    }

    const email = emailLoader.emails[index];
    if (email) {
        const formatted = emailLoader.formatEmailForDisplay(email);

        const elements = {
            id: document.getElementById('emailId'),
            from: document.getElementById('emailFrom'),
            date: document.getElementById('emailDate'),
            subject: document.getElementById('emailSubject'),
            body: document.getElementById('emailBody'),
            analyzeBtn: document.getElementById('analyzeBtn')
        };

        if (elements.id) elements.id.textContent = email.id || 'N/A';
        if (elements.from) elements.from.textContent = formatted.cleanFrom;
        if (elements.date) elements.date.textContent = formatted.formattedDate;
        if (elements.subject) elements.subject.textContent = formatted.cleanSubject;

        if (elements.body) {
            elements.body.dataset.textContent = formatted.cleanBody;
            elements.body.dataset.htmlContent = email.body;
            updateBodyDisplay();
        }

        if (elements.analyzeBtn) {
            elements.analyzeBtn.dataset.emailType = email.type;
        }

        // Reset analysis
        const analysisResult = document.getElementById('analysisResult');
        if (analysisResult) analysisResult.textContent = '';

        document.querySelectorAll('input[name="satisfaction"]').forEach(input => {
            input.checked = false;
        });
    }
}

function refreshEmails() {
    console.log('🔄 Actualisation des emails...');
    if (emailLoader.currentSource === 'csv_live') {
        loadLiveEmails();
    } else {
        loadDemoEmails();
    }
}

// Export pour utilisation dans d'autres scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EmailCSVLoader, emailLoader };
}