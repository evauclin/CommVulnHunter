// csvEmailLoader.js
// Gestionnaire d'emails avec support CSV

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
                    // Double quote - échappement
                    current += '"';
                    i += 2;
                } else {
                    // Toggle quote state
                    inQuotes = !inQuotes;
                    i++;
                }
            } else if (char === ',' && !inQuotes) {
                // Fin de champ
                result.push(current);
                current = '';
                i++;
            } else {
                current += char;
                i++;
            }
        }

        // Ajouter le dernier champ
        result.push(current);
        return result;
    }

    /**
     * Charge les emails depuis un fichier CSV
     */
    async loadEmailsFromCSV(filename = 'emails_live.csv') {
        try {
            console.log(`📂 Chargement du fichier CSV: ${filename}`);

            const response = await fetch(filename);
            if (!response.ok) {
                throw new Error(`Fichier ${filename} non trouvé`);
            }

            const csvText = await response.text();
            this.emails = this.parseCSV(csvText);
            this.currentSource = 'csv_live';

            console.log(`✅ ${this.emails.length} emails chargés depuis CSV`);
            this.updateSourceIndicator(`📊 Emails CSV chargés (${this.emails.length})`, 'success');

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
            const response = await fetch('../../email_stats.json');
            if (response.ok) {
                this.stats = await response.json();
                console.log('📊 Statistiques chargées:', this.stats);
            }
        } catch (error) {
            console.warn('⚠️ Impossible de charger les statistiques:', error);
        }
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
        return this.emails.filter(email =>
            (email.from && email.from.toLowerCase().includes(searchTerm)) ||
            (email.subject && email.subject.toLowerCase().includes(searchTerm)) ||
            (email.body && email.body.toLowerCase().includes(searchTerm))
        );
    }

    /**
     * Formate un email pour l'affichage
     */
    formatEmailForDisplay(email) {
        const isSpam = email.type === 'SPAM';
        const date = this.formatDate(email.date);
        const cleanBody = this.cleanEmailContent(email.body);

        return {
            ...email,
            isSpam,
            formattedDate: date,
            shortDate: this.formatShortDate(email.date),
            cleanBody,
            truncatedFrom: this.truncate(email.from, 35),
            truncatedSubject: this.truncate(email.subject, 50),
            truncatedBody: this.truncate(cleanBody, 70)
        };
    }

    /**
     * Nettoie le contenu d'un email
     */
    cleanEmailContent(content) {
        if (!content) return '';

        // Nettoyer les caractères de contrôle et normaliser les espaces
        let cleaned = content
            .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]/g, '') // Caractères de contrôle
            .replace(/\s+/g, ' ') // Multiples espaces
            .trim();

        return cleaned;
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
     * Tronque un texte
     */
    truncate(text, length) {
        if (!text) return '';
        return text.length > length ? text.substring(0, length) + '...' : text;
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

            // Masquer automatiquement après 3 secondes pour les succès
            if (type === 'success') {
                setTimeout(() => {
                    indicator.style.display = 'none';
                }, 3000);
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
     * Affiche les emails dans l'interface
     */
    displayEmails(container, emails = null) {
        const emailsToDisplay = emails || this.emails;

        if (!container) {
            console.error('Container not found for displaying emails');
            return;
        }

        if (emailsToDisplay.length === 0) {
            container.innerHTML = '<div class="text-center p-4 text-muted">Aucun email trouvé</div>';
            return;
        }

        let html = '';
        emailsToDisplay.forEach((email, index) => {
            const formatted = this.formatEmailForDisplay(email);

            html += `
                <div class="email-item ${formatted.isSpam ? 'spam' : ''}" 
                     onclick="selectEmail(${index})" 
                     data-index="${index}">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <div class="d-flex align-items-center mb-2">
                                <strong class="me-2">${formatted.truncatedFrom}</strong>
                                <span class="email-type-badge ${formatted.isSpam ? 'bg-danger' : 'bg-success'} text-white">
                                    ${email.type}
                                </span>
                            </div>
                            <div class="text-dark fw-semibold mb-1">${formatted.truncatedSubject}</div>
                            <div class="text-muted small">${formatted.truncatedBody}</div>
                        </div>
                        <div class="text-end ms-3">
                            <small class="text-muted">${formatted.shortDate}</small>
                        </div>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
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
}

// Instance globale
const emailLoader = new EmailCSVLoader();

// Fonctions globales pour compatibilité
async function loadLiveEmails() {

    try {
        await emailLoader.loadEmailsFromCSV('src/pages/emails_live.csv');
        await emailLoader.loadStats();

        const container = document.getElementById('emailList');
        emailLoader.displayEmails(container);
        emailLoader.updateStatsDisplay();

        console.log('📧 Emails live chargés avec succès');

        // Sélectionner le premier email
        if (emailLoader.emails.length > 0) {
            selectEmail(0);
        }

    } catch (error) {
        console.warn('⚠️ Fallback vers emails de démo');
        loadDemoEmails();
    }
}

async function loadDemoEmails() {
    try {
        // Essayer d'abord le CSV de démo
        try {
            await emailLoader.loadEmailsFromCSV('src/pages/emails_demo.csv');
        } catch {
            // Fallback vers les données intégrées
            const demoData = document.getElementById('emailData');
            if (demoData) {
                // Convertir les données intégrées en format CSV-like
                emailLoader.emails = parseOldFormat(demoData.textContent);
                emailLoader.currentSource = 'embedded';
            }
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

function selectEmail(index) {
    // Supprimer la sélection précédente
    document.querySelectorAll('.email-item').forEach(item => {
        item.classList.remove('active');
    });

    // Ajouter la nouvelle sélection
    const emailElement = document.querySelector(`[data-index="${index}"]`);
    if (emailElement) {
        emailElement.classList.add('active');
    }

    // Afficher l'aperçu
    const email = emailLoader.emails[index];
    if (email) {
        const fromElement = document.getElementById('emailFrom');
        const subjectElement = document.getElementById('emailSubject');
        const bodyElement = document.getElementById('emailBody');
        const analyzeBtn = document.getElementById('analyzeBtn');

        if (fromElement) fromElement.textContent = email.from;
        if (subjectElement) subjectElement.textContent = email.subject;

        if (bodyElement) {
            bodyElement.dataset.textContent = email.body;
            bodyElement.dataset.htmlContent = email.body;
            updateBodyDisplay();
        }

        if (analyzeBtn) {
            analyzeBtn.dataset.emailType = email.type;
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

// Fonction pour convertir l'ancien format au nouveau
function parseOldFormat(text) {
    const emailBlocks = text.trim().split('--------------------------------------------');
    return emailBlocks.map((block, index) => {
        const lines = block.trim().split('\n');
        if (lines.length < 5) return null;

        let type = '', from = '', to = '', date = '', subject = '', body = '';
        let i = 0;

        // Type
        if (lines[0].startsWith('[')) {
            type = lines[0].replace(/\[|\]/g, '');
            i++;
        }

        // Headers
        for (; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.startsWith('From:')) from = line.substring(5).trim();
            else if (line.startsWith('To:')) to = line.substring(3).trim();
            else if (line.startsWith('Date:')) date = line.substring(5).trim();
            else if (line.startsWith('Subject:')) {
                subject = line.substring(8).trim();
                i++;
                break;
            }
        }

        body = lines.slice(i).join('\n').trim();

        return {
            id: `demo_${index}`,
            type: type || 'IMPORTANT',
            from,
            to,
            date,
            subject,
            body,
            message_id: '',
            processed_at: new Date().toISOString()
        };
    }).filter(email => email !== null);
}

// Export pour utilisation dans d'autres scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EmailCSVLoader, emailLoader };
}