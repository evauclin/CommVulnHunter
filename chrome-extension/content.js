// content.js - Script de contenu pour la détection de phishing en temps réel (VERSION CORRIGÉE)
console.log('🛡️ AI Anti-Phishing Guardian activé sur:', window.location.hostname);

class PhishingDetector {
    constructor() {
        // Configuration de l'API - utilise votre API FastAPI
        this.apiUrl = 'http://localhost:8000';
        this.isAnalyzing = false;
        this.currentUrl = window.location.href;
        this.pageData = null;
        this.analysisResult = null;
        this.hasAnalyzed = false;

        // Vérifier si on doit ignorer cette page
        if (this.shouldSkipPage()) {
            console.log('⏩ Page ignorée:', window.location.hostname);
            return;
        }

        this.init();
    }

    shouldSkipPage() {
        const hostname = window.location.hostname;
        const skipPatterns = [
            'localhost',
            '127.0.0.1',
            'chrome://',
            'chrome-extension://',
            'moz-extension://',
            'about:',
            'data:',
            'file://'
        ];

        return skipPatterns.some(pattern =>
            hostname.includes(pattern) || window.location.href.startsWith(pattern)
        );
    }

    async init() {
        // Attendre que la page soit complètement chargée
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                setTimeout(() => this.startAnalysis(), 1000);
            });
        } else {
            setTimeout(() => this.startAnalysis(), 1000);
        }

        // Surveiller les changements d'URL pour les SPAs
        this.setupUrlMonitoring();
    }

    setupUrlMonitoring() {
        let lastUrl = location.href;
        new MutationObserver(() => {
            const url = location.href;
            if (url !== lastUrl) {
                lastUrl = url;
                this.currentUrl = url;
                this.hasAnalyzed = false;
                setTimeout(() => this.startAnalysis(), 1500);
            }
        }).observe(document, { subtree: true, childList: true });
    }

    async startAnalysis() {
        if (this.isAnalyzing || this.hasAnalyzed) return;

        try {
            this.isAnalyzing = true;
            this.hasAnalyzed = true;

            console.log('🔍 Démarrage analyse ML pour:', this.currentUrl);
            this.showAnalysisIndicator();

            // Extraire les données de la page
            this.pageData = this.extractPageData();

            // Analyser avec votre modèle ML
            this.analysisResult = await this.analyzeWithML();

            // Afficher le résultat
            this.displayResult();

        } catch (error) {
            console.error('❌ Erreur analyse phishing:', error);
            this.hideAnalysisIndicator();
        } finally {
            this.isAnalyzing = false;
        }
    }

    extractPageData() {
        // Extraire le contenu principal de la page
        const textContent = this.extractMainTextContent();
        const title = document.title || '';
        const domain = window.location.hostname;

        return {
            url: this.currentUrl,
            domain: domain,
            title: title,
            text_content: textContent,
            form_data: this.extractFormData(),
            links: this.extractSuspiciousLinks(),
            has_login_form: this.hasLoginForm(),
            has_payment_form: this.hasPaymentForm(),
            suspicious_indicators: this.findSuspiciousIndicators(),
            page_language: this.detectPageLanguage(),
            timestamp: new Date().toISOString(),
            // Combiner tout en un texte pour votre API
            combined_text: this.createCombinedText(title, textContent)
        };
    }

    extractMainTextContent() {
        // Extraire le texte principal en évitant scripts/styles
        const elementsToSkip = ['script', 'style', 'noscript', 'nav', 'footer', 'aside'];
        let text = '';

        // Prioriser le contenu principal
        const mainContent = document.querySelector('main, article, .content, .main, #content, #main');
        const contentSource = mainContent || document.body;

        const walker = document.createTreeWalker(
            contentSource,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: function(node) {
                    const parent = node.parentElement;
                    if (!parent) return NodeFilter.FILTER_REJECT;

                    const tagName = parent.tagName.toLowerCase();
                    if (elementsToSkip.includes(tagName)) {
                        return NodeFilter.FILTER_REJECT;
                    }

                    // Ignorer les textes très courts ou vides
                    if (node.textContent.trim().length < 3) {
                        return NodeFilter.FILTER_REJECT;
                    }

                    return NodeFilter.FILTER_ACCEPT;
                }
            }
        );

        let node;
        while (node = walker.nextNode()) {
            text += node.textContent.trim() + ' ';
        }

        // Nettoyer et limiter la taille
        text = text.replace(/\s+/g, ' ').trim();
        return text.substring(0, 3000); // Limiter à 3000 chars
    }

    createCombinedText(title, content) {
        // Format similaire à vos emails pour l'API
        return `Title: ${title}\nDomain: ${window.location.hostname}\nContent: ${content}`;
    }

    extractFormData() {
        const forms = Array.from(document.querySelectorAll('form'));
        return forms.map(form => ({
            action: form.action || window.location.href,
            method: form.method || 'get',
            input_types: Array.from(form.querySelectorAll('input')).map(input => input.type),
            has_password: form.querySelector('input[type="password"]') !== null,
            has_email: form.querySelector('input[type="email"], input[name*="email"]') !== null,
            // 🔧 CORRECTION : Utiliser la fonction correcte avec le bon paramètre
            has_credit_card: this.hasPaymentForm(form)
        }));
    }

    extractSuspiciousLinks() {
        const links = Array.from(document.querySelectorAll('a[href]'));
        return links.slice(0, 5).map(link => ({
            href: link.href,
            text: link.textContent.trim().substring(0, 50),
            is_external: !link.href.includes(window.location.hostname),
            is_suspicious: this.isLinkSuspicious(link.href)
        }));
    }

    hasLoginForm(container = document) {
        const loginIndicators = [
            'input[type="password"]',
            'input[name*="password"]',
            'input[name*="login"]',
            'input[name*="username"]',
            'input[placeholder*="password" i]',
            'input[placeholder*="mot de passe" i]'
        ];

        return loginIndicators.some(selector => container.querySelector(selector) !== null);
    }

    hasPaymentForm(container = document) {
        const paymentIndicators = [
            'input[name*="card"]',
            'input[name*="credit"]',
            'input[name*="payment"]',
            'input[placeholder*="card" i]',
            'input[placeholder*="cvv" i]',
            'input[placeholder*="carte" i]',
            'input[maxlength="16"]', // Numéro de carte
            'input[maxlength="3"]',  // CVV
            'input[maxlength="4"]'   // CVV Amex
        ];

        return paymentIndicators.some(selector => container.querySelector(selector) !== null);
    }

    findSuspiciousIndicators() {
        const suspicious = [];
        const text = document.body.textContent.toLowerCase();

        // Patterns de phishing communs
        const phishingPatterns = [
            /urgent.*verify.*account/i,
            /suspended.*account.*click/i,
            /confirm.*identity.*immediately/i,
            /unusual.*activity.*detected/i,
            /click.*here.*now/i,
            /compte.*suspendu/i,
            /vérifiez.*maintenant/i,
            /activité.*suspecte/i
        ];

        phishingPatterns.forEach((pattern, index) => {
            if (pattern.test(text)) {
                suspicious.push(`suspicious_text_pattern_${index}`);
            }
        });

        // Vérifier les domaines suspects dans l'URL
        if (this.isDomainSuspicious(window.location.hostname)) {
            suspicious.push('suspicious_domain');
        }

        return suspicious;
    }

    isDomainSuspicious(domain) {
        const suspiciousPatterns = [
            /paypal.*(?!\.com$)/i,
            /amazon.*(?!\.com$|\.fr$)/i,
            /google.*(?!\.com$|\.fr$)/i,
            /microsoft.*(?!\.com$)/i,
            /apple.*(?!\.com$)/i,
            /facebook.*(?!\.com$)/i,
            /twitter.*(?!\.com$)/i
        ];

        return suspiciousPatterns.some(pattern => pattern.test(domain));
    }

    isLinkSuspicious(href) {
        try {
            const url = new URL(href);
            return this.isDomainSuspicious(url.hostname) ||
                   href.includes('bit.ly') ||
                   href.includes('tinyurl') ||
                   href.match(/\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/); // IP address
        } catch {
            return true; // URL malformée = suspect
        }
    }

    detectPageLanguage() {
        const lang = document.documentElement.lang ||
                    document.querySelector('meta[http-equiv="content-language"]')?.content ||
                    'en';
        return lang.substring(0, 2).toLowerCase();
    }

    async analyzeWithML() {
        try {
            console.log('🤖 Envoi vers API ML...');

            // Utiliser votre endpoint /predict avec le texte combiné
            const analysisData = {
                text: this.pageData.combined_text
            };

            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(analysisData),
                signal: AbortSignal.timeout(30000)
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }

            const result = await response.json();
            console.log('✅ Réponse API ML:', result);

            return {
                prediction: result.prediction,
                probability: result.probability,
                confidence: result.confidence,
                language_detected: result.language_detected || this.pageData.page_language,
                api_response: result
            };

        } catch (error) {
            console.error('❌ Erreur API ML:', error);
            // Fallback avec analyse basique
            return this.basicPhishingAnalysis();
        }
    }

    basicPhishingAnalysis() {
        // Analyse de fallback sans ML
        let suspicionScore = 0;
        const factors = [];

        // Analyser les indicateurs suspects
        if (this.pageData.suspicious_indicators.length > 0) {
            suspicionScore += Math.min(this.pageData.suspicious_indicators.length * 0.2, 0.6);
            factors.push(`${this.pageData.suspicious_indicators.length} indicateurs suspects`);
        }

        // Domaine suspect
        if (this.isDomainSuspicious(this.pageData.domain)) {
            suspicionScore += 0.7;
            factors.push('Domaine imitant une marque connue');
        }

        // Formulaires sensibles
        if (this.pageData.has_login_form && this.pageData.has_payment_form) {
            suspicionScore += 0.4;
            factors.push('Formulaires de connexion et paiement');
        }

        // Liens externes suspects
        const suspiciousLinks = this.pageData.links.filter(link => link.is_suspicious).length;
        if (suspiciousLinks > 0) {
            suspicionScore += suspiciousLinks * 0.15;
            factors.push(`${suspiciousLinks} liens suspects`);
        }

        return {
            prediction: suspicionScore > 0.6 ? 'phishing' : 'legitimate',
            probability: Math.min(suspicionScore, 0.95),
            confidence: suspicionScore > 0.8 ? 'HIGH' : suspicionScore > 0.4 ? 'MEDIUM' : 'LOW',
            language_detected: this.pageData.page_language,
            analysis_type: 'basic_rules',
            factors: factors
        };
    }

    showAnalysisIndicator() {
        if (document.getElementById('phishing-detector-indicator')) return;

        const indicator = document.createElement('div');
        indicator.id = 'phishing-detector-indicator';
        indicator.innerHTML = `
            <div class="phishing-indicator-content">
                <div class="spinner"></div>
                <span>🛡️ Analyse de sécurité en cours...</span>
            </div>
        `;

        document.body.appendChild(indicator);
    }

    hideAnalysisIndicator() {
        const indicator = document.getElementById('phishing-detector-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    displayResult() {
        this.hideAnalysisIndicator();

        if (!this.analysisResult) return;

        const isPhishing = this.analysisResult.prediction === 'phishing';
        const confidence = this.getConfidenceScore();

        // Envoyer le résultat au background script
        if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.sendMessage) {
            chrome.runtime.sendMessage({
                type: 'ANALYSIS_COMPLETE',
                data: {
                    url: this.currentUrl,
                    domain: this.pageData.domain,
                    isPhishing: isPhishing,
                    confidence: confidence,
                    analysis: this.analysisResult,
                    pageData: {
                        title: this.pageData.title,
                        hasLoginForm: this.pageData.has_login_form,
                        hasPaymentForm: this.pageData.has_payment_form,
                        suspiciousCount: this.pageData.suspicious_indicators.length
                    }
                }
            }).catch(() => {
                console.log('⚠️ Impossible d\'envoyer au background script');
            });
        }

        // Afficher l'alerte si phishing détecté avec confiance élevée
        if (isPhishing && confidence > 60) {
            this.showPhishingAlert();
        } else if (!isPhishing) {
            this.showSafeIndicator();
        }
    }

    getConfidenceScore() {
        if (typeof this.analysisResult.probability === 'number') {
            return Math.round(this.analysisResult.probability * 100);
        }

        // Fallback basé sur le niveau de confiance
        switch (this.analysisResult.confidence) {
            case 'HIGH': return 85;
            case 'MEDIUM': return 60;
            case 'LOW': return 35;
            default: return 50;
        }
    }

    showPhishingAlert() {
        // Supprimer les alertes existantes
        const existing = document.getElementById('phishing-alert');
        if (existing) existing.remove();

        const confidence = this.getConfidenceScore();
        const alert = document.createElement('div');
        alert.id = 'phishing-alert';
        alert.innerHTML = `
            <div class="phishing-alert-content">
                <div class="alert-header">
                    <span class="alert-icon">⚠️</span>
                    <strong>SITE SUSPECT DÉTECTÉ</strong>
                    <button class="close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
                </div>
                <div class="alert-body">
                    <p><strong>Notre IA a détecté que ce site pourrait être malveillant.</strong></p>
                    <p>Confiance de la détection: <strong>${confidence}%</strong></p>
                    <p><small>Domaine: ${this.pageData.domain}</small></p>
                    <div class="alert-actions">
                        <button class="btn-danger" onclick="window.history.back()">🔙 Retour sécurisé</button>
                        <button class="btn-secondary" onclick="window.phishingDetector.sendFeedback('ignore')">Ignorer cette alerte</button>
                        <button class="btn-primary" onclick="window.phishingDetector.sendFeedback('correct')">Signaler comme fausse alerte</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(alert);

        // Notification système
        if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.sendMessage) {
            chrome.runtime.sendMessage({
                type: 'SHOW_NOTIFICATION',
                title: '🚨 Site suspect détecté',
                message: `${this.pageData.domain} pourrait être dangereux (${confidence}% de confiance)`,
                type: 'danger'
            }).catch(() => {
                console.log('⚠️ Impossible d\'envoyer la notification');
            });
        }
    }

    showSafeIndicator() {
        const existing = document.getElementById('safe-indicator');
        if (existing) existing.remove();

        const indicator = document.createElement('div');
        indicator.id = 'safe-indicator';
        indicator.innerHTML = `
            <div class="safe-indicator-content">
                <span class="safe-icon">🛡️</span>
                <span>Site vérifié par IA</span>
            </div>
        `;

        document.body.appendChild(indicator);

        // Auto-hide après 3 secondes
        setTimeout(() => {
            if (indicator.parentNode) {
                indicator.remove();
            }
        }, 3000);
    }

    async sendFeedback(type) {
        try {
            const feedbackData = {
                email_text: this.pageData.combined_text,
                predicted_class: this.analysisResult.prediction,
                predicted_probability: this.analysisResult.probability || 0.5,
                user_satisfaction: type === 'correct' ? 'yes' : 'no',
                language_detected: this.analysisResult.language_detected || 'en'
            };

            console.log('📝 Envoi feedback vers API...');

            const response = await fetch(`${this.apiUrl}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedbackData),
                signal: AbortSignal.timeout(10000)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const result = await response.json();
            console.log('✅ Feedback envoyé:', result);

            // Masquer l'alerte
            const alert = document.getElementById('phishing-alert');
            if (alert) alert.remove();

            // Afficher message de remerciement
            this.showThankYouMessage(result.auto_finetuning_triggered);

        } catch (error) {
            console.error('❌ Erreur envoi feedback:', error);
            alert('Erreur lors de l\'envoi du feedback. Veuillez réessayer.');
        }
    }

    showThankYouMessage(finetuningTriggered = false) {
        const message = document.createElement('div');
        message.id = 'thank-you-message';
        message.innerHTML = `
            <div class="thank-you-content">
                <span>✅ Merci pour votre feedback !</span>
                ${finetuningTriggered ? '<small>🧠 L\'IA s\'améliore grâce à vous</small>' : ''}
            </div>
        `;

        document.body.appendChild(message);

        setTimeout(() => {
            if (message.parentNode) {
                message.remove();
            }
        }, 4000);
    }
}

// Initialiser le détecteur
let phishingDetector;

// Attendre que la page soit prête
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDetector);
} else {
    initializeDetector();
}

function initializeDetector() {
    phishingDetector = new PhishingDetector();
    // Exposer globalement pour les boutons d'alerte
    window.phishingDetector = phishingDetector;
}

// 🔧 CORRECTION : Vérifier que chrome et chrome.runtime existent avant d'écouter
if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.onMessage) {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (!phishingDetector) {
            sendResponse({ error: 'Detector not initialized' });
            return;
        }

        try {
            if (request.type === 'GET_PAGE_STATUS') {
                sendResponse({
                    url: window.location.href,
                    domain: window.location.hostname,
                    isAnalyzing: phishingDetector.isAnalyzing,
                    hasAnalyzed: phishingDetector.hasAnalyzed,
                    lastResult: phishingDetector.analysisResult,
                    pageData: phishingDetector.pageData
                });
            } else if (request.type === 'REANALYZE_PAGE') {
                phishingDetector.hasAnalyzed = false;
                phishingDetector.startAnalysis();
                sendResponse({ status: 'reanalysis_started' });
            } else if (request.type === 'GET_PAGE_DATA') {
                sendResponse({
                    pageData: phishingDetector.pageData,
                    analysisResult: phishingDetector.analysisResult
                });
            } else {
                sendResponse({ error: 'Unknown request type' });
            }
        } catch (error) {
            console.error('❌ Erreur handling message:', error);
            sendResponse({ error: error.message });
        }
    });
} else {
    console.warn('⚠️ Chrome runtime API non disponible - Mode standalone');
}

console.log('✅ AI Anti-Phishing Guardian initialisé pour:', window.location.hostname);