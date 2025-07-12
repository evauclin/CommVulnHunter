// mail-detector.js - Version corrigée avec debugging avancé
(() => {
    'use strict';

    // Marquer que le détecteur est actif
    window.guardianDetectorActive = true;

    console.log('🚀 [GUARDIAN DETECTOR] Démarrage du détecteur...');
    console.log('📍 [GUARDIAN DETECTOR] URL:', window.location.href);

    // === CONFIGURATION ===
    const isGmail = window.location.hostname.includes('google.com');
    const isYahoo = window.location.hostname.includes('yahoo.com');
    const serviceName = isGmail ? 'Gmail' : isYahoo ? 'Yahoo' : 'Inconnu';

    if (!isGmail && !isYahoo) {
        console.warn('⚠️ [GUARDIAN DETECTOR] Service non supporté:', window.location.hostname);
        return;
    }

    console.log(`📧 [GUARDIAN DETECTOR] Service détecté: ${serviceName}`);

    // Variables de tracking
    let lastUnreadCount = null;
    let detectionCount = 0;
    let isActive = true;

    // === FONCTION DE NOTIFICATION ===
    function sendNotification(mailData) {
        detectionCount++;
        console.log(`🔔 [GUARDIAN DETECTOR] Notification #${detectionCount}:`, mailData);

        // Envoyer via postMessage
        try {
            window.postMessage({
                type: 'GUARDIAN_NEW_MAIL',
                mailData: {
                    ...mailData,
                    detectionId: detectionCount,
                    timestamp: Date.now()
                }
            }, '*');

            console.log('✅ [GUARDIAN DETECTOR] Message envoyé via postMessage');

        } catch (error) {
            console.error('❌ [GUARDIAN DETECTOR] Erreur envoi notification:', error);
        }
    }

    // === DÉTECTION GMAIL ===
    if (isGmail) {
        console.log('🔧 [GUARDIAN DETECTOR] Configuration Gmail...');

        // 1. SURVEILLANCE DU TITRE
        function monitorGmailTitle() {
            const checkTitle = () => {
                const title = document.title;
                const unreadMatch = title.match(/^\((\d+)\)/);
                const currentCount = unreadMatch ? parseInt(unreadMatch[1]) : 0;

                // Log pour debugging
                if (currentCount !== lastUnreadCount) {
                    console.log(`📊 [GUARDIAN DETECTOR] Titre: "${title}" -> ${currentCount} non lus`);
                }

                // Première détection
                if (lastUnreadCount === null) {
                    lastUnreadCount = currentCount;
                    console.log(`📋 [GUARDIAN DETECTOR] État initial: ${currentCount} mails non lus`);
                    return;
                }

                // Nouveaux mails détectés
                if (currentCount > lastUnreadCount) {
                    const newMails = currentCount - lastUnreadCount;
                    console.log(`🆕 [GUARDIAN DETECTOR] ${newMails} nouveau(x) mail(s) détecté(s)!`);

                    // Créer une notification pour chaque nouveau mail
                    for (let i = 0; i < newMails; i++) {
                        sendNotification({
                            subject: `Nouveau mail Gmail #${i + 1}`,
                            sender: 'Gmail - Détection automatique',
                            preview: `Détecté via changement de titre (${lastUnreadCount} → ${currentCount})`,
                            source: 'Gmail - Titre',
                            method: 'title-change'
                        });
                    }

                    lastUnreadCount = currentCount;
                }
                // Mails lus
                else if (currentCount < lastUnreadCount) {
                    console.log(`📖 [GUARDIAN DETECTOR] Mails lus: ${lastUnreadCount} → ${currentCount}`);
                    lastUnreadCount = currentCount;
                }
            };

            // Vérifier toutes les 2 secondes
            const intervalId = setInterval(checkTitle, 2000);
            console.log('⏰ [GUARDIAN DETECTOR] Surveillance titre Gmail activée (2s)');

            // Vérification immédiate
            checkTitle();

            return intervalId;
        }

        // 2. OBSERVER LES CHANGEMENTS DOM
        function observeGmailDOM() {
            // Chercher le conteneur principal
            const containers = [
                'div[role="main"]',
                '.AO',
                '.nH',
                '[jsmodel]',
                '.aeF'
            ];

            let inboxContainer = null;
            for (const selector of containers) {
                inboxContainer = document.querySelector(selector);
                if (inboxContainer) {
                    console.log(`📦 [GUARDIAN DETECTOR] Conteneur trouvé: ${selector}`);
                    break;
                }
            }

            if (!inboxContainer) {
                console.warn('⚠️ [GUARDIAN DETECTOR] Aucun conteneur Gmail trouvé, retry dans 5s...');
                setTimeout(observeGmailDOM, 5000);
                return;
            }

            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.addedNodes.length > 0) {
                        console.log(`👀 [GUARDIAN DETECTOR] ${mutation.addedNodes.length} nouveaux éléments DOM`);

                        mutation.addedNodes.forEach((node) => {
                            if (node.nodeType === Node.ELEMENT_NODE) {
                                // Chercher les mails non lus
                                const mailSelectors = ['tr.zE', '.zA.zE', '[jsaction*="open"]'];

                                mailSelectors.forEach(selector => {
                                    if (node.matches && node.matches(selector)) {
                                        console.log(`📧 [GUARDIAN DETECTOR] Nouvel élément mail détecté: ${selector}`);

                                        setTimeout(() => {
                                            sendNotification({
                                                subject: 'Nouveau mail Gmail - DOM',
                                                sender: 'Gmail - Observer DOM',
                                                preview: 'Détecté via mutation observer',
                                                source: 'Gmail - DOM',
                                                method: 'dom-observer'
                                            });
                                        }, 500);
                                    }
                                });
                            }
                        });
                    }
                });
            });

            observer.observe(inboxContainer, {
                childList: true,
                subtree: true
            });

            console.log('👁️ [GUARDIAN DETECTOR] Observer DOM Gmail activé');
        }

        // Démarrer surveillance Gmail
        const titleInterval = monitorGmailTitle();
        setTimeout(observeGmailDOM, 3000);
    }

    // === TESTS ET FALLBACKS ===

    // Test automatique après 5 secondes
    setTimeout(() => {
        console.log('🧪 [GUARDIAN DETECTOR] Test automatique...');

        sendNotification({
            subject: 'Test automatique Guardian',
            sender: 'Système de test',
            preview: 'Notification de test pour vérifier le fonctionnement',
            source: `${serviceName} - Test auto`,
            method: 'auto-test'
        });

    }, 5000);

    // === COMMUNICATION EXTERNE ===

    // Écouter les messages
    window.addEventListener('message', (event) => {
        if (event.source !== window) return;

        console.log('📨 [GUARDIAN DETECTOR] Message reçu:', event.data.type);

        if (event.data.type === 'GUARDIAN_PING') {
            console.log('🏓 [GUARDIAN DETECTOR] Ping reçu, envoi status...');

            window.postMessage({
                type: 'GUARDIAN_STATUS',
                status: {
                    active: isActive,
                    service: serviceName,
                    detectionCount: detectionCount,
                    lastUnreadCount: lastUnreadCount,
                    timestamp: Date.now(),
                    version: '3.0-debug'
                }
            }, '*');
        }

        else if (event.data.type === 'GUARDIAN_FORCE_TEST') {
            console.log('🎯 [GUARDIAN DETECTOR] Test forcé demandé');

            sendNotification({
                subject: 'Test forcé par commande',
                sender: 'Test manuel',
                preview: 'Notification générée par commande externe',
                source: `${serviceName} - Force test`,
                method: 'force-test'
            });
        }
    });

    // === DEBUGGING AVANCÉ ===

    // Exposer des fonctions de test globalement
    window.guardianTest = {
        sendTest: () => {
            sendNotification({
                subject: 'Test manuel depuis console',
                sender: 'Console JavaScript',
                preview: 'Test depuis window.guardianTest.sendTest()',
                source: 'Console',
                method: 'manual-console'
            });
        },

        simulateTitle: (count) => {
            const oldTitle = document.title;
            document.title = `(${count}) Test - Gmail`;
            console.log(`🎭 [GUARDIAN TEST] Titre simulé: ${count} nouveaux mails`);

            setTimeout(() => {
                document.title = oldTitle;
                console.log('🔄 [GUARDIAN TEST] Titre restauré');
            }, 3000);
        },

        getStatus: () => {
            return {
                active: isActive,
                service: serviceName,
                detectionCount: detectionCount,
                lastUnreadCount: lastUnreadCount
            };
        }
    };

    console.log('🎮 [GUARDIAN DETECTOR] Fonctions de test disponibles:');
    console.log('  - window.guardianTest.sendTest()');
    console.log('  - window.guardianTest.simulateTitle(3)');
    console.log('  - window.guardianTest.getStatus()');

    // Status final
    console.log(`✅ [GUARDIAN DETECTOR] ${serviceName} initialisé et actif`);
    console.log(`🔍 [GUARDIAN DETECTOR] Surveillance: titre, DOM, tests automatiques`);

})();