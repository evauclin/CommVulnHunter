// loadMails.js

// Fonction pour parser l'archive
function parseArchive(text) {
  const mailsRaw = text.trim().split('--------------------------------------------');
  return mailsRaw.map(raw => {
    const lines = raw.trim().split('\n').map(l => l.trim());
    if (lines.length < 6) return null;

    const tag = lines[0].replace(/\[|\]/g, '');
    let from = '', to = '', date = '', subject = '';
    let i = 1;

    for (; i < lines.length; i++) {
      if (lines[i].startsWith('From:')) from = lines[i].substring(5).trim();
      else if (lines[i].startsWith('To:')) to = lines[i].substring(3).trim();
      else if (lines[i].startsWith('Date:')) date = lines[i].substring(5).trim();
      else if (lines[i].startsWith('Subject:')) {
        subject = lines[i].substring(8).trim();
        break;
      }
    }

    return { tag, from, to, date, subject };
  }).filter(m => m !== null);
}

// Fonction pour afficher les mails dans le div #mailContainer
function displayMails(mails) {
  const container = document.getElementById('mailContainer');
  if (!container) {
    console.error('Element with id "mailContainer" not found.');
    return;
  }

  container.innerHTML = '';

  mails.forEach(mail => {
    const div = document.createElement('div');
    div.className = 'mail-item ' + (mail.tag === 'SPAM' ? 'spam' : 'important');
    div.style.border = mail.tag === 'SPAM' ? '2px solid red' : '2px solid green';
    div.style.padding = '10px';
    div.style.marginBottom = '10px';
    div.style.borderRadius = '5px';
    div.style.cursor = 'pointer';

    div.innerHTML = `<strong>${mail.from}</strong> — <em>${mail.subject}</em><br><small>${mail.date}</small>`;

    container.appendChild(div);
  });
}

// Fonction principale pour charger et afficher les mails
function loadAndDisplayMails(archiveFile = 'archive_20_mails.txt') {
  fetch(archiveFile)
    .then(response => {
      if (!response.ok) throw new Error('Erreur chargement fichier');
      return response.text();
    })
    .then(text => {
      const mails = parseArchive(text);
      displayMails(mails);
    })
    .catch(error => {
      const container = document.getElementById('mailContainer');
      if (container) {
        container.textContent = 'Erreur : ' + error.message;
      }
      console.error(error);
    });
}

// Appelle la fonction au chargement du script
document.addEventListener('DOMContentLoaded', () => {
  loadAndDisplayMails();
});
// Fonction pour parser l'archive (votre fonction existante)
function parseArchive(text) {
  const mailsRaw = text.trim().split('--------------------------------------------');
  return mailsRaw.map(raw => {
    const lines = raw.trim().split('\n').map(l => l.trim());
    if (lines.length < 6) return null;

    const tag = lines[0].replace(/\[|\]/g, '');
    let from = '', to = '', date = '', subject = '';
    let i = 1;

    for (; i < lines.length; i++) {
      if (lines[i].startsWith('From:')) from = lines[i].substring(5).trim();
      else if (lines[i].startsWith('To:')) to = lines[i].substring(3).trim();
      else if (lines[i].startsWith('Date:')) date = lines[i].substring(5).trim();
      else if (lines[i].startsWith('Subject:')) {
        subject = lines[i].substring(8).trim();
        break;
      }
    }

    return { tag, from, to, date, subject };
  }).filter(m => m !== null);
}

// Fonction pour afficher les mails (votre fonction existante)
function displayMails(mails) {
  const container = document.getElementById('mailContainer');
  if (!container) {
    console.error('Element with id "mailContainer" not found.');
    return;
  }

  container.innerHTML = '';

  mails.forEach(mail => {
    const div = document.createElement('div');
    div.className = 'mail-item ' + (mail.tag === 'SPAM' ? 'spam' : 'important');
    div.style.border = mail.tag === 'SPAM' ? '2px solid red' : '2px solid green';
    div.style.padding = '10px';
    div.style.marginBottom = '10px';
    div.style.borderRadius = '5px';
    div.style.cursor = 'pointer';

    div.innerHTML = `<strong>${mail.from}</strong> — <em>${mail.subject}</em><br><small>${mail.date}</small>`;

    container.appendChild(div);
  });
}

// === NOUVELLES FONCTIONS AJOUTÉES ===

// Fonction pour charger les emails live depuis Gmail
function loadLiveEmails() {
  fetch('archive_mails_live.txt')
    .then(response => {
      if (!response.ok) throw new Error('Fichier live non trouvé');
      return response.text();
    })
    .then(text => {
      const mails = parseArchive(text);
      displayMails(mails);
      console.log('📧 Emails live chargés:', mails.length);
      updateSourceIndicator('Emails Gmail (Live)', 'success');
    })
    .catch(error => {
      console.warn('⚠️ Erreur chargement emails live:', error);
      // Fallback vers les emails statiques
      loadAndDisplayMails();
    });
}

// Fonction pour charger les emails statiques (votre fonction originale)
function loadAndDisplayMails(archiveFile = 'archive_mails.txt') {
  fetch(archiveFile)
    .then(response => {
      if (!response.ok) throw new Error('Erreur chargement fichier');
      return response.text();
    })
    .then(text => {
      const mails = parseArchive(text);
      displayMails(mails);
      console.log('📧 Emails statiques chargés:', mails.length);
      updateSourceIndicator('Emails statiques', 'info');
    })
    .catch(error => {
      const container = document.getElementById('mailContainer');
      if (container) {
        container.textContent = 'Erreur : ' + error.message;
      }
      console.error(error);
      updateSourceIndicator('Erreur de chargement', 'error');
    });
}

// Fonction pour mettre à jour l'indicateur de source
function updateSourceIndicator(source, type = 'info') {
  let indicator = document.getElementById('sourceIndicator');
  if (!indicator) {
    // Créer l'indicateur s'il n'existe pas
    indicator = document.createElement('div');
    indicator.id = 'sourceIndicator';
    indicator.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      padding: 8px 12px;
      border-radius: 5px;
      font-size: 12px;
      z-index: 1000;
      color: white;
    `;
    document.body.appendChild(indicator);
  }

  // Couleurs selon le type
  const colors = {
    'success': '#28a745',
    'info': '#17a2b8',
    'warning': '#ffc107',
    'error': '#dc3545'
  };

  indicator.style.backgroundColor = colors[type] || colors.info;
  indicator.innerHTML = `📊 Source: ${source}`;
}

// Fonction pour ajouter les boutons de contrôle
function addControlButtons() {
  // Chercher un endroit pour ajouter les boutons
  let controlContainer = document.querySelector('.control-buttons');

  if (!controlContainer) {
    // Créer le conteneur de boutons
    controlContainer = document.createElement('div');
    controlContainer.className = 'control-buttons';
    controlContainer.style.cssText = `
      margin: 10px 0;
      text-align: center;
      padding: 10px;
      background: #f8f9fa;
      border-radius: 5px;
    `;

    // L'insérer avant le conteneur d'emails
    const mailContainer = document.getElementById('mailContainer');
    if (mailContainer && mailContainer.parentNode) {
      mailContainer.parentNode.insertBefore(controlContainer, mailContainer);
    } else {
      document.body.appendChild(controlContainer);
    }
  }

  controlContainer.innerHTML = `
    <h4>📧 Gestion des emails</h4>
    <button onclick="loadLiveEmails()" style="margin: 5px; padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">
      🔄 Charger emails Gmail
    </button>
    <button onclick="loadAndDisplayMails()" style="margin: 5px; padding: 8px 16px; background: #17a2b8; color: white; border: none; border-radius: 4px; cursor: pointer;">
      📁 Charger emails statiques
    </button>
    <button onclick="refreshEmails()" style="margin: 5px; padding: 8px 16px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">
      🔃 Actualiser
    </button>
  `;
}

// Fonction pour actualiser les emails
function refreshEmails() {
  updateSourceIndicator('Actualisation...', 'warning');

  // Essayer d'abord les emails live, puis fallback
  loadLiveEmails();
}

// === INITIALISATION ===

// Attendre que le DOM soit chargé
document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 Initialisation du gestionnaire d\'emails étendu');

  // Ajouter les boutons de contrôle
  addControlButtons();

  // Essayer de charger les emails live d'abord
  console.log('📡 Tentative de chargement des emails live...');
  loadLiveEmails();
});

// Fonction de compatibilité avec votre code existant
if (typeof window !== 'undefined') {
  window.loadAndDisplayMails = loadAndDisplayMails;
  window.loadLiveEmails = loadLiveEmails;
  window.refreshEmails = refreshEmails;
}

