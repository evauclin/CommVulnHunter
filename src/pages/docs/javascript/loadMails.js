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