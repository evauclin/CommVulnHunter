const express = require('express');
const fs = require('fs');
const bcrypt = require('bcryptjs');
const path = require('path');
const session = require('express-session');
const { google } = require('googleapis');

const app = express();
const USERS_FILE = path.join(__dirname, 'users.json');

require('dotenv').config();

const CLIENT_ID = process.env.CLIENT_ID;
const CLIENT_SECRET = process.env.CLIENT_SECRET;
const REDIRECT_URI = process.env.REDIRECT_URI;
const SCOPES = ['https://www.googleapis.com/auth/gmail.readonly'];
// Ensure data folder exists
const dataFolder = path.join(__dirname, 'app', 'emailsCallback');
if (!fs.existsSync(dataFolder)) fs.mkdirSync(dataFolder, { recursive: true });

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(session({
    secret: 'your-session-secret',
    resave: false,
    saveUninitialized: true
}));

// Serve login.html on root
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'src', 'pages', 'login.html'));
});

// Serve static files from src/pages
app.use(express.static(path.join(__dirname, 'src', 'pages'), { index: false }));

// Serve data folder statically
app.use('/data', express.static(dataFolder));

// Helpers
function loadUsers() {
    if (!fs.existsSync(USERS_FILE)) return [];
    const data = fs.readFileSync(USERS_FILE, 'utf8');
    return JSON.parse(data);
}

function saveUsers(users) {
    fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2));
}

function getBody(payload) {
    let body = '';

    if (!payload) return body;

    if (payload.parts && payload.parts.length) {
        for (const part of payload.parts) {
            if (part.mimeType === 'text/plain' && part.body && part.body.data) {
                body = Buffer.from(part.body.data, 'base64').toString('utf-8');
                break;
            }
            if (part.mimeType === 'text/html' && part.body && part.body.data && !body) {
                body = Buffer.from(part.body.data, 'base64').toString('utf-8');
            }
        }
    } else if (payload.body && payload.body.data) {
        body = Buffer.from(payload.body.data, 'base64').toString('utf-8');
    }

    return body;
}

// Registration endpoint
app.post('/register', async (req, res) => {
    const { name, email, password } = req.body;
    if (!name || !email || !password) {
        return res.status(400).json({ success: false, message: "Missing fields" });
    }

    const users = loadUsers();
    if (users.find(u => u.email === email)) {
        return res.status(400).json({ success: false, message: "User already exists" });
    }

    try {
        const hashedPassword = await bcrypt.hash(password, 10);
        users.push({ name, email, password: hashedPassword });
        saveUsers(users);
        return res.json({ success: true, message: "User registered successfully" });
    } catch (err) {
        console.error('Error hashing password:', err);
        return res.status(500).json({ success: false, message: "Server error" });
    }
});

// Login endpoint
app.post('/login', async (req, res) => {
    const { email, password } = req.body;
    if (!email || !password) {
        return res.status(400).json({ success: false, message: 'Missing email or password' });
    }

    const users = loadUsers();
    const user = users.find(u => u.email === email);
    if (!user) {
        return res.status(400).json({ success: false, message: 'User not found' });
    }

    try {
        const match = await bcrypt.compare(password, user.password);
        if (match) {
            req.session.userEmail = email;
            return res.json({ success: true, message: 'Login successful' });
        } else {
            return res.status(400).json({ success: false, message: 'Incorrect password' });
        }
    } catch (err) {
        console.error('Login error:', err);
        return res.status(500).json({ success: false, message: 'Server error' });
    }
});

// Forgot Password endpoint
app.post('/forgot-password', async (req, res) => {
    const { email, newPassword } = req.body;
    if (!email || !newPassword) {
        return res.status(400).json({ success: false, message: 'Missing email or new password' });
    }

    const users = loadUsers();
    const userIndex = users.findIndex(u => u.email === email);

    if (userIndex === -1) {
        return res.status(404).json({ success: false, message: 'User not found' });
    }

    try {
        const hashedPassword = await bcrypt.hash(newPassword, 10);
        users[userIndex].password = hashedPassword;
        saveUsers(users);
        return res.json({ success: true, message: 'Password reset successful' });
    } catch (err) {
        console.error('Error resetting password:', err);
        return res.status(500).json({ success: false, message: 'Server error' });
    }
});

// Google OAuth client
function createOAuthClient() {
    return new google.auth.OAuth2(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI);
}

// Start OAuth flow
app.get('/auth/google', (req, res) => {
    const oauth2Client = createOAuthClient();
    const authUrl = oauth2Client.generateAuthUrl({
        access_type: 'offline',
        scope: SCOPES,
        prompt: 'consent'
    });
    res.redirect(authUrl);
});

// Logout from index.html
app.post('/logout', (req, res) => {
    req.session.destroy(err => {
        if (err) {
            console.error('Logout error:', err);
            return res.status(500).send('Could not log out');
        }
        res.clearCookie('connect.sid'); // Replace with your session cookie name if different
        res.sendStatus(200);
    });
});

// OAuth callback
app.get('/auth/google/callback', async (req, res) => {
    const code = req.query.code;
    if (!code) return res.status(400).send('No code provided');

    const oauth2Client = createOAuthClient();

    try {
        const { tokens } = await oauth2Client.getToken(code);
        oauth2Client.setCredentials(tokens);

        const gmail = google.gmail({ version: 'v1', auth: oauth2Client });
        const listResponse = await gmail.users.messages.list({
            userId: 'me',
            maxResults: 50,
        });

        const messages = listResponse.data.messages || [];
        const emails = [];

        for (const msg of messages) {
            const msgRes = await gmail.users.messages.get({
                userId: 'me',
                id: msg.id,
                format: 'full',
            });

            const headers = msgRes.data.payload.headers;
            const body = getBody(msgRes.data.payload);

            emails.push({
                subject: headers.find(h => h.name === 'Subject')?.value || '',
                from: headers.find(h => h.name === 'From')?.value || '',
                date: headers.find(h => h.name === 'Date')?.value || '',
                body: body,
            });
        }

        const jsonPath = path.join(dataFolder, 'emails.json');
        fs.writeFileSync(jsonPath, JSON.stringify(emails, null, 2));

        req.session.tokens = tokens;
        res.redirect('/index.html');

    } catch (err) {
        console.error('Error in Gmail callback:', err);
        res.status(500).send('Failed to get emails');
    }
});

const PORT = 4000;
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
