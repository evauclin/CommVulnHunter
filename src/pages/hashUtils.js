console.log("🔥 FORCE CACHE CLEAR - hashUtils.js v3.0");
/**
 * Hash utilities - Module centralisé pour toutes les opérations de hash
 * Assure la cohérence avec Python (hash_utils.py)
 */

/**
 * Generate a hash for email to create user-specific directories
 * IDENTICAL implementation to Python hash_utils.hash_email_for_directory()
 * 
 * @param {string} email - Email address to hash
 * @returns {Promise<string>} 12-character SHA-256 hash prefix
 * 
 * @example
 * await hashEmailForDirectory('user@example.com') // -> '4a7d1ed414df'
 */
async function hashEmailForDirectory(email) {
    // Normalisation identique côté Python: trim + lowercase
    const normalizedEmail = email.trim().toLowerCase();
    
    // SHA-256 hash avec troncature à 12 caractères
    const encoder = new TextEncoder();
    const data = encoder.encode(normalizedEmail);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    const result = hashHex.substring(0, 12);
    
    console.log(`🔍 hashEmailForDirectory("${email}") -> "${result}"`);
    return result;
}

/**
 * Generate a unique ID for an email based on metadata
 * IDENTICAL implementation to Python hash_utils.generate_email_id()
 * 
 * @param {string} fromAddr - Sender email address
 * @param {string} subject - Email subject
 * @param {string} dateStr - Date string
 * @returns {Promise<string>} 12-character SHA-256 hash prefix
 */
async function generateEmailId(fromAddr, subject, dateStr) {
    const content = `${fromAddr}-${subject}-${dateStr}`;
    
    const encoder = new TextEncoder();
    const data = encoder.encode(content);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    
    return hashHex.substring(0, 12);
}

/**
 * Test function to verify hash consistency with any email
 * @param {string} testEmail - Email to test (optional, defaults to test emails)
 * @returns {Promise<Object>} Test results
 */
async function verifyHashConsistency(testEmail = null) {
    let testCases = {};
    
    if (testEmail) {
        // Test avec l'email fourni
        const baseHash = await hashEmailForDirectory(testEmail);
        testCases = {
            [testEmail]: baseHash,
            [testEmail.toUpperCase()]: baseHash,  // Should normalize to same
            [` ${testEmail} `]: baseHash,  // Should trim spaces
        };
    } else {
        // Tests de base avec des emails génériques
        const testBase = 'test@example.com';
        const baseHash = await hashEmailForDirectory(testBase);
        testCases = {
            [testBase]: baseHash,
            'TEST@EXAMPLE.COM': baseHash,  // Should normalize to same
            ' test@example.com ': baseHash,  // Should trim spaces
        };
    }
    
    const results = {};
    
    for (const [email, expected] of Object.entries(testCases)) {
        const actual = await hashEmailForDirectory(email);
        results[email] = {
            expected: expected,
            actual: actual,
            match: actual === expected
        };
    }
    
    return results;
}

/**
 * Legacy function name for backward compatibility
 * @deprecated Use hashEmailForDirectory instead
 */
async function hashEmail(email) {
    console.warn('⚠️ hashEmail() is deprecated. Use hashEmailForDirectory() instead.');
    return await hashEmailForDirectory(email);
}

// Auto-test on load - dynamique avec l'utilisateur connecté
if (typeof window !== 'undefined') {
    window.addEventListener('DOMContentLoaded', async () => {
        console.log('🧪 Test de cohérence des hash JavaScript (dynamique):');
        
        // Test avec l'utilisateur connecté s'il existe
        let testEmail = null;
        try {
            const userData = localStorage.getItem('user_data');
            if (userData) {
                const user = JSON.parse(userData);
                if (user.email) {
                    testEmail = user.email;
                }
            }
        } catch (e) {
            // Ignore, utilise le test de base
        }
        
        const results = await verifyHashConsistency(testEmail);
        
        for (const [email, result] of Object.entries(results)) {
            const status = result.match ? '✅' : '❌';
            console.log(`${status} "${email}" -> ${result.actual} (attendu: ${result.expected})`);
        }
    });
}

// Expose functions globally for browser use
if (typeof window !== 'undefined') {
    window.hashEmailForDirectory = hashEmailForDirectory;
    window.generateEmailId = generateEmailId;
    window.verifyHashConsistency = verifyHashConsistency;
    window.hashEmail = hashEmail; // deprecated
    console.log('🔧 Hash functions exposed globally:', {
        hashEmailForDirectory: typeof window.hashEmailForDirectory,
        generateEmailId: typeof window.generateEmailId,
        verifyHashConsistency: typeof window.verifyHashConsistency,
        hashEmail: typeof window.hashEmail
    });
}

// Export functions for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        hashEmailForDirectory,
        generateEmailId,
        verifyHashConsistency,
        hashEmail // deprecated
    };
}