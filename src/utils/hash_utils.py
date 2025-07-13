"""
Hash utilities - Module centralisé pour toutes les opérations de hash
Assure la cohérence entre JavaScript et Python
"""

import hashlib


def hash_email_for_directory(email: str) -> str:
    """
    Generate a hash for email to create user-specific directories
    
    Args:
        email (str): Email address to hash
        
    Returns:
        str: 12-character SHA-256 hash prefix
        
    Example:
        hash_email_for_directory('user@example.com') -> '4a7d1ed414df'
    """
    # Normalisation identique côté JavaScript: trim + lowercase
    normalized_email = email.strip().lower()
    
    # SHA-256 hash avec troncature à 12 caractères
    return hashlib.sha256(normalized_email.encode('utf-8')).hexdigest()[:12]


def generate_email_id(from_addr: str, subject: str, date_str: str) -> str:
    """
    Generate a unique ID for an email based on metadata
    
    Args:
        from_addr (str): Sender email address
        subject (str): Email subject
        date_str (str): Date string
        
    Returns:
        str: 12-character SHA-256 hash prefix
    """
    content = f"{from_addr}-{subject}-{date_str}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]


def verify_hash_consistency(test_email=None):
    """
    Test function to verify hash consistency with any email
    
    Args:
        test_email (str, optional): Email to test. If None, uses generic test cases.
    
    Returns:
        dict: Test results showing hash consistency
    """
    if test_email:
        # Test avec l'email fourni
        base_hash = hash_email_for_directory(test_email)
        test_cases = {
            test_email: base_hash,
            test_email.upper(): base_hash,  # Should normalize to same
            f' {test_email} ': base_hash,  # Should trim spaces
        }
    else:
        # Tests de base avec des emails génériques
        test_base = 'test@example.com'
        base_hash = hash_email_for_directory(test_base)
        test_cases = {
            test_base: base_hash,
            'TEST@EXAMPLE.COM': base_hash,  # Should normalize to same
            ' test@example.com ': base_hash,  # Should trim spaces
        }
    
    results = {}
    for email, expected in test_cases.items():
        actual = hash_email_for_directory(email)
        results[email] = {
            'expected': expected,
            'actual': actual,
            'match': actual == expected
        }
    
    return results


if __name__ == "__main__":
    # Test de cohérence générique
    print("🧪 Test de cohérence des hash (générique):")
    results = verify_hash_consistency()
    
    for email, result in results.items():
        status = "✅" if result['match'] else "❌"
        print(f"{status} {email!r} -> {result['actual']} (attendu: {result['expected']})")
    
    # Test avec un email spécifique en exemple
    print("\n🧪 Test avec email spécifique:")
    specific_results = verify_hash_consistency('user@example.com')
    
    for email, result in specific_results.items():
        status = "✅" if result['match'] else "❌"
        print(f"{status} {email!r} -> {result['actual']} (attendu: {result['expected']})")