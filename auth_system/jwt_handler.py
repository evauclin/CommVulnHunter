"""
JWT Token handler for authentication system
"""
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
from pathlib import Path
import json

# JWT Configuration
JWT_SECRET_KEY = "your-super-secret-jwt-key-change-in-production"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 30

class JWTHandler:
    def __init__(self):
        self.secret_key = JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
        self.access_token_expire = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        self.refresh_token_expire = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + self.access_token_expire
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + self.refresh_token_expire
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
    
    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify access token"""
        payload = self.decode_token(token)
        if payload and payload.get("type") == "access":
            return payload
        return None
    
    def verify_refresh_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify refresh token"""
        payload = self.decode_token(token)
        if payload and payload.get("type") == "refresh":
            return payload
        return None
    
    def extract_token_from_header(self, authorization: str) -> Optional[str]:
        """Extract token from Authorization header"""
        if not authorization:
            return None
        
        try:
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                return None
            return token
        except ValueError:
            return None
    
    def create_token_pair(self, user_data: Dict[str, Any]) -> Dict[str, str]:
        """Create both access and refresh tokens"""
        # Prepare token payload
        token_data = {
            "sub": str(user_data["id"]),
            "email": user_data["email"],
            "name": user_data["name"],
            "role": user_data["role"]
        }
        
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(token_data)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
        }
    
    def get_current_user_from_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Get current user info from valid access token"""
        payload = self.verify_access_token(token)
        if payload:
            return {
                "id": int(payload["sub"]),
                "email": payload["email"],
                "name": payload["name"],
                "role": payload["role"]
            }
        return None


# Global JWT handler instance
jwt_handler = JWTHandler()


class SecurityHeaders:
    """Security headers for HTTP responses"""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get recommended security headers"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' cdn.jsdelivr.net; font-src 'self' cdn.jsdelivr.net; img-src 'self' data:;",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = {}
        self.max_requests = 10  # Max requests per minute
        self.window_size = 60   # 1 minute window
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for this IP"""
        current_time = datetime.now().timestamp()
        
        # Clean old entries
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < self.window_size
            ]
        
        # Check current request count
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_ip].append(current_time)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


class PasswordValidator:
    """Password strength validator"""
    
    @staticmethod
    def validate_password(password: str) -> Dict[str, Any]:
        """Validate password strength"""
        errors = []
        
        if len(password) < 8:
            errors.append("Le mot de passe doit contenir au moins 8 caractères")
        
        if not any(c.isupper() for c in password):
            errors.append("Le mot de passe doit contenir au moins une majuscule")
        
        if not any(c.islower() for c in password):
            errors.append("Le mot de passe doit contenir au moins une minuscule")
        
        if not any(c.isdigit() for c in password):
            errors.append("Le mot de passe doit contenir au moins un chiffre")
        
        # Check for special characters
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            errors.append("Le mot de passe doit contenir au moins un caractère spécial")
        
        # Check for common passwords
        common_passwords = [
            "password", "123456", "123456789", "qwerty", "abc123",
            "password123", "admin", "user", "test", "demo"
        ]
        
        if password.lower() in common_passwords:
            errors.append("Ce mot de passe est trop commun")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "strength_score": max(0, 100 - len(errors) * 20)
        }


class EmailValidator:
    """Email validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Basic email validation"""
        import re
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def normalize_email(email: str) -> str:
        """Normalize email address"""
        return email.lower().strip()


class AuditLogger:
    """Security audit logger"""
    
    def __init__(self):
        self.log_file = Path("auth_system/audit.log")
        self.log_file.parent.mkdir(exist_ok=True)
    
    def log_event(self, event_type: str, user_id: Optional[int], details: Dict[str, Any], ip_address: str = None):
        """Log security event"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Failed to write audit log: {e}")
    
    def log_login_attempt(self, email: str, success: bool, ip_address: str = None):
        """Log login attempt"""
        self.log_event(
            event_type="login_attempt",
            user_id=None,
            details={"email": email, "success": success},
            ip_address=ip_address
        )
    
    def log_logout(self, user_id: int, ip_address: str = None):
        """Log logout"""
        self.log_event(
            event_type="logout",
            user_id=user_id,
            details={},
            ip_address=ip_address
        )
    
    def log_password_change(self, user_id: int, ip_address: str = None):
        """Log password change"""
        self.log_event(
            event_type="password_change",
            user_id=user_id,
            details={},
            ip_address=ip_address
        )
    
    def log_account_lockout(self, user_id: int, ip_address: str = None):
        """Log account lockout"""
        self.log_event(
            event_type="account_lockout",
            user_id=user_id,
            details={},
            ip_address=ip_address
        )


# Global audit logger instance
audit_logger = AuditLogger()