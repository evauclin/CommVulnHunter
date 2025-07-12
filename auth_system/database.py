"""
Database models and connection for authentication system
"""
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import secrets
from typing import Optional, Dict, Any
import bcrypt

DATABASE_PATH = Path("auth_system/auth.db")


class AuthDatabase:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        DATABASE_PATH.parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP
                )
            """)
            
            # Sessions table for JWT refresh tokens
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    refresh_token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # Password reset tokens
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    used BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            conn.commit()
        
        # Create default admin user if not exists
        self.create_default_users()
    
    def create_default_users(self):
        """Create default users for testing"""
        default_users = [
            {
                "email": "admin@emailfilter.com",
                "password": "admin123",
                "name": "Administrateur",
                "role": "admin"
            },
            {
                "email": "user@emailfilter.com", 
                "password": "user123",
                "name": "Utilisateur Standard",
                "role": "user"
            },
            {
                "email": "demo@emailfilter.com",
                "password": "demo123", 
                "name": "Compte Démo",
                "role": "demo"
            }
        ]
        
        for user_data in default_users:
            if not self.get_user_by_email(user_data["email"]):
                self.create_user(
                    email=user_data["email"],
                    password=user_data["password"],
                    name=user_data["name"],
                    role=user_data["role"]
                )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create_user(self, email: str, password: str, name: str, role: str = "user") -> Optional[int]:
        """Create a new user"""
        try:
            password_hash = self.hash_password(password)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (email, password_hash, name, role)
                    VALUES (?, ?, ?, ?)
                """, (email, password_hash, name, role))
                
                user_id = cursor.lastrowid
                conn.commit()
                return user_id
                
        except sqlite3.IntegrityError:
            return None  # User already exists
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, email, password_hash, name, role, is_active, 
                       created_at, last_login, failed_login_attempts, locked_until
                FROM users 
                WHERE email = ?
            """, (email,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, email, password_hash, name, role, is_active,
                       created_at, last_login, failed_login_attempts, locked_until
                FROM users 
                WHERE id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with email and password"""
        user = self.get_user_by_email(email)
        if not user:
            return None
        
        # Check if account is locked
        if user['locked_until']:
            locked_until = datetime.fromisoformat(user['locked_until'])
            if datetime.now() < locked_until:
                return None
        
        # Check if account is active
        if not user['is_active']:
            return None
        
        # Verify password
        if not self.verify_password(password, user['password_hash']):
            self.increment_failed_login(user['id'])
            return None
        
        # Reset failed login attempts and update last login
        self.reset_failed_login(user['id'])
        self.update_last_login(user['id'])
        
        # Remove password hash from returned data
        user.pop('password_hash', None)
        return user
    
    def increment_failed_login(self, user_id: int):
        """Increment failed login attempts and lock account if necessary"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users 
                SET failed_login_attempts = failed_login_attempts + 1
                WHERE id = ?
            """, (user_id,))
            
            # Lock account after 5 failed attempts for 30 minutes
            cursor.execute("""
                UPDATE users 
                SET locked_until = datetime('now', '+30 minutes')
                WHERE id = ? AND failed_login_attempts >= 5
            """, (user_id,))
            
            conn.commit()
    
    def reset_failed_login(self, user_id: int):
        """Reset failed login attempts"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users 
                SET failed_login_attempts = 0, locked_until = NULL
                WHERE id = ?
            """, (user_id,))
            conn.commit()
    
    def update_last_login(self, user_id: int):
        """Update last login timestamp"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (user_id,))
            conn.commit()
    
    def create_session(self, user_id: int, ip_address: str = None, user_agent: str = None) -> str:
        """Create a new session with refresh token"""
        refresh_token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=30)  # 30 days for refresh token
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_sessions (user_id, refresh_token, expires_at, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, refresh_token, expires_at, ip_address, user_agent))
            conn.commit()
        
        return refresh_token
    
    def get_session(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Get session by refresh token"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, u.email, u.name, u.role
                FROM user_sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.refresh_token = ? AND s.is_active = 1 AND s.expires_at > datetime('now')
            """, (refresh_token,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def invalidate_session(self, refresh_token: str):
        """Invalidate a session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_sessions 
                SET is_active = 0
                WHERE refresh_token = ?
            """, (refresh_token,))
            conn.commit()
    
    def invalidate_all_user_sessions(self, user_id: int):
        """Invalidate all sessions for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_sessions 
                SET is_active = 0
                WHERE user_id = ?
            """, (user_id,))
            conn.commit()
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM user_sessions 
                WHERE expires_at < datetime('now')
            """)
            conn.commit()
    
    def create_password_reset_token(self, user_id: int) -> str:
        """Create password reset token"""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=1)  # 1 hour expiry
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO password_reset_tokens (user_id, token, expires_at)
                VALUES (?, ?, ?)
            """, (user_id, token, expires_at))
            conn.commit()
        
        return token
    
    def get_password_reset_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Get password reset token"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM password_reset_tokens
                WHERE token = ? AND used = 0 AND expires_at > datetime('now')
            """, (token,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def use_password_reset_token(self, token: str):
        """Mark password reset token as used"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE password_reset_tokens 
                SET used = 1
                WHERE token = ?
            """, (token,))
            conn.commit()
    
    def update_password(self, user_id: int, new_password: str):
        """Update user password"""
        password_hash = self.hash_password(new_password)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users 
                SET password_hash = ?
                WHERE id = ?
            """, (password_hash, user_id))
            conn.commit()
    
    def get_all_users(self, limit: int = 100, offset: int = 0) -> list:
        """Get all users (for admin interface)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, email, name, role, is_active, created_at, last_login
                FROM users
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def update_user_role(self, user_id: int, role: str):
        """Update user role"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users 
                SET role = ?
                WHERE id = ?
            """, (role, user_id))
            conn.commit()
    
    def deactivate_user(self, user_id: int):
        """Deactivate user account"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users 
                SET is_active = 0
                WHERE id = ?
            """, (user_id,))
            conn.commit()
        
        # Invalidate all sessions
        self.invalidate_all_user_sessions(user_id)


# Global instance
auth_db = AuthDatabase()