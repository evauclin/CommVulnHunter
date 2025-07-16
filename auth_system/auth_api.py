"""
Authentication API endpoints for the new authentication system
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime
import uvicorn

from .database import auth_db
from .jwt_handler import (
    jwt_handler,
    SecurityHeaders,
    rate_limiter,
    PasswordValidator,
    EmailValidator,
    audit_logger,
)

# Initialize FastAPI app
auth_app = FastAPI(
    title="CommVulnHunter Authentication API",
    description="Secure authentication system with JWT tokens",
    version="1.0.0",
)

# CORS middleware
auth_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer()


# Pydantic models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    confirm_password: str
    google_app_password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str
    confirm_password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str


class GmailValidationRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str
    created_at: str
    last_login: Optional[str]


# Utility functions
def get_client_ip(request: Request) -> str:
    """Get client IP address"""
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host


def add_security_headers(response: Response):
    """Add security headers to response"""
    headers = SecurityHeaders.get_security_headers()
    for key, value in headers.items():
        response.headers[key] = value


# Dependency to get current user
async def get_current_user(
    request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current authenticated user"""
    token = credentials.credentials
    user_data = jwt_handler.get_current_user_from_token(token)

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify user still exists and is active
    user = auth_db.get_user_by_id(user_data["id"])
    if not user or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_data


# Optional authentication dependency
async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """Get current user if authenticated, None otherwise"""
    if not credentials:
        return None

    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


# Admin role dependency
async def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Require admin role"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    return current_user


# Rate limiting middleware
@auth_app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    client_ip = get_client_ip(request)

    # Apply rate limiting to authentication endpoints
    if request.url.path in ["/auth/login", "/auth/register", "/auth/forgot-password"]:
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests. Please try again later."},
            )

    response = await call_next(request)
    add_security_headers(response)
    return response


# Authentication endpoints
@auth_app.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserRegister, request: Request):
    """Register a new user"""
    client_ip = get_client_ip(request)

    # Validate email
    if not EmailValidator.validate_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email format"
        )

    # Normalize email
    email = EmailValidator.normalize_email(user_data.email)

    # Validate password
    password_validation = PasswordValidator.validate_password(user_data.password)
    if not password_validation["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Password validation failed",
                "errors": password_validation["errors"],
            },
        )

    # Check password confirmation
    if user_data.password != user_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Passwords do not match"
        )

    # Check if user already exists
    existing_user = auth_db.get_user_by_email(email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Validate Gmail credentials before creating user
    try:
        import imaplib

        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email, user_data.google_app_password)
        mail.logout()
    except imaplib.IMAP4.error as e:
        error_msg = str(e).lower()
        if "authenticationfailed" in error_msg or "invalid credentials" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Identifiants Gmail invalides. Vérifiez votre email et mot de passe d'application Gmail.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Erreur de connexion Gmail. Veuillez vérifier vos identifiants.",
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la validation Gmail. Veuillez réessayer.",
        )

    # Create user
    user_id = auth_db.create_user(
        email=email,
        password=user_data.password,
        name=user_data.name.strip(),
        role="user",  # Default role
        google_app_password=user_data.google_app_password,
    )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user",
        )

    # Get created user
    user = auth_db.get_user_by_id(user_id)
    user.pop("password_hash", None)

    # Create session
    refresh_token_db = auth_db.create_session(
        user_id=user_id,
        ip_address=client_ip,
        user_agent=request.headers.get("User-Agent"),
    )

    # Create JWT tokens
    tokens = jwt_handler.create_token_pair(user)

    # Log registration
    audit_logger.log_event(
        event_type="user_registration",
        user_id=user_id,
        details={"email": email},
        ip_address=client_ip,
    )

    # Déclencher la récupération automatique des emails en arrière-plan
    try:
        import sys

        # Chemin absolu pour Docker
        sys.path.append("/app/src/utils")
        from gmail_fetcher import fetch_emails_from_gmail

        # Lancer la récupération en arrière-plan
        import threading

        def fetch_user_emails():
            try:
                success, message, emails_data = fetch_emails_from_gmail(
                    email, user_data.google_app_password
                )
                if success:
                    print(
                        f"✅ Emails récupérés pour {email}: {len(emails_data)} emails"
                    )
                else:
                    print(f"❌ Erreur récupération emails pour {email}: {message}")
            except Exception as e:
                print(f"❌ Erreur lors de la récupération d'emails: {e}")

        email_thread = threading.Thread(target=fetch_user_emails)
        email_thread.daemon = True
        email_thread.start()

    except Exception as e:
        print(f"⚠️ Impossible de lancer la récupération d'emails: {e}")

    # Calculer le hash de l'email pour les dossiers
    import hashlib

    normalized_email = user["email"].strip().lower()
    email_hash = hashlib.sha256(normalized_email.encode("utf-8")).hexdigest()[:12]

    return {
        **tokens,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "email_hash": email_hash,
        },
    }


@auth_app.post("/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin, request: Request):
    """Login user"""
    client_ip = get_client_ip(request)
    email = EmailValidator.normalize_email(credentials.email)

    # Authenticate user
    user = auth_db.authenticate_user(email, credentials.password)

    # Log login attempt
    audit_logger.log_login_attempt(email, user is not None, client_ip)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    # Create session
    refresh_token_db = auth_db.create_session(
        user_id=user["id"],
        ip_address=client_ip,
        user_agent=request.headers.get("User-Agent"),
    )

    # Create JWT tokens
    tokens = jwt_handler.create_token_pair(user)

    # Déclencher la mise à jour des emails en arrière-plan si Google App Password disponible
    try:
        if user.get("google_app_password"):
            import sys

            # Chemin absolu pour Docker
            sys.path.append("/app/src/utils")
            from gmail_fetcher import fetch_emails_from_gmail

            # Lancer la récupération en arrière-plan
            import threading

            def update_user_emails():
                try:
                    success, message, emails_data = fetch_emails_from_gmail(
                        email, user["google_app_password"]
                    )
                    if success:
                        print(
                            f"🔄 Emails mis à jour pour {email}: {len(emails_data)} emails"
                        )
                    else:
                        print(f"⚠️ Erreur mise à jour emails pour {email}: {message}")
                except Exception as e:
                    print(f"❌ Erreur lors de la mise à jour d'emails: {e}")

            email_thread = threading.Thread(target=update_user_emails)
            email_thread.daemon = True
            email_thread.start()
            print(f"🔄 Mise à jour des emails démarrée pour {email}")

    except Exception as e:
        print(f"⚠️ Impossible de lancer la mise à jour d'emails: {e}")

    # Calculer le hash de l'email pour les dossiers
    import hashlib

    normalized_email = user["email"].strip().lower()
    email_hash = hashlib.sha256(normalized_email.encode("utf-8")).hexdigest()[:12]

    return {
        **tokens,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "email_hash": email_hash,
        },
    }


@auth_app.post("/auth/refresh")
async def refresh_token(refresh_data: RefreshTokenRequest, request: Request):
    """Refresh access token"""
    client_ip = get_client_ip(request)

    # Verify refresh token
    payload = jwt_handler.verify_refresh_token(refresh_data.refresh_token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Get user
    user_id = int(payload["sub"])
    user = auth_db.get_user_by_id(user_id)

    if not user or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive"
        )

    # Verify session still exists
    session = auth_db.get_session(refresh_data.refresh_token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session not found or expired",
        )

    user.pop("password_hash", None)

    # Create new access token
    access_token = jwt_handler.create_access_token(
        {
            "sub": str(user["id"]),
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 30 * 60,  # 30 minutes
    }


@auth_app.post("/auth/logout")
async def logout(
    refresh_data: RefreshTokenRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Logout user"""
    client_ip = get_client_ip(request)

    # Invalidate refresh token
    auth_db.invalidate_session(refresh_data.refresh_token)

    # Log logout
    audit_logger.log_logout(current_user["id"], client_ip)

    return {"message": "Successfully logged out"}


@auth_app.post("/auth/logout-all")
async def logout_all(
    request: Request, current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Logout from all devices"""
    client_ip = get_client_ip(request)

    # Invalidate all user sessions
    auth_db.invalidate_all_user_sessions(current_user["id"])

    # Log logout all
    audit_logger.log_event(
        event_type="logout_all",
        user_id=current_user["id"],
        details={},
        ip_address=client_ip,
    )

    return {"message": "Successfully logged out from all devices"}


@auth_app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get current user information"""
    user = auth_db.get_user_by_id(current_user["id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "created_at": user["created_at"],
        "last_login": user["last_login"],
    }


@auth_app.post("/auth/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Change user password"""
    client_ip = get_client_ip(request)

    # Get user with password hash
    user = auth_db.get_user_by_id(current_user["id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Verify current password
    if not auth_db.verify_password(
        password_data.current_password, user["password_hash"]
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Validate new password
    password_validation = PasswordValidator.validate_password(
        password_data.new_password
    )
    if not password_validation["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Password validation failed",
                "errors": password_validation["errors"],
            },
        )

    # Check password confirmation
    if password_data.new_password != password_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="New passwords do not match"
        )

    # Update password
    auth_db.update_password(current_user["id"], password_data.new_password)

    # Invalidate all sessions except current one
    auth_db.invalidate_all_user_sessions(current_user["id"])

    # Log password change
    audit_logger.log_password_change(current_user["id"], client_ip)

    return {"message": "Password changed successfully"}


@auth_app.post("/auth/forgot-password")
async def forgot_password(request_data: PasswordResetRequest, request: Request):
    """Request password reset"""
    client_ip = get_client_ip(request)
    email = EmailValidator.normalize_email(request_data.email)

    # Check if user exists
    user = auth_db.get_user_by_email(email)
    if not user:
        # Don't reveal if user exists or not
        return {"message": "If the email exists, a reset link has been sent"}

    # Create password reset token
    reset_token = auth_db.create_password_reset_token(user["id"])

    # Log password reset request
    audit_logger.log_event(
        event_type="password_reset_request",
        user_id=user["id"],
        details={"email": email},
        ip_address=client_ip,
    )

    # In a real application, you would send an email here
    # For now, we'll just return the token (remove this in production)
    return {
        "message": "Password reset token created",
        "reset_token": reset_token,  # Remove this line in production
    }


@auth_app.post("/auth/reset-password")
async def reset_password(reset_data: PasswordResetConfirm, request: Request):
    """Reset password with token"""
    client_ip = get_client_ip(request)

    # Verify reset token
    token_data = auth_db.get_password_reset_token(reset_data.token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Validate new password
    password_validation = PasswordValidator.validate_password(reset_data.new_password)
    if not password_validation["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Password validation failed",
                "errors": password_validation["errors"],
            },
        )

    # Check password confirmation
    if reset_data.new_password != reset_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Passwords do not match"
        )

    # Update password
    auth_db.update_password(token_data["user_id"], reset_data.new_password)

    # Mark token as used
    auth_db.use_password_reset_token(reset_data.token)

    # Invalidate all user sessions
    auth_db.invalidate_all_user_sessions(token_data["user_id"])

    # Log password reset
    audit_logger.log_event(
        event_type="password_reset_completed",
        user_id=token_data["user_id"],
        details={},
        ip_address=client_ip,
    )

    return {"message": "Password reset successfully"}


@auth_app.post("/auth/validate-gmail")
async def validate_gmail_credentials(
    validation_data: GmailValidationRequest, request: Request
):
    """Validate Gmail credentials"""
    client_ip = get_client_ip(request)

    # Normalize email
    email = EmailValidator.normalize_email(validation_data.email)

    # Validate Gmail password length
    if len(validation_data.password) != 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gmail app password must be exactly 16 characters",
        )

    try:
        # Import gmail fetcher and test connection
        import sys

        # Chemin absolu pour Docker
        sys.path.append("/app/src/utils")

        # Test connection without fetching emails (by limiting to 1 email)
        import imaplib

        # Quick test connection
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email, validation_data.password)
        mail.logout()

        # Log validation attempt
        audit_logger.log_event(
            event_type="gmail_validation",
            user_id=None,  # Pas d'utilisateur connecté lors de la validation
            details={"email": email, "success": True},
            ip_address=client_ip,
        )

        return {"valid": True, "message": "Gmail credentials are valid"}

    except imaplib.IMAP4.error as e:
        # Log failed validation
        audit_logger.log_event(
            event_type="gmail_validation",
            user_id=None,  # Pas d'utilisateur connecté lors de la validation
            details={"email": email, "success": False, "error": str(e)},
            ip_address=client_ip,
        )

        error_msg = str(e).lower()
        if "authenticationfailed" in error_msg or "invalid credentials" in error_msg:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "valid": False,
                    "message": "Identifiants Gmail invalides. Vérifiez votre email et mot de passe d'application.",
                },
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "valid": False,
                    "message": "Erreur de connexion Gmail. Veuillez réessayer.",
                },
            )
    except Exception as e:
        # Log failed validation
        audit_logger.log_event(
            event_type="gmail_validation",
            user_id=None,  # Pas d'utilisateur connecté lors de la validation
            details={"email": email, "success": False, "error": str(e)},
            ip_address=client_ip,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "valid": False,
                "message": "Erreur lors de la validation. Veuillez réessayer.",
            },
        )


# Admin endpoints
@auth_app.get("/auth/admin/users")
async def get_all_users(
    limit: int = 100,
    offset: int = 0,
    admin_user: Dict[str, Any] = Depends(require_admin),
):
    """Get all users (admin only)"""
    users = auth_db.get_all_users(limit, offset)
    return {"users": users, "total": len(users)}


@auth_app.put("/auth/admin/users/{user_id}/role")
async def update_user_role(
    user_id: int, new_role: str, admin_user: Dict[str, Any] = Depends(require_admin)
):
    """Update user role (admin only)"""
    if new_role not in ["admin", "user", "demo"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role"
        )

    user = auth_db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    auth_db.update_user_role(user_id, new_role)

    return {"message": f"User role updated to {new_role}"}


@auth_app.delete("/auth/admin/users/{user_id}")
async def deactivate_user(
    user_id: int, admin_user: Dict[str, Any] = Depends(require_admin)
):
    """Deactivate user (admin only)"""
    user = auth_db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Prevent admin from deactivating themselves
    if user_id == admin_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account",
        )

    auth_db.deactivate_user(user_id)

    return {"message": "User deactivated successfully"}


# Health check
@auth_app.get("/auth/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Authentication API",
    }


# Token verification for nginx auth_request
@auth_app.get("/auth/verify")
async def verify_token(
    request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Verify JWT token for nginx auth_request"""
    try:
        # Extract token from Bearer authorization
        token = credentials.credentials

        # Validate token
        payload = jwt_handler.verify_token(token)
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Check if user exists and is active
        user = auth_db.get_user_by_id(user_id)
        if not user or not user.get("is_active", True):
            raise HTTPException(status_code=401, detail="User not found or inactive")

        # Return 200 OK for nginx
        return Response(status_code=200)

    except Exception:
        # Return 401 for any authentication failure
        raise HTTPException(status_code=401, detail="Authentication failed")


# Cleanup expired sessions periodically
@auth_app.on_event("startup")
async def startup_event():
    """Cleanup expired sessions on startup"""
    auth_db.cleanup_expired_sessions()


if __name__ == "__main__":
    uvicorn.run("auth_api:auth_app", host="127.0.0.1", port=9000, reload=True)
