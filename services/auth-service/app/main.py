#!/usr/bin/env python3
"""
FinSim Auth Service

OAuth2 JWT authentication service with role-based access control (RBAC).
Provides secure authentication and authorization for the FinSim platform.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
import jwt
from passlib.context import CryptContext
import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import sessionmaker, Session
from contextlib import asynccontextmanager
import uuid
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
security = HTTPBearer()

# User Roles and Permissions
class UserRole(str, Enum):
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"

class Permission(str, Enum):
    # Trading permissions
    TRADE_WRITE = "trade:write"
    TRADE_READ = "trade:read"
    
    # Portfolio permissions
    PORTFOLIO_WRITE = "portfolio:write"
    PORTFOLIO_READ = "portfolio:read"
    
    # Risk permissions
    RISK_READ = "risk:read"
    RISK_WRITE = "risk:write"
    
    # Market data permissions
    MARKET_DATA_READ = "market_data:read"
    
    # Admin permissions
    USER_ADMIN = "user:admin"
    SYSTEM_ADMIN = "system:admin"

# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.TRADE_WRITE, Permission.TRADE_READ,
        Permission.PORTFOLIO_WRITE, Permission.PORTFOLIO_READ,
        Permission.RISK_READ, Permission.RISK_WRITE,
        Permission.MARKET_DATA_READ,
        Permission.USER_ADMIN, Permission.SYSTEM_ADMIN
    ],
    UserRole.TRADER: [
        Permission.TRADE_WRITE, Permission.TRADE_READ,
        Permission.PORTFOLIO_WRITE, Permission.PORTFOLIO_READ,
        Permission.RISK_READ, Permission.MARKET_DATA_READ
    ],
    UserRole.ANALYST: [
        Permission.TRADE_READ, Permission.PORTFOLIO_READ,
        Permission.RISK_READ, Permission.RISK_WRITE,
        Permission.MARKET_DATA_READ
    ],
    UserRole.VIEWER: [
        Permission.TRADE_READ, Permission.PORTFOLIO_READ,
        Permission.RISK_READ, Permission.MARKET_DATA_READ
    ]
}

# Data Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: UserRole = UserRole.VIEWER
    is_active: bool = True

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    permissions: List[str] = []

class LoginRequest(BaseModel):
    email: str
    password: str

# Database Models
metadata = MetaData()

users_table = Table(
    'users',
    metadata,
    Column('id', String, primary_key=True),
    Column('email', String, unique=True, index=True),
    Column('hashed_password', String),
    Column('full_name', String),
    Column('role', String),
    Column('is_active', Boolean, default=True),
    Column('created_at', DateTime, default=datetime.utcnow),
    Column('last_login', DateTime, nullable=True)
)

sessions_table = Table(
    'user_sessions',
    metadata,
    Column('id', String, primary_key=True),
    Column('user_id', String, ForeignKey('users.id')),
    Column('refresh_token', String, unique=True),
    Column('expires_at', DateTime),
    Column('created_at', DateTime, default=datetime.utcnow),
    Column('is_active', Boolean, default=True)
)

# Auth utilities
class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self, db_engine):
        self.db_engine = db_engine
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            email: str = payload.get("email")
            role: str = payload.get("role")
            
            if user_id is None:
                return None
            
            # Get permissions for role
            permissions = [p.value for p in ROLE_PERMISSIONS.get(UserRole(role), [])]
            
            token_data = TokenData(
                user_id=user_id,
                email=email,
                permissions=permissions
            )
            return token_data
        except jwt.PyJWTError:
            return None
    
    async def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Authenticate user with email and password"""
        with self.SessionLocal() as db:
            # Get user from database
            query = users_table.select().where(users_table.c.email == email)
            result = db.execute(query)
            user = result.fetchone()
            
            if not user:
                return None
            
            if not self.verify_password(password, user.hashed_password):
                return None
            
            if not user.is_active:
                return None
            
            # Update last login
            update_query = users_table.update().where(
                users_table.c.id == user.id
            ).values(last_login=datetime.utcnow())
            db.execute(update_query)
            db.commit()
            
            return {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "is_active": user.is_active
            }
    
    async def create_user(self, user_data: UserCreate) -> str:
        """Create new user"""
        with self.SessionLocal() as db:
            # Check if user already exists
            query = users_table.select().where(users_table.c.email == user_data.email)
            result = db.execute(query)
            if result.fetchone():
                raise HTTPException(
                    status_code=400,
                    detail="User with this email already exists"
                )
            
            # Create user
            user_id = str(uuid.uuid4())
            hashed_password = self.get_password_hash(user_data.password)
            
            insert_query = users_table.insert().values(
                id=user_id,
                email=user_data.email,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                role=user_data.role.value,
                is_active=user_data.is_active
            )
            
            db.execute(insert_query)
            db.commit()
            
            return user_id
    
    async def get_user(self, user_id: str) -> Optional[dict]:
        """Get user by ID"""
        with self.SessionLocal() as db:
            query = users_table.select().where(users_table.c.id == user_id)
            result = db.execute(query)
            user = result.fetchone()
            
            if user:
                return {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role,
                    "is_active": user.is_active,
                    "created_at": user.created_at,
                    "last_login": user.last_login
                }
            return None
    
    def has_permission(self, user_role: UserRole, required_permission: Permission) -> bool:
        """Check if user role has required permission"""
        user_permissions = ROLE_PERMISSIONS.get(user_role, [])
        return required_permission in user_permissions

# Global variables
auth_manager = None
db_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global auth_manager, db_engine
    
    # Startup
    logger.info("Starting Auth Service...")
    
    # Initialize database connection
    db_engine = create_engine("postgresql://finsim:finsim123@postgres:5432/finsim")
    
    # Create tables
    metadata.create_all(db_engine)
    
    # Initialize auth manager
    auth_manager = AuthManager(db_engine)
    
    # Create default admin user if not exists
    try:
        await create_default_admin()
    except Exception as e:
        logger.warning(f"Could not create default admin: {e}")
    
    logger.info("Auth Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Auth Service...")
    if db_engine:
        db_engine.dispose()

async def create_default_admin():
    """Create default admin user"""
    admin_data = UserCreate(
        email="admin@finsim.com",
        password="admin123",
        full_name="Default Admin",
        role=UserRole.ADMIN
    )
    
    try:
        await auth_manager.create_user(admin_data)
        logger.info("Default admin user created: admin@finsim.com / admin123")
    except HTTPException:
        logger.info("Default admin user already exists")

app = FastAPI(
    title="FinSim Auth Service",
    description="Authentication and authorization service",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for getting current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Get current authenticated user"""
    token = credentials.credentials
    token_data = auth_manager.verify_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token_data

def require_permission(permission: Permission):
    """Decorator factory for requiring specific permissions"""
    def permission_checker(current_user: TokenData = Depends(get_current_user)):
        if permission.value not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission.value}"
            )
        return current_user
    return permission_checker

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/api/v1/auth/register", response_model=dict)
async def register(user_data: UserCreate):
    """Register new user"""
    try:
        user_id = await auth_manager.create_user(user_data)
        return {"message": "User created successfully", "user_id": user_id}
    
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/auth/login", response_model=Token)
async def login(login_data: LoginRequest):
    """Login user and return tokens"""
    try:
        user = await auth_manager.authenticate_user(login_data.email, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Create tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_manager.create_access_token(
            data={"sub": user["id"], "email": user["email"], "role": user["role"]},
            expires_delta=access_token_expires
        )
        
        refresh_token = auth_manager.create_refresh_token(
            data={"sub": user["id"], "email": user["email"], "role": user["role"]}
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/v1/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token endpoint"""
    try:
        user = await auth_manager.authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_manager.create_access_token(
            data={"sub": user["id"], "email": user["email"], "role": user["role"]},
            expires_delta=access_token_expires
        )
        
        refresh_token = auth_manager.create_refresh_token(
            data={"sub": user["id"], "email": user["email"], "role": user["role"]}
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during token request: {e}")
        raise HTTPException(status_code=500, detail="Token request failed")

@app.get("/api/v1/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information"""
    try:
        user = await auth_manager.get_user(current_user.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(**user)
    
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/auth/permissions")
async def get_user_permissions(current_user: TokenData = Depends(get_current_user)):
    """Get current user permissions"""
    return {
        "user_id": current_user.user_id,
        "permissions": current_user.permissions
    }

@app.post("/api/v1/auth/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    try:
        token_data = auth_manager.verify_token(refresh_token)
        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user = await auth_manager.get_user(token_data.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Create new tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_manager.create_access_token(
            data={"sub": user["id"], "email": user["email"], "role": user["role"]},
            expires_delta=access_token_expires
        )
        
        new_refresh_token = auth_manager.create_refresh_token(
            data={"sub": user["id"], "email": user["email"], "role": user["role"]}
        )
        
        return Token(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

@app.get("/api/v1/auth/users")
async def list_users(current_user: TokenData = Depends(require_permission(Permission.USER_ADMIN))):
    """List all users (admin only)"""
    try:
        with auth_manager.SessionLocal() as db:
            query = users_table.select()
            result = db.execute(query)
            users = result.fetchall()
            
            return [
                {
                    "id": user.id,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role,
                    "is_active": user.is_active,
                    "created_at": user.created_at,
                    "last_login": user.last_login
                }
                for user in users
            ]
    
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/auth/check-permission")
async def check_permission(permission: str, current_user: TokenData = Depends(get_current_user)):
    """Check if current user has specific permission"""
    has_permission = permission in current_user.permissions
    
    return {
        "user_id": current_user.user_id,
        "permission": permission,
        "granted": has_permission
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )