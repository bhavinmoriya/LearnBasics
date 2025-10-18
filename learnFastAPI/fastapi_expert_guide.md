# FastAPI: Zero to Expert Complete Guide

## Table of Contents
1. [Introduction & Setup](#introduction--setup)
2. [Level 1: Fundamentals](#level-1-fundamentals)
3. [Level 2: Request & Response Handling](#level-2-request--response-handling)
4. [Level 3: Data Validation & Models](#level-3-data-validation--models)
5. [Level 4: Database Integration](#level-4-database-integration)
6. [Level 5: Authentication & Security](#level-5-authentication--security)
7. [Level 6: Advanced Features](#level-6-advanced-features)
8. [Level 7: Production & Deployment](#level-7-production--deployment)
9. [Expert Projects](#expert-projects)

---

## Introduction & Setup

### What is FastAPI?
FastAPI is a modern, high-performance Python web framework for building APIs. It's based on standard Python type hints and provides:
- Automatic API documentation (Swagger UI)
- Data validation using Pydantic
- Async support
- High performance (comparable to NodeJS and Go)

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install FastAPI and Uvicorn (ASGI server)
pip install fastapi uvicorn[standard]
pip install pydantic[email]  # Extended Pydantic features
```

---

## Level 1: Fundamentals

### Your First FastAPI Application

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

**Run the server:**
```bash
uvicorn main:app --reload
```

**Access:**
- API: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs
- Alternative docs: http://127.0.0.1:8000/redoc

### HTTP Methods

```python
from fastapi import FastAPI

app = FastAPI()

# GET - Retrieve data
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id, "name": "John"}

# POST - Create data
@app.post("/users")
def create_user(name: str, email: str):
    return {"name": name, "email": email, "id": 123}

# PUT - Update entire resource
@app.put("/users/{user_id}")
def update_user(user_id: int, name: str, email: str):
    return {"user_id": user_id, "name": name, "email": email}

# PATCH - Partial update
@app.patch("/users/{user_id}")
def patch_user(user_id: int, name: str = None):
    return {"user_id": user_id, "updated_field": "name"}

# DELETE - Remove data
@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    return {"message": f"User {user_id} deleted"}
```

### Path Parameters & Query Parameters

```python
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

# Path parameters (required)
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

# Query parameters (optional)
@app.get("/search")
def search(q: Optional[str] = None, limit: int = 10):
    return {"query": q, "limit": limit}

# Mixed
@app.get("/users/{user_id}/items")
def get_user_items(user_id: int, skip: int = 0, limit: int = 10):
    return {
        "user_id": user_id,
        "skip": skip,
        "limit": limit
    }

# Multiple path parameters
@app.get("/users/{user_id}/posts/{post_id}")
def get_user_post(user_id: int, post_id: int):
    return {"user_id": user_id, "post_id": post_id}
```

---

## Level 2: Request & Response Handling

### Request Body with Pydantic Models

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.post("/items")
def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

# Multiple body parameters
class User(BaseModel):
    username: str
    email: str

@app.post("/items-with-user")
def create_item_with_user(item: Item, user: User, importance: int):
    return {"item": item, "user": user, "importance": importance}
```

### Response Models

```python
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr
from typing import List

app = FastAPI()

class UserIn(BaseModel):
    username: str
    password: str
    email: EmailStr

class UserOut(BaseModel):
    username: str
    email: EmailStr

# Response model excludes password
@app.post("/users", response_model=UserOut)
def create_user(user: UserIn):
    return user

# List response
@app.get("/users", response_model=List[UserOut])
def list_users():
    return [
        {"username": "john", "email": "john@example.com"},
        {"username": "jane", "email": "jane@example.com"}
    ]
```

### Status Codes

```python
from fastapi import FastAPI, status

app = FastAPI()

@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(name: str):
    return {"name": name}

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int):
    return None

@app.get("/items/{item_id}", status_code=status.HTTP_200_OK)
def read_item(item_id: int):
    return {"item_id": item_id}
```

### Error Handling

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

items = {"foo": "The Foo Wrestlers"}

@app.get("/items/{item_id}")
def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found",
            headers={"X-Error": "Custom header"}
        )
    return {"item": items[item_id]}

# Custom exception handler
from fastapi.responses import JSONResponse
from fastapi import Request

class CustomException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something wrong"}
    )

@app.get("/custom/{name}")
async def read_custom(name: str):
    if name == "error":
        raise CustomException(name=name)
    return {"name": name}
```

---

## Level 3: Data Validation & Models

### Advanced Pydantic Models

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum

app = FastAPI()

class CategoryEnum(str, Enum):
    electronics = "electronics"
    clothing = "clothing"
    food = "food"

class Product(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    description: Optional[str] = Field(None, max_length=300)
    price: float = Field(..., gt=0, description="Price must be greater than 0")
    tax: float = Field(0.0, ge=0, le=1)
    tags: List[str] = []
    category: CategoryEnum
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('name')
    def name_must_not_contain_space_at_ends(cls, v):
        if v.strip() != v:
            raise ValueError('name must not have leading/trailing spaces')
        return v
    
    @validator('price')
    def check_price_tax_relationship(cls, v, values):
        if 'tax' in values and v < values['tax']:
            raise ValueError('price must be greater than tax')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Laptop",
                "description": "A powerful laptop",
                "price": 1299.99,
                "tax": 0.15,
                "tags": ["electronics", "computers"],
                "category": "electronics"
            }
        }

@app.post("/products")
def create_product(product: Product):
    return product
```

### Nested Models

```python
from typing import List, Optional
from pydantic import BaseModel

class Image(BaseModel):
    url: str
    name: str

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    images: Optional[List[Image]] = None

class Offer(BaseModel):
    name: str
    description: Optional[str] = None
    items: List[Item]

@app.post("/offers")
def create_offer(offer: Offer):
    return offer

# Example request body:
# {
#   "name": "Summer Sale",
#   "items": [
#     {
#       "name": "Laptop",
#       "price": 999.99,
#       "images": [
#         {"url": "http://example.com/img1.jpg", "name": "Front view"}
#       ]
#     }
#   ]
# }
```

### Form Data & File Uploads

```python
from fastapi import FastAPI, Form, File, UploadFile
from typing import List

app = FastAPI()

# Form data
@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    return {"username": username}

# File upload
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents)
    }

# Multiple files
@app.post("/uploadfiles")
async def upload_files(files: List[UploadFile] = File(...)):
    return {
        "filenames": [file.filename for file in files]
    }

# Mixed form and file
@app.post("/upload-with-data")
async def upload_with_data(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(...)
):
    return {
        "filename": file.filename,
        "name": name,
        "description": description
    }
```

---

## Level 4: Database Integration

### SQLAlchemy Setup

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
# For PostgreSQL: "postgresql://user:password@localhost/dbname"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # Only for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Models and Schemas

```python
# models.py
from sqlalchemy import Boolean, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    
    items = relationship("Item", back_populates="owner")

class Item(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    price = Column(Float)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="items")

# schemas.py
from pydantic import BaseModel
from typing import List, Optional

class ItemBase(BaseModel):
    title: str
    description: Optional[str] = None
    price: float

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    owner_id: int
    
    class Config:
        orm_mode = True

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    items: List[Item] = []
    
    class Config:
        orm_mode = True
```

### CRUD Operations

```python
# crud.py
from sqlalchemy.orm import Session
import models, schemas

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = models.User(
        email=user.email,
        hashed_password=fake_hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_item(db: Session, item: schemas.ItemCreate, user_id: int):
    db_item = models.Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def update_item(db: Session, item_id: int, item: schemas.ItemCreate):
    db_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if db_item:
        for key, value in item.dict().items():
            setattr(db_item, key, value)
        db.commit()
        db.refresh(db_item)
    return db_item

def delete_item(db: Session, item_id: int):
    db_item = db.query(models.Item).filter(models.Item.id == item_id).first()
    if db_item:
        db.delete(db_item)
        db.commit()
    return db_item
```

### Main Application with Database

```python
# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

import crud, models, schemas
from database import SessionLocal, engine, get_db

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

@app.post("/users", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)

@app.get("/users", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/users/{user_id}/items", response_model=schemas.Item)
def create_item_for_user(
    user_id: int,
    item: schemas.ItemCreate,
    db: Session = Depends(get_db)
):
    return crud.create_item(db=db, item=item, user_id=user_id)
```

---

## Level 5: Authentication & Security

### Password Hashing

```python
# security.py
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt

# Install: pip install passlib[bcrypt] python-jose[cryptography]

SECRET_KEY = "your-secret-key-here"  # Use environment variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
```

### OAuth2 with JWT

```python
# auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

import crud, schemas, security
from database import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def authenticate_user(db: Session, email: str, password: str):
    user = crud.get_user_by_email(db, email)
    if not user:
        return False
    if not security.verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = security.verify_token(token)
    if payload is None:
        raise credentials_exception
    
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception
    
    user = crud.get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception
    
    return user

async def get_current_active_user(
    current_user: schemas.User = Depends(get_current_user)
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# main.py additions
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordRequestForm

@app.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=schemas.User)
async def read_users_me(
    current_user: schemas.User = Depends(get_current_active_user)
):
    return current_user

@app.get("/protected")
async def protected_route(
    current_user: schemas.User = Depends(get_current_active_user)
):
    return {"message": f"Hello {current_user.email}, you are authenticated!"}
```

### API Key Security

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY = "your-api-key-here"
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key

@app.get("/secure-data")
def get_secure_data(api_key: str = Depends(get_api_key)):
    return {"data": "This is secure data"}
```

---

## Level 6: Advanced Features

### Dependency Injection

```python
from fastapi import Depends, Header, HTTPException

# Simple dependency
def common_parameters(q: str = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items")
def read_items(commons: dict = Depends(common_parameters)):
    return commons

# Class-based dependency
class CommonQueryParams:
    def __init__(self, q: str = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/users")
def read_users(commons: CommonQueryParams = Depends()):
    return commons

# Nested dependencies
def verify_token(x_token: str = Header(...)):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")
    return x_token

def verify_key(x_key: str = Header(...)):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key

@app.get("/protected-items")
def read_protected_items(
    token: str = Depends(verify_token),
    key: str = Depends(verify_key)
):
    return {"message": "Access granted"}
```

### Background Tasks

```python
from fastapi import BackgroundTasks
import time

def write_log(message: str):
    time.sleep(2)  # Simulate slow operation
    with open("log.txt", "a") as f:
        f.write(f"{message}\n")

def send_email(email: str, message: str):
    time.sleep(3)
    print(f"Email sent to {email}: {message}")

@app.post("/send-notification/{email}")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, email, "Thank you for signing up!")
    background_tasks.add_task(write_log, f"Email sent to {email}")
    return {"message": "Notification sent in background"}
```

### WebSocket Support

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")
```

### Middleware

```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### Testing

```python
# test_main.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_user():
    response = client.post(
        "/users",
        json={"email": "test@example.com", "password": "secret"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"

def test_read_user():
    response = client.get("/users/1")
    assert response.status_code == 200
    assert "email" in response.json()

def test_authentication():
    # Create user
    client.post(
        "/users",
        json={"email": "auth@example.com", "password": "secret"}
    )
    
    # Login
    response = client.post(
        "/token",
        data={"username": "auth@example.com", "password": "secret"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    # Access protected route
    response = client.get(
        "/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
```

---

## Level 7: Production & Deployment

### Configuration Management

```python
# config.py
from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "My FastAPI App"
    admin_email: str
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

# .env file
# ADMIN_EMAIL=admin@example.com
# DATABASE_URL=postgresql://user:password@localhost/dbname
# SECRET_KEY=your-secret-key

# Usage in main.py
from config import get_settings

@app.get("/info")
def get_info(settings: Settings = Depends(get_settings)):
    return {
        "app_name": settings.app_name,
        "admin_email": settings.admin_email
    }
```

### Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.get("/items/{item_id}")
def read_item(item_id: int):
    logger.info(f"Reading item {item_id}")
    try:
        # Your logic here
        return {"item_id": item_id}
    except Exception as e:
        logger.error(f"Error reading item {item_id}: {str(e)}")
        raise
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/dbname
    depends_on:
      - db
    volumes:
      - .:/app
  
  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=dbname
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Performance Optimization

```python
# Use async for I/O operations
import httpx
from typing import List

# Async database queries
@app.get("/users/{user_id}")
async def read_user_async(user_id: int, db: Session = Depends(get_db)):
    # Use async SQLAlchemy or databases library
    user = await db.execute(select(User).where(User.id == user_id))
    return user

# Async HTTP requests
@app.get("/external-data")
async def get_external_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

# Parallel async operations
@app.get("/multiple-sources")
async def get_multiple_sources():
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            client.get("https://api1.example.com/data"),
            client.get("https://api2.example.com/data"),
            client.get("https://api3.example.com/data")
        )
        return [r.json() for r in results]

# Caching with Redis
from redis import asyncio as aioredis
import json

redis = aioredis.from_url("redis://localhost")

@app.get("/cached-data/{item_id}")
async def get_cached_data(item_id: int):
    # Try cache first
    cached = await redis.get(f"item:{item_id}")
    if cached:
        return json.loads(cached)
    
    # Fetch from database
    item = await fetch_from_db(item_id)
    
    # Store in cache
    await redis.setex(
        f"item:{item_id}",
        3600,  # 1 hour
        json.dumps(item)
    )
    return item

# Response caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

@cache(expire=60)  # Cache for 60 seconds
@app.get("/expensive-operation")
async def expensive_operation():
    # Expensive computation
    result = perform_heavy_computation()
    return result
```

### Rate Limiting

```python
# Install: pip install slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/limited")
@limiter.limit("5/minute")
async def limited_route(request: Request):
    return {"message": "This route is rate limited"}

@app.get("/strict-limit")
@limiter.limit("2/minute")
async def strict_limited_route(request: Request):
    return {"message": "Very strict rate limit"}
```

### API Versioning

```python
from fastapi import APIRouter

# Version 1
router_v1 = APIRouter(prefix="/api/v1")

@router_v1.get("/users")
def get_users_v1():
    return {"version": "1.0", "users": []}

@router_v1.get("/items/{item_id}")
def get_item_v1(item_id: int):
    return {"version": "1.0", "item_id": item_id}

# Version 2 with breaking changes
router_v2 = APIRouter(prefix="/api/v2")

@router_v2.get("/users")
def get_users_v2(include_inactive: bool = False):
    return {
        "version": "2.0",
        "users": [],
        "include_inactive": include_inactive
    }

@router_v2.get("/items/{item_id}")
def get_item_v2(item_id: int):
    return {
        "version": "2.0",
        "id": item_id,  # Changed field name
        "data": {}
    }

# Register routers
app.include_router(router_v1)
app.include_router(router_v2)
```

---

## Expert Projects

### Project 1: Full E-Commerce API

```python
# Complete e-commerce API with all features
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from enum import Enum

app = FastAPI(title="E-Commerce API")

# Models
class OrderStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    shipped = "shipped"
    delivered = "delivered"
    cancelled = "cancelled"

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    category_id = Column(Integer, ForeignKey("categories.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    category = relationship("Category", back_populates="products")
    order_items = relationship("OrderItem", back_populates="product")

class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    products = relationship("Product", back_populates="category")

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default=OrderStatus.pending.value)
    total = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    
    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")

# Schemas
class ProductBase(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    stock: int
    category_id: int

class ProductCreate(ProductBase):
    pass

class ProductOut(ProductBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class OrderItemCreate(BaseModel):
    product_id: int
    quantity: int

class OrderCreate(BaseModel):
    items: List[OrderItemCreate]

class OrderOut(BaseModel):
    id: int
    user_id: int
    status: OrderStatus
    total: float
    created_at: datetime
    items: List[dict]
    
    class Config:
        orm_mode = True

# Routes
@app.post("/products", response_model=ProductOut)
async def create_product(
    product: ProductCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Only admin can create products
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    db_product = Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

@app.get("/products", response_model=List[ProductOut])
async def list_products(
    skip: int = 0,
    limit: int = 100,
    category_id: Optional[int] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Product)
    
    if category_id:
        query = query.filter(Product.category_id == category_id)
    if min_price:
        query = query.filter(Product.price >= min_price)
    if max_price:
        query = query.filter(Product.price <= max_price)
    if search:
        query = query.filter(Product.name.contains(search))
    
    products = query.offset(skip).limit(limit).all()
    return products

@app.post("/orders", response_model=OrderOut)
async def create_order(
    order: OrderCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks
):
    # Validate products and calculate total
    total = 0
    order_items = []
    
    for item in order.items:
        product = db.query(Product).filter(Product.id == item.product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail=f"Product {item.product_id} not found")
        if product.stock < item.quantity:
            raise HTTPException(status_code=400, detail=f"Insufficient stock for {product.name}")
        
        total += product.price * item.quantity
        order_items.append({
            "product_id": product.id,
            "quantity": item.quantity,
            "price": product.price
        })
        
        # Update stock
        product.stock -= item.quantity
    
    # Create order
    db_order = Order(user_id=current_user.id, total=total)
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    
    # Create order items
    for item_data in order_items:
        db_item = OrderItem(order_id=db_order.id, **item_data)
        db.add(db_item)
    
    db.commit()
    
    # Send confirmation email in background
    background_tasks.add_task(
        send_order_confirmation,
        current_user.email,
        db_order.id
    )
    
    return db_order

@app.get("/orders/me", response_model=List[OrderOut])
async def get_my_orders(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    orders = db.query(Order).filter(Order.user_id == current_user.id).all()
    return orders

@app.patch("/orders/{order_id}/status")
async def update_order_status(
    order_id: int,
    status: OrderStatus,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Only admin can update order status
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    order.status = status.value
    db.commit()
    
    return {"message": "Order status updated", "status": status}
```

### Project 2: Real-time Chat Application

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from typing import List, Dict
import json
from datetime import datetime

app = FastAPI()

class ChatManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}
        self.messages: Dict[int, List[dict]] = {}
    
    async def connect(self, websocket: WebSocket, room_id: int, user_id: int):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
            self.messages[room_id] = []
        self.active_connections[room_id].append(websocket)
        
        # Send chat history
        await websocket.send_json({
            "type": "history",
            "messages": self.messages[room_id]
        })
    
    def disconnect(self, websocket: WebSocket, room_id: int):
        self.active_connections[room_id].remove(websocket)
    
    async def broadcast(self, message: dict, room_id: int):
        # Save message
        self.messages[room_id].append(message)
        
        # Broadcast to all connections in room
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                await connection.send_json(message)

manager = ChatManager()

@app.websocket("/chat/{room_id}")
async def chat_endpoint(
    websocket: WebSocket,
    room_id: int,
    token: str
):
    # Verify token and get user
    user = verify_websocket_token(token)
    if not user:
        await websocket.close(code=1008)
        return
    
    await manager.connect(websocket, room_id, user.id)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = {
                "user_id": user.id,
                "username": user.username,
                "message": data,
                "timestamp": datetime.utcnow().isoformat(),
                "type": "message"
            }
            await manager.broadcast(message, room_id)
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
        await manager.broadcast({
            "type": "user_left",
            "user_id": user.id,
            "username": user.username,
            "timestamp": datetime.utcnow().isoformat()
        }, room_id)

# REST endpoints for chat rooms
@app.post("/rooms")
async def create_room(
    name: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    room = ChatRoom(name=name, creator_id=current_user.id)
    db.add(room)
    db.commit()
    return {"id": room.id, "name": room.name}

@app.get("/rooms")
async def list_rooms(db: Session = Depends(get_db)):
    rooms = db.query(ChatRoom).all()
    return rooms
```

### Project 3: File Processing Service

```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from typing import List
import pandas as pd
import io
from PIL import Image

app = FastAPI()

# CSV Processing
@app.post("/process-csv")
async def process_csv(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    # Process in background
    task_id = generate_task_id()
    background_tasks.add_task(process_dataframe, df, task_id)
    
    return {
        "task_id": task_id,
        "message": "Processing started",
        "rows": len(df),
        "columns": list(df.columns)
    }

async def process_dataframe(df: pd.DataFrame, task_id: str):
    # Heavy processing
    results = {
        "summary": df.describe().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum()
    }
    
    # Store results
    await store_results(task_id, results)

# Image Processing
@app.post("/process-images")
async def process_images(files: List[UploadFile] = File(...)):
    results = []
    
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process image
        thumbnail = image.copy()
        thumbnail.thumbnail((200, 200))
        
        # Save thumbnail
        output = io.BytesIO()
        thumbnail.save(output, format='JPEG')
        
        results.append({
            "filename": file.filename,
            "original_size": image.size,
            "thumbnail_size": thumbnail.size,
            "format": image.format
        })
    
    return {"processed": len(files), "results": results}

# Batch processing with progress tracking
processing_status = {}

@app.post("/batch-process")
async def batch_process(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    batch_id = generate_batch_id()
    processing_status[batch_id] = {
        "total": len(files),
        "processed": 0,
        "status": "processing"
    }
    
    background_tasks.add_task(process_batch, files, batch_id)
    
    return {"batch_id": batch_id, "total_files": len(files)}

@app.get("/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    if batch_id not in processing_status:
        raise HTTPException(status_code=404, detail="Batch not found")
    return processing_status[batch_id]

async def process_batch(files: List[UploadFile], batch_id: str):
    for i, file in enumerate(files):
        # Process each file
        await process_single_file(file)
        processing_status[batch_id]["processed"] = i + 1
    
    processing_status[batch_id]["status"] = "completed"
```

---

## Best Practices & Tips

### 1. Project Structure
```
my_fastapi_project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── dependencies.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── crud/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── item.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── users.py
│   │   │   └── items.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py
│   │   └── database.py
│   └── tests/
│       ├── __init__.py
│       ├── test_users.py
│       └── test_items.py
├── .env
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

### 2. Error Handling Best Practices
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class CustomException(Exception):
    def __init__(self, name: str, detail: str):
        self.name = name
        self.detail = detail

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=418,
        content={
            "error": exc.name,
            "detail": exc.detail,
            "path": str(request.url)
        }
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

### 3. Database Best Practices
```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True
)

# Use async SQLAlchemy for better performance
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

async_engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    echo=True
)

# Always use transactions
async with AsyncSession(async_engine) as session:
    async with session.begin():
        # Your database operations
        pass
```

### 4. Security Checklist
- ✅ Use HTTPS in production
- ✅ Implement rate limiting
- ✅ Validate all inputs with Pydantic
- ✅ Use parameterized queries (SQLAlchemy does this)
- ✅ Implement proper authentication (JWT)
- ✅ Use environment variables for secrets
- ✅ Enable CORS only for trusted origins
- ✅ Implement request size limits
- ✅ Use secure password hashing (bcrypt)
- ✅ Implement CSRF protection for forms
- ✅ Add security headers middleware

### 5. Performance Tips
- Use async/await for I/O operations
- Implement caching (Redis)
- Use database indexes
- Paginate large result sets
- Use background tasks for heavy operations
- Enable Gzip compression
- Use CDN for static files
- Monitor with APM tools (New Relic, DataDog)
- Use connection pooling
- Optimize database queries (avoid N+1)

### 6. Documentation Tips
```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="A comprehensive API for...",
    version="1.0.0",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "API Support",
        "url": "http://example.com/contact/",
        "email": "support@example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

@app.get(
    "/items/{item_id}",
    summary="Get an item",
    description="Get an item by its ID",
    response_description="The requested item",
    tags=["items"]
)
async def read_item(item_id: int):
    """
    Retrieve an item with all the information:
    
    - **item_id**: unique identifier
    - Returns item details
    """
    return {"item_id": item_id}
```

---

## Resources for Continued Learning

1. **Official Documentation**: https://fastapi.tiangolo.com
2. **FastAPI GitHub**: https://github.com/tiangolo/fastapi
3. **Awesome FastAPI**: Collection of FastAPI resources
4. **FastAPI Tutorial Series**: Build real-world projects
5. **Pydantic Documentation**: Master data validation
6. **SQLAlchemy Documentation**: Database expertise
7. **OAuth2/JWT**: Authentication deep dive

## Practice Projects

1. Blog API with comments and likes
2. Task management system with teams
3. Social media API with posts and followers
4. Video streaming platform API
5. Real-time notification service
6. Payment processing integration
7. Multi-tenant SaaS application
8. GraphQL integration
9. Microservices architecture
10. AI/ML model serving API

---

**Congratulations!** You now have a comprehensive guide from zero to expert in FastAPI. Practice by building real projects and refer back to this guide as needed.