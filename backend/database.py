# backend/database.py

import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Check if we're in production (Render sets RENDER environment variable)
IS_PRODUCTION = os.getenv("RENDER") is not None

if IS_PRODUCTION:
    # In production, use Render's database URL
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set in production")
else:
    # In development, use local database or fallback
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:Abdulnasar8051@localhost:5432/voice_transcriber_db")

# Enhanced validation
if not DATABASE_URL or DATABASE_URL.strip() == "" or DATABASE_URL == "None":
    raise ValueError("DATABASE_URL is empty or invalid")

print(f"Environment: {'Production' if IS_PRODUCTION else 'Development'}")
print(f"Using DATABASE_URL: {DATABASE_URL[:50]}...")

# Check if it's a postgres:// URL that needs conversion
if DATABASE_URL.startswith("postgres://"):
    print("Converting postgres:// to postgresql://")
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# For async SQLAlchemy, ensure it uses the async driver
if DATABASE_URL.startswith("postgresql://") and "+asyncpg" not in DATABASE_URL:
    print("Adding +asyncpg driver to PostgreSQL URL")
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

print(f"Final DATABASE_URL format: {DATABASE_URL[:50]}...")

try:
    # Create an async engine
    engine = create_async_engine(DATABASE_URL)
    print("✅ Database engine created successfully")
except Exception as e:
    print(f"❌ Failed to create database engine: {e}")
    raise

# Create a session maker
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)