# backend/models.py

from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False) # Using Text for potentially long transcripts
    created_at = Column(DateTime(timezone=True), server_default=func.now())