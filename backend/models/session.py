from sqlalchemy import Column, Integer, String
from backend.db.database import Base

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
