from sqlalchemy import Column, Integer, Float, String
from backend.db.database import Base

class GameLog(Base):
    __tablename__ = "game_logs"
    id = Column(Integer, primary_key=True)
