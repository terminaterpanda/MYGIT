import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
import os 
DB_URL = DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL, echo=True) #database 연결 관리 
SessionLocal = sessionmaker(autocommit = False, autoflush=False, bind=engine)
#sessionlocal == "database session make에 use."
Base = declarative_base()
#BASE -> 모든 모델 클래스가 상속받을 기본 class

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer,primary_key=True, index=True)
    name = Column(String(255),index=True)
    email = Column(String(255), unique=True, index =True
                   )
    todos = relationship("Todo", back_populates="owner")
    is_active = Column(Boolean, default=False)
class Todo(Base):
    __tablename__ = "todos"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True)
    description = Column(String(255), index=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="todos")
    
    