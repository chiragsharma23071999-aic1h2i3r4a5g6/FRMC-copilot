from sqlalchemy import Column, Integer, String, DateTime, Boolean, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from config import DATABASE_URL

Base = declarative_base()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

# Example table: Reports
class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    content = Column(String, nullable=False)
    author = Column(String, default="System")
    timestamp = Column(DateTime, default=utc_now)

# Example table: Logs
class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent = Column(String, nullable=False)
    action = Column(String, nullable=False)
    details = Column(String)
    timestamp = Column(DateTime, default=utc_now)

# Example table: Users
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    is_active = Column(Boolean, default=True)

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

def init_db():
    Base.metadata.create_all(engine)
