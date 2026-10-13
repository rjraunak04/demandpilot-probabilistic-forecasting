"""Database sessions."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def database_url(): return os.getenv("DEMANDPILOT_DATABASE_URL","sqlite+pysqlite:///./demandpilot.db")
def build_engine(url=None): return create_engine(url or database_url(), pool_pre_ping=True)
def session_factory(url=None): return sessionmaker(bind=build_engine(url), expire_on_commit=False)
def get_session():
    with session_factory()() as session: yield session
