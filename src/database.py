from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from contextlib import contextmanager
from typing import Generator

from src.config import DATABASE_PATH

Base = declarative_base()

# ##################################################################
# create engine
# create the SQLAlchemy engine for SQLite
_engine = None


# ##################################################################
# get engine
# lazily create and return the database engine
def get_engine():
    global _engine
    if _engine is None:
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)
    return _engine


# ##################################################################
# get session factory
# return a sessionmaker bound to the engine
def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine())


# ##################################################################
# get session
# context manager for database sessions with automatic commit/rollback
@contextmanager
def get_session() -> Generator[Session, None, None]:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ##################################################################
# init db
# create all tables in the database
def init_db() -> None:
    from src import models  # noqa: F401 - needed to register models

    Base.metadata.create_all(get_engine())
