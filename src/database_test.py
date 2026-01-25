import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import init_db


# ##################################################################
# test init db creates tables
# verify that init_db creates the expected tables
def test_init_db_creates_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine = create_engine(f"sqlite:///{db_path}", echo=False)

        import src.database as db_module

        original_engine = db_module._engine
        db_module._engine = engine

        try:
            init_db()
            session_factory = sessionmaker(bind=engine)
            session = session_factory()
            from src.models import Feed, FeedEntry, PendingEntry

            feeds = session.query(Feed).all()
            assert feeds == []
            entries = session.query(FeedEntry).all()
            assert entries == []
            pending = session.query(PendingEntry).all()
            assert pending == []
            session.close()
        finally:
            db_module._engine = original_engine
