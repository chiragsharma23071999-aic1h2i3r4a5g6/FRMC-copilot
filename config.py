import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import make_url

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///frmc.db"
_url = make_url(DATABASE_URL)
if _url.drivername.startswith("sqlite") and _url.database not in (None, "", ":memory:"):
    if not Path(_url.database).is_absolute():
        DATABASE_URL = _url.set(database=str(BASE_DIR / _url.database)).render_as_string(hide_password=False)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = "local-lexical-v1"
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
