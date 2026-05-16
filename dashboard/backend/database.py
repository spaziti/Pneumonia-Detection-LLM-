"""
Dashboard Backend: SQLite database for patient prediction history.

Uses aiosqlite for async operations with FastAPI.
"""

import aiosqlite
from pathlib import Path
from datetime import datetime

DB_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DB_DIR / "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_name TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    uncertainty REAL NOT NULL,
    spatial_region TEXT,
    report TEXT,
    ehr_json TEXT
);
"""


async def init_db():
    """Create the database and tables if they don't exist."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.executescript(SCHEMA)
        await db.commit()


async def save_prediction(record: dict):
    """Insert a new prediction record."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute(
            """INSERT INTO predictions
               (timestamp, image_name, prediction, confidence,
                uncertainty, spatial_region, report, ehr_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.get("timestamp", datetime.now().isoformat()),
                record.get("image_name", "unknown"),
                record.get("prediction", ""),
                record.get("confidence", 0.0),
                record.get("uncertainty", 0.0),
                record.get("spatial_region", ""),
                record.get("report", ""),
                record.get("ehr_json", ""),
            ),
        )
        await db.commit()


async def get_history(limit: int = 50):
    """Get the most recent prediction records."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
