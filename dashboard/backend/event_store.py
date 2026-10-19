
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class EventStore:
    """Armazenamento persistente de eventos do VisionAI."""

    def __init__(self, database_path="data/visionai_events.sqlite3"):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self.lock, self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    person_name TEXT,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_created_at
                ON events(created_at)
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_status
                ON events(status)
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_event_key
                ON events(event_key, status)
            """)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record(self, event_key, event_type, person_name, message,
               cooldown_seconds=4):
        """
        Cria um evento se não houver um evento ativo da mesma chave
        nos últimos cooldown_seconds. Eventos ativos são reutilizados.
        """
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat(timespec="seconds")

        with self.lock, self._connect() as connection:
            existing = connection.execute("""
                SELECT id, created_at
                FROM events
                WHERE event_key = ? AND status = 'active'
                ORDER BY id DESC
                LIMIT 1
            """, (event_key,)).fetchone()

            if existing:
                previous = datetime.fromisoformat(existing["created_at"])
                elapsed = (now - previous).total_seconds()

                if elapsed < cooldown_seconds:
                    return {
                        "created": False,
                        "id": existing["id"],
                        "reason": "cooldown"
                    }

                # O evento permanece ativo. Não criamos duplicatas
                # para a mesma ocorrência contínua.
                connection.execute("""
                    UPDATE events
                    SET updated_at = ?
                    WHERE id = ?
                """, (timestamp, existing["id"]))

                return {
                    "created": False,
                    "id": existing["id"],
                    "reason": "already_active"
                }

            cursor = connection.execute("""
                INSERT INTO events (
                    event_key, event_type, person_name, message,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'active', ?, ?)
            """, (
                event_key, event_type, person_name,
                message, timestamp, timestamp
            ))

            return {
                "created": True,
                "id": cursor.lastrowid,
                "reason": "created"
            }

    def resolve(self, event_key):
        """Encerra os eventos ativos associados à chave."""
        timestamp = self._now()

        with self.lock, self._connect() as connection:
            cursor = connection.execute("""
                UPDATE events
                SET status = 'resolved', updated_at = ?
                WHERE event_key = ? AND status = 'active'
            """, (timestamp, event_key))

            return cursor.rowcount

    def list_events(self, limit=500, status="all"):
        limit = max(1, min(int(limit), 2000))

        query = """
            SELECT id, event_key, event_type, person_name,
                   message, status, created_at, updated_at
            FROM events
        """
        params = []

        if status in ("active", "resolved"):
            query += " WHERE status = ?"
            params.append(status)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self.lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def summary(self):
        today = datetime.now(timezone.utc).date().isoformat()

        with self.lock, self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM events"
            ).fetchone()[0]

            active = connection.execute(
                "SELECT COUNT(*) FROM events WHERE status = 'active'"
            ).fetchone()[0]

            resolved = connection.execute(
                "SELECT COUNT(*) FROM events WHERE status = 'resolved'"
            ).fetchone()[0]

            today_count = connection.execute("""
                SELECT COUNT(*) FROM events
                WHERE substr(created_at, 1, 10) = ?
            """, (today,)).fetchone()[0]

        return {
            "total": total,
            "active": active,
            "resolved": resolved,
            "today": today_count
        }
