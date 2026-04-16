import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

DB_PATH = Path('data/archive.db')

ENTITY_TYPE_MAP = {
    'PERS': 'PERSON',
    'PERSON': 'PERSON',
    'LOC': 'LOCATION',
    'PLACE': 'LOCATION',
    'LOCATION': 'LOCATION',
    'ORG': 'ORGANIZATION',
    'ORGANIZATION': 'ORGANIZATION',
    'DATE': 'DATE',
    'OBJECT': 'OBJECT',
    'MISC': 'MISC'
}


def get_connection(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_db(conn: sqlite3.Connection):
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT UNIQUE NOT NULL,
        title TEXT,
        image_path TEXT,
        description TEXT,
        ocr_text_raw TEXT,
        ocr_text_clean TEXT,
        translation_input TEXT,
        language TEXT DEFAULT 'ar',
        translation_status TEXT DEFAULT 'not_started',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        summary_type TEXT NOT NULL,
        summary_text TEXT,
        method TEXT,
        quality_note TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        entity_type TEXT NOT NULL,
        entity_text TEXT NOT NULL,
        normalized_text TEXT,
        confidence REAL,
        source_method TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS links (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        target_type TEXT NOT NULL,
        target_id TEXT NOT NULL,
        relation TEXT NOT NULL,
        confidence REAL,
        notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_entities_document_id ON entities(document_id);
    CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
    CREATE INDEX IF NOT EXISTS idx_summaries_document_id ON summaries(document_id);
    CREATE INDEX IF NOT EXISTS idx_links_document_id ON links(document_id);

    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
        source,
        title,
        description,
        ocr_text_clean,
        summary_text
    );
    ''')
    conn.commit()


def _now() -> str:
    return datetime.utcnow().isoformat(timespec='seconds')


def normalize_entity_type(entity_type: str) -> str:
    return ENTITY_TYPE_MAP.get(entity_type, entity_type)


def upsert_document(
    conn: sqlite3.Connection,
    source: str,
    title: Optional[str] = None,
    image_path: Optional[str] = None,
    description: Optional[str] = None,
    ocr_text_raw: Optional[str] = None,
    ocr_text_clean: Optional[str] = None,
    translation_input: Optional[str] = None,
    translation_status: str = 'not_started'
) -> int:
    now = _now()
    cur = conn.execute('SELECT id, created_at FROM documents WHERE source = ?', (source,))
    row = cur.fetchone()
    if row:
        conn.execute('''
            UPDATE documents
            SET title = COALESCE(?, title),
                image_path = COALESCE(?, image_path),
                description = COALESCE(?, description),
                ocr_text_raw = COALESCE(?, ocr_text_raw),
                ocr_text_clean = COALESCE(?, ocr_text_clean),
                translation_input = COALESCE(?, translation_input),
                translation_status = ?,
                updated_at = ?
            WHERE id = ?
        ''', (title, image_path, description, ocr_text_raw, ocr_text_clean,
              translation_input, translation_status, now, row['id']))
        doc_id = row['id']
    else:
        cur = conn.execute('''
            INSERT INTO documents (
                source, title, image_path, description, ocr_text_raw,
                ocr_text_clean, translation_input, translation_status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (source, title, image_path, description, ocr_text_raw,
              ocr_text_clean, translation_input, translation_status, now, now))
        doc_id = cur.lastrowid
    conn.commit()
    return doc_id


def replace_entities(conn: sqlite3.Connection, document_id: int, entities: Dict[str, List[str]], source_method: str = 'pipeline', confidence: Optional[float] = None):
    conn.execute('DELETE FROM entities WHERE document_id = ? AND source_method = ?', (document_id, source_method))
    now = _now()
    rows = []
    for entity_type, values in entities.items():
        norm_type = normalize_entity_type(entity_type)
        for value in values or []:
            value = (value or '').strip()
            if not value:
                continue
            rows.append((document_id, norm_type, value, value, confidence, source_method, now))
    if rows:
        conn.executemany('''
            INSERT INTO entities (
                document_id, entity_type, entity_text, normalized_text,
                confidence, source_method, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', rows)
    conn.commit()


def upsert_summary(conn: sqlite3.Connection, document_id: int, summary_type: str, summary_text: str, method: str = 'mt_draft', quality_note: str = 'experimental'):
    now = _now()
    cur = conn.execute(
        'SELECT id FROM summaries WHERE document_id = ? AND summary_type = ?',
        (document_id, summary_type)
    )
    row = cur.fetchone()
    if row:
        conn.execute('''
            UPDATE summaries
            SET summary_text = ?, method = ?, quality_note = ?, updated_at = ?
            WHERE id = ?
        ''', (summary_text, method, quality_note, now, row['id']))
    else:
        conn.execute('''
            INSERT INTO summaries (
                document_id, summary_type, summary_text, method,
                quality_note, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (document_id, summary_type, summary_text, method, quality_note, now, now))
    conn.commit()
    rebuild_fts(conn)


def add_link(conn: sqlite3.Connection, document_id: int, target_type: str, target_id: str, relation: str, confidence: Optional[float] = None, notes: Optional[str] = None):
    conn.execute('''
        INSERT INTO links (document_id, target_type, target_id, relation, confidence, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (document_id, target_type, target_id, relation, confidence, notes, _now()))
    conn.commit()


def rebuild_fts(conn: sqlite3.Connection):
    conn.execute('DELETE FROM documents_fts')
    conn.execute('''
        INSERT INTO documents_fts (rowid, source, title, description, ocr_text_clean, summary_text)
        SELECT d.id,
               d.source,
               COALESCE(d.title, ''),
               COALESCE(d.description, ''),
               COALESCE(d.ocr_text_clean, ''),
               COALESCE((
                   SELECT group_concat(summary_text, ' ')
                   FROM summaries s
                   WHERE s.document_id = d.id
               ), '')
        FROM documents d
    ''')
    conn.commit()


def get_document_by_source(conn: sqlite3.Connection, source: str):
    row = conn.execute('SELECT * FROM documents WHERE source = ?', (source,)).fetchone()
    return dict(row) if row else None


def get_document_details(conn: sqlite3.Connection, source: str):
    doc = get_document_by_source(conn, source)
    if not doc:
        return None
    entities = conn.execute('''
        SELECT entity_type, entity_text, source_method, confidence
        FROM entities WHERE document_id = ?
        ORDER BY entity_type, entity_text
    ''', (doc['id'],)).fetchall()
    summaries = conn.execute('''
        SELECT summary_type, summary_text, method, quality_note
        FROM summaries WHERE document_id = ?
        ORDER BY id
    ''', (doc['id'],)).fetchall()
    links = conn.execute('''
        SELECT target_type, target_id, relation, confidence, notes
        FROM links WHERE document_id = ?
        ORDER BY id
    ''', (doc['id'],)).fetchall()
    doc['entities'] = [dict(r) for r in entities]
    doc['summaries'] = [dict(r) for r in summaries]
    doc['links'] = [dict(r) for r in links]
    return doc


def search_documents_fts(conn: sqlite3.Connection, query: str, limit: int = 10):
    rebuild_fts(conn)
    rows = conn.execute('''
        SELECT d.id, d.source, d.description, d.ocr_text_clean,
               bm25(documents_fts) AS rank
        FROM documents_fts
        JOIN documents d ON d.id = documents_fts.rowid
        WHERE documents_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    ''', (query, limit)).fetchall()
    return [dict(r) for r in rows]
