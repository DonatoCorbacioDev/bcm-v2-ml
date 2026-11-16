"""OWASP SQL Injection Prevention Cheat Sheet, denied by default: the ML
service reaches the database only through SQLAlchemy's ORM and expression
language, which bind every value. Raw SQL text, a driver-level statement or a
query written in a string fails here."""

import re
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"

# Ways of handing the database a string of SQL.
RAW_SQL = {
    "sqlalchemy text()": re.compile(r"\btext\s*\(\s*[fr]?[\"']"),
    "import of text": re.compile(r"from\s+sqlalchemy(?:\.sql)?\s+import\s+[^\n]*\btext\b"),
    "exec_driver_sql": re.compile(r"\.exec_driver_sql\s*\("),
    "execute() with a string": re.compile(r"\.execute\s*\(\s*[fr]?[\"']"),
    "cursor": re.compile(r"\.cursor\s*\("),
    "SQL in a string": re.compile(r"[fr]?[\"']\s*(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)\s+[^\"']*\b(FROM|INTO|SET|TABLE)\b",
                                  re.IGNORECASE),
}


def _sources():
    return sorted(APP.rglob("*.py"))


def test_no_raw_sql_reaches_the_database():
    found = {}
    for path in _sources():
        code = path.read_text(encoding="utf-8")
        for what, pattern in RAW_SQL.items():
            if pattern.search(code):
                found.setdefault(what, []).append(str(path.relative_to(APP)))
    assert found == {}


def test_the_guard_sees_raw_sql():
    samples = {
        "sqlalchemy text()": 'db.execute(text("SELECT * FROM users"))',
        "exec_driver_sql": 'conn.exec_driver_sql("DELETE FROM contracts")',
        "execute() with a string": 'db.execute(f"SELECT * FROM contracts WHERE id = {cid}")',
        "SQL in a string": 'query = "select id from contracts where organization_id = " + org',
    }
    for what, sample in samples.items():
        assert RAW_SQL[what].search(sample), what


def test_the_guard_reads_the_app():
    assert len(_sources()) > 5
