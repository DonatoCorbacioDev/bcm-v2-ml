from unittest.mock import MagicMock, patch
from app.database import get_db


def test_get_db_yields_session_and_closes():
    mock_session = MagicMock()
    with patch("app.database.SessionLocal", return_value=mock_session):
        gen = get_db()
        db = next(gen)
        assert db is mock_session
        try:
            next(gen)
        except StopIteration:
            pass
        mock_session.close.assert_called_once()
