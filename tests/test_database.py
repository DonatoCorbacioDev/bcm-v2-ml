from unittest.mock import MagicMock, patch
from app.database import connect_args, get_db


def test_no_ca_no_tls_settings():
    assert connect_args("") == {}


def test_a_ca_verifies_the_server_certificate_and_its_name():
    # The chain against the CA, and the name against the host it connects to.
    assert connect_args("/run/secrets/MYSQL_CA_CERT") == {
        "ssl": {"ca": "/run/secrets/MYSQL_CA_CERT", "check_hostname": True}
    }


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
