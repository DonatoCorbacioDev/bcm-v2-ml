# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

"""Docker secrets: settings read from files named after them (OWASP Secrets Management)."""

from app import config
from app.config import Settings


def test_a_secret_file_is_the_setting_of_the_same_name(tmp_path, monkeypatch):
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    (tmp_path / "DB_URL").write_text("mysql+pymysql://ro:from-file@mysql:3306/bcm\n")
    (tmp_path / "INTERNAL_API_KEY").write_text("key-from-file\n")

    settings = Settings(_env_file=None, _secrets_dir=str(tmp_path))

    assert settings.DB_URL == "mysql+pymysql://ro:from-file@mysql:3306/bcm"
    assert settings.INTERNAL_API_KEY == "key-from-file"


def test_an_environment_variable_still_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", "mysql+pymysql://ro:from-env@mysql:3306/bcm")
    (tmp_path / "DB_URL").write_text("mysql+pymysql://ro:from-file@mysql:3306/bcm")

    settings = Settings(_env_file=None, _secrets_dir=str(tmp_path))

    assert settings.DB_URL == "mysql+pymysql://ro:from-env@mysql:3306/bcm"


def test_the_directory_is_used_only_when_it_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SECRETS_DIR", str(tmp_path))
    assert config._secrets_dir() == str(tmp_path)

    monkeypatch.setattr(config, "SECRETS_DIR", str(tmp_path / "missing"))
    assert config._secrets_dir() is None
