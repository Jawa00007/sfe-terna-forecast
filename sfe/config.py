"""Runtime configuration.

Settings come from (in precedence order): constructor args, environment variables (prefix
``SFE_``, nested with ``__``), a ``.env`` file, then defaults. YAML files under ``conf/``
carry the domain configuration (endpoints, zones, regimes, charges) and are loaded lazily
via :func:`load_yaml`.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = REPO_ROOT / "conf"


class TernaSettings(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    token_url: str = "https://api.terna.it/transparency/oauth/accessToken"
    base_url: str = "https://api.terna.it/transparency/v1.0"
    request_timeout_s: float = 30.0
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_max_s: float = 60.0
    # Refresh the token this many seconds before its stated expiry.
    token_refresh_margin_s: int = 60

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


class StorageSettings(BaseModel):
    backend: Literal["local"] = "local"
    root: Path = REPO_ROOT / "data"

    @property
    def landing_dir(self) -> Path:
        return self.root / "landing"

    @property
    def curated_dir(self) -> Path:
        return self.root / "curated"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"

    @property
    def catalog_path(self) -> Path:
        return self.root / "catalog.duckdb"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SFE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    conf_dir: Path = CONF_DIR

    terna: TernaSettings = Field(default_factory=TernaSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache(maxsize=16)
def load_yaml(name: str) -> dict[str, Any]:
    """Load ``conf/<name>.yaml`` (``.yaml`` optional in *name*)."""
    fname = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    path = get_settings().conf_dir / fname
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
