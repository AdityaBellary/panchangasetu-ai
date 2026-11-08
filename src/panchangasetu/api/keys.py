"""API keys: random, shown once, stored only as SHA-256 (keys are high-entropy, so a fast hash is
safe here; there is nothing to brute-force)."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from panchangasetu.storage.db import session_scope
from panchangasetu.storage.models import ApiKey, utcnow

KEY_PREFIX = "psk_"
_LAST_USED_RESOLUTION = timedelta(minutes=1)  # don't write on every request


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


@dataclass(frozen=True, slots=True)
class Client:
    id: str
    name: str
    rate_limit_per_minute: int | None
    is_admin: bool = False


@dataclass(frozen=True, slots=True)
class KeyInfo:
    id: str
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None
    is_admin: bool = False


class ApiKeyStore:
    def __init__(self, factory: sessionmaker[Session]) -> None:
        self.factory = factory

    def create(
        self, name: str, *, rate_limit_per_minute: int | None = None, admin: bool = False
    ) -> tuple[str, KeyInfo]:
        """Returns (the key, its record). The key cannot be recovered later."""
        key = new_key()
        with session_scope(self.factory) as s:
            row = ApiKey(
                name=name,
                prefix=key[: len(KEY_PREFIX) + 6],
                key_hash=hash_key(key),
                rate_limit_per_minute=rate_limit_per_minute,
                is_admin=admin,
            )
            s.add(row)
            s.flush()
            return key, _info(row)

    def authenticate(self, key: str) -> Client | None:
        if not key.startswith(KEY_PREFIX):
            return None
        with session_scope(self.factory) as s:
            row = s.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(key)))
            if row is None or row.revoked_at is not None:
                return None
            now = utcnow()
            if row.last_used_at is None or now - _aware(row.last_used_at) > _LAST_USED_RESOLUTION:
                row.last_used_at = now
            return Client(str(row.id), row.name, row.rate_limit_per_minute, row.is_admin)

    def list(self) -> list[KeyInfo]:
        with session_scope(self.factory) as s:
            return [_info(r) for r in s.scalars(select(ApiKey).order_by(ApiKey.created_at))]

    def revoke(self, id_or_prefix: str) -> int:
        with session_scope(self.factory) as s:
            rows = list(s.scalars(select(ApiKey).where(ApiKey.revoked_at.is_(None))))
            hits = [r for r in rows if str(r.id) == id_or_prefix or r.prefix == id_or_prefix]
            for r in hits:
                r.revoked_at = utcnow()
            return len(hits)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=utcnow().tzinfo)


def _info(r: ApiKey) -> KeyInfo:
    return KeyInfo(str(r.id), r.name, r.prefix, r.created_at, r.last_used_at, r.revoked_at, r.is_admin)
