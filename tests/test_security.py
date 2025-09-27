import asyncio
import hashlib
import os

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api import security


async def _receive() -> dict:
    return {"type": "http.request"}


def _build_request(path: str = "/predict", headers: dict | None = None, query: str = "") -> Request:
    raw_headers = []
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode("latin-1"), value.encode("latin-1")))

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": raw_headers,
        "query_string": query.encode("latin-1"),
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope, _receive)


@pytest.fixture(autouse=True)
def _reset_environment(monkeypatch):
    monkeypatch.delenv("TRUSTSHIELD_API_KEYS", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_API_KEYS_FILE", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_DISABLE_API_KEY", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_API_KEY_HEADER", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_API_KEY_QUERY", raising=False)
    monkeypatch.delenv("TRUSTSHIELD_PUBLIC_PATHS", raising=False)
    security.reset_api_key_cache()
    yield
    security.reset_api_key_cache()


def test_enforce_api_key_accepts_valid_header(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_API_KEYS", "local-key")
    request = _build_request(headers={"X-API-Key": "local-key"})

    asyncio.run(security.enforce_api_key(request))


def test_enforce_api_key_accepts_public_paths(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_API_KEYS", "another-key")
    request = _build_request(path="/healthz")

    asyncio.run(security.enforce_api_key(request))


def test_enforce_api_key_supports_query_parameter(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_API_KEYS", "dev")
    request = _build_request(query="api_key=dev")

    asyncio.run(security.enforce_api_key(request))


def test_enforce_api_key_rejects_invalid(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_API_KEYS", "secret")
    request = _build_request(headers={"X-API-Key": "wrong"})

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(security.enforce_api_key(request))

    assert excinfo.value.status_code == 403


def test_enforce_api_key_raises_when_missing(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_API_KEYS", "missing-test")
    request = _build_request()

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(security.enforce_api_key(request))

    assert excinfo.value.status_code == 401


def test_enforce_api_key_respects_disable_flag(monkeypatch):
    monkeypatch.setenv("TRUSTSHIELD_API_KEYS", "disabled")
    monkeypatch.setenv("TRUSTSHIELD_DISABLE_API_KEY", "true")
    request = _build_request()

    asyncio.run(security.enforce_api_key(request))


def test_enforce_api_key_accepts_hashed_tokens(monkeypatch):
    hashed = hashlib.sha256(b"super-secret").hexdigest()
    monkeypatch.setenv("TRUSTSHIELD_API_KEYS", f"sha256:{hashed}")

    request = _build_request(headers={"X-API-Key": "super-secret"})

    asyncio.run(security.enforce_api_key(request))


def test_api_key_loaded_from_file(monkeypatch, tmp_path):
    secret_file = tmp_path / "api_key.txt"
    secret_file.write_text("file-secret", encoding="utf-8")
    monkeypatch.delenv("TRUSTSHIELD_API_KEYS", raising=False)
    monkeypatch.setenv("TRUSTSHIELD_API_KEYS_FILE", os.fspath(secret_file))

    request = _build_request(headers={"X-API-Key": "file-secret"})

    asyncio.run(security.enforce_api_key(request))
