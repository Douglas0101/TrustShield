"""Security helpers for TrustShield API.

This module centralises the logic for enforcing an IP allow list and API key
based authentication.  Configuration is sourced from environment variables (or
Docker secrets exposed via the ``*_FILE`` convention) and cached in-memory to
avoid recomputing the same values on every request.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from fastapi import HTTPException, Request, status

__all__ = [
    "APIKeyConfig",
    "IPAccessControl",
    "build_api_key_config",
    "build_ip_access_control",
    "enforce_api_key",
    "enforce_ip_whitelist",
    "extract_client_ips",
    "reset_api_key_cache",
    "reset_ip_access_control_cache",
]


logger = logging.getLogger("trustshield.api.security")

# Environment variable names -------------------------------------------------
_ALLOWED_IPS_ENV = "TRUSTSHIELD_ALLOWED_IPS"
_DISABLE_WHITELIST_ENV = "TRUSTSHIELD_DISABLE_IP_WHITELIST"
_API_KEYS_ENV = "TRUSTSHIELD_API_KEYS"
_DISABLE_API_KEY_ENV = "TRUSTSHIELD_DISABLE_API_KEY"
_API_KEY_HEADER_ENV = "TRUSTSHIELD_API_KEY_HEADER"
_API_KEY_QUERY_ENV = "TRUSTSHIELD_API_KEY_QUERY"
_PUBLIC_PATHS_ENV = "TRUSTSHIELD_PUBLIC_PATHS"

# Default identifiers that should be trusted when the environment does not
# provide an explicit allow list.  Besides loopback addresses we also trust the
# common private network ranges used by Docker (172.16.0.0/12) and home/office
# LANs (10.0.0.0/8 and 192.168.0.0/16), as well as the special hostname
# ``host.docker.internal`` exposed by Docker Desktop. This makes the API usable
# out of the box in local development environments while still requiring an
# explicit allow list for public networks.
_DEFAULT_ALLOWED_IDENTIFIERS = (
    "127.0.0.1",
    "::1",
    "localhost",
    "host.docker.internal",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
)

_DEFAULT_PUBLIC_PATHS = (
    "/healthz",
    "/readyz",
    "/status",
    "/docs",
    "/redoc",
    "/openapi.json",
)

# Cache for the parsed configuration so we do not parse strings on every
# request. The cache is invalidated whenever the underlying environment value
# changes.
_cached_ip_signature: Optional[str] = None
_cached_access_control: Optional["IPAccessControl"] = None

_cached_api_key_signature: Optional[str] = None
_cached_api_key_config: Optional["APIKeyConfig"] = None
_logged_empty_api_keys = False


def _read_env_or_file(name: str) -> str:
    """Return the value of ``name`` honouring the ``*_FILE`` convention."""

    file_name = f"{name}_FILE"
    file_path = os.getenv(file_name)
    if file_path:
        try:
            return Path(file_path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.error("Unable to read secret file for %s: %s", name, exc)

    return os.getenv(name, "").strip()


@dataclass
class IPAccessControl:
    """In-memory representation of an IP allow list."""

    allows_all: bool
    hosts: set[str]
    networks: List[ipaddress._BaseNetwork]

    def is_allowed(self, candidate: str) -> bool:
        """Return ``True`` when *candidate* is part of the allow list."""

        if self.allows_all:
            return True

        if not candidate:
            return False

        value = candidate.strip()
        if not value:
            return False

        lowered = value.lower()
        if lowered in self.hosts:
            return True

        try:
            ip_obj = ipaddress.ip_address(value)
        except ValueError:
            return lowered in self.hosts

        if str(ip_obj) in self.hosts:
            return True

        return any(ip_obj in network for network in self.networks)


@dataclass
class APIKeyConfig:
    """Configuration required to enforce API key authentication."""

    hashed_keys: set[str]
    header_name: str
    query_param: str
    public_paths: Sequence[str]


def _normalise_tokens(raw_tokens: Iterable[str]) -> IPAccessControl:
    allows_all = False
    hosts: set[str] = set()
    networks: List[ipaddress._BaseNetwork] = []

    for raw in raw_tokens:
        token = raw.strip()
        if not token:
            continue

        lowered = token.lower()
        if lowered == "*":
            allows_all = True
            continue

        if lowered == "localhost":
            hosts.update({"localhost", "127.0.0.1", "::1"})
            continue

        try:
            network = ipaddress.ip_network(token, strict=False)
        except ValueError:
            try:
                ip_obj = ipaddress.ip_address(token)
            except ValueError:
                hosts.add(lowered)
            else:
                hosts.add(str(ip_obj))
        else:
            networks.append(network)

    return IPAccessControl(allows_all=allows_all, hosts=hosts, networks=networks)


def _allowed_tokens_from_env() -> List[str]:
    raw = _read_env_or_file(_ALLOWED_IPS_ENV)
    if not raw:
        return list(_DEFAULT_ALLOWED_IDENTIFIERS)

    tokens = [token.strip() for token in raw.split(",")]
    filtered = [token for token in tokens if token]
    return filtered or list(_DEFAULT_ALLOWED_IDENTIFIERS)


def build_ip_access_control() -> IPAccessControl:
    """Build (or reuse) the :class:`IPAccessControl` instance."""

    global _cached_ip_signature, _cached_access_control

    signature = _read_env_or_file(_ALLOWED_IPS_ENV)
    if _cached_access_control is None or signature != _cached_ip_signature:
        tokens = _allowed_tokens_from_env()
        _cached_access_control = _normalise_tokens(tokens)
        _cached_ip_signature = signature
        logger.debug("IP allow list rebuilt from signature '%s'", signature)

    return _cached_access_control


def reset_ip_access_control_cache() -> None:
    """Clear the cached allow list (mainly used in tests)."""

    global _cached_ip_signature, _cached_access_control
    _cached_ip_signature = None
    _cached_access_control = None


def whitelist_disabled() -> bool:
    flag = os.getenv(_DISABLE_WHITELIST_ENV, "")
    return flag.lower() in {"1", "true", "yes", "on"}


def extract_client_ips(request: Request) -> List[str]:
    """Return the ordered list of client IPs for *request*."""

    forwarded_for = request.headers.get("x-forwarded-for")
    candidates: List[str] = []

    if forwarded_for:
        forwarded_values = [value.strip() for value in forwarded_for.split(",")]
        candidates.extend(value for value in forwarded_values if value)

    if request.client and request.client.host:
        host = request.client.host
        candidates.append(host)
        if host == "testclient":
            candidates.append("127.0.0.1")

    return candidates


async def enforce_ip_whitelist(request: Request) -> None:
    """Ensure that the incoming request originates from an allowed IP."""

    if whitelist_disabled():
        return

    controller = build_ip_access_control()
    if controller.allows_all:
        return

    candidates = extract_client_ips(request)
    for candidate in candidates:
        if controller.is_allowed(candidate):
            return

    logger.warning(
        "Rejected request from unauthorised client IP(s): %s",
        ", ".join(candidates) or "<unknown>",
    )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Client IP address is not allowed to access this resource.",
    )


def _hash_api_key(value: str) -> str:
    digest = hashlib.sha256()
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _parse_api_keys(raw_tokens: Iterable[str]) -> set[str]:
    hashed: set[str] = set()
    for raw in raw_tokens:
        token = raw.strip()
        if not token:
            continue
        if token.lower().startswith("sha256:"):
            hashed.add(token.split(":", 1)[1].strip())
        else:
            hashed.add(_hash_api_key(token))
    return hashed


def _public_paths_from_env() -> Sequence[str]:
    raw = os.getenv(_PUBLIC_PATHS_ENV, "")
    if not raw:
        return _DEFAULT_PUBLIC_PATHS
    paths = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(paths) or _DEFAULT_PUBLIC_PATHS


def build_api_key_config() -> APIKeyConfig:
    """Return the cached API key configuration."""

    global _cached_api_key_signature, _cached_api_key_config, _logged_empty_api_keys

    raw_keys = _read_env_or_file(_API_KEYS_ENV)
    header_name = os.getenv(_API_KEY_HEADER_ENV, "X-API-Key")
    query_param = os.getenv(_API_KEY_QUERY_ENV, "api_key")
    public_paths = _public_paths_from_env()

    hashed_keys = _parse_api_keys(raw_keys.split(","))
    signature_components = [
        ",".join(sorted(hashed_keys)),
        header_name.lower(),
        query_param.lower(),
        ",".join(public_paths),
    ]
    signature = "|".join(signature_components)

    if _cached_api_key_config is None or signature != _cached_api_key_signature:
        _cached_api_key_config = APIKeyConfig(
            hashed_keys=hashed_keys,
            header_name=header_name,
            query_param=query_param,
            public_paths=public_paths,
        )
        _cached_api_key_signature = signature
        _logged_empty_api_keys = False
        logger.debug("API key configuration refreshed (header=%s, query=%s)", header_name, query_param)

    if not _cached_api_key_config.hashed_keys and not _logged_empty_api_keys:
        logger.warning(
            "API key authentication is enabled but no keys were provided. "
            "Requests will be accepted."
        )
        _logged_empty_api_keys = True

    return _cached_api_key_config


def reset_api_key_cache() -> None:
    """Clear the cached API key configuration (used in tests)."""

    global _cached_api_key_signature, _cached_api_key_config, _logged_empty_api_keys
    _cached_api_key_signature = None
    _cached_api_key_config = None
    _logged_empty_api_keys = False


def api_key_disabled() -> bool:
    flag = os.getenv(_DISABLE_API_KEY_ENV, "")
    return flag.lower() in {"1", "true", "yes", "on"}


def _is_public_path(path: str, public_paths: Sequence[str]) -> bool:
    for candidate in public_paths:
        if not candidate:
            continue
        if candidate.endswith("*"):
            prefix = candidate[:-1]
            if path.startswith(prefix):
                return True
        elif path == candidate:
            return True
    return False


async def enforce_api_key(request: Request) -> None:
    """Validate the API key provided via header or query parameter."""

    if api_key_disabled():
        return

    config = build_api_key_config()
    if not config.hashed_keys:
        return

    if _is_public_path(request.url.path, config.public_paths):
        return

    header_value = request.headers.get(config.header_name)
    query_value = request.query_params.get(config.query_param)
    candidate = header_value or query_value

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide it via header or query parameter.",
        )

    hashed_candidate = _hash_api_key(candidate)
    for stored in config.hashed_keys:
        if hmac.compare_digest(stored, hashed_candidate):
            return

    logger.warning("Rejected request with invalid API key from %s", request.client)
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key provided.",
    )
