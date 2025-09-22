"""Security helpers for TrustShield API.

This module centralizes the logic for enforcing an IP allow list.
It parses the configuration from environment variables and exposes
helpers that can be reused by the FastAPI application and in tests.
"""
from __future__ import annotations

import ipaddress
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

from fastapi import HTTPException, Request, status

__all__ = [
    "IPAccessControl",
    "build_ip_access_control",
    "enforce_ip_whitelist",
    "extract_client_ips",
    "reset_ip_access_control_cache",
]


logger = logging.getLogger("trustshield.api.security")

# Environment variable names
_ALLOWED_IPS_ENV = "TRUSTSHIELD_ALLOWED_IPS"
_DISABLE_ENV = "TRUSTSHIELD_DISABLE_IP_WHITELIST"

# Default set of loopback identifiers that should be trusted when the
# environment does not provide an explicit allow list.
_DEFAULT_ALLOWED_IDENTIFIERS = ("127.0.0.1", "::1", "localhost")

# Cache for the parsed configuration so we do not parse strings on every
# request. The cache is invalidated whenever the underlying environment
# value changes.
_cached_signature: Optional[str] = None
_cached_access_control: Optional["IPAccessControl"] = None


@dataclass
class IPAccessControl:
    """In-memory representation of an IP allow list.

    The allow list supports individual IP addresses, hostnames and CIDR
    ranges. Tokens are normalised on creation for efficient lookups.
    """

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
    raw = os.getenv(_ALLOWED_IPS_ENV, "")
    if not raw:
        return list(_DEFAULT_ALLOWED_IDENTIFIERS)

    tokens = [token.strip() for token in raw.split(",")]
    filtered = [token for token in tokens if token]
    return filtered or list(_DEFAULT_ALLOWED_IDENTIFIERS)


def build_ip_access_control() -> IPAccessControl:
    """Build (or reuse) the :class:`IPAccessControl` instance."""
    global _cached_signature, _cached_access_control

    signature = os.getenv(_ALLOWED_IPS_ENV, "")
    if _cached_access_control is None or signature != _cached_signature:
        tokens = _allowed_tokens_from_env()
        _cached_access_control = _normalise_tokens(tokens)
        _cached_signature = signature
        logger.debug("IP allow list rebuilt from signature '%s'", signature)

    return _cached_access_control


def reset_ip_access_control_cache() -> None:
    """Clear the cached allow list (mainly used in tests)."""
    global _cached_signature, _cached_access_control
    _cached_signature = None
    _cached_access_control = None


def whitelist_disabled() -> bool:
    flag = os.getenv(_DISABLE_ENV, "")
    return flag.lower() in {"1", "true", "yes", "on"}


def extract_client_ips(request: Request) -> List[str]:
    """Return the ordered list of client IPs for *request*.

    The first entries correspond to the ``X-Forwarded-For`` header, followed
    by the actual socket peer address. The special identifier ``testclient``
    used by Starlette's :class:`~starlette.testclient.TestClient` is mapped to
    the IPv4 loopback address so that unit tests behave like local requests.
    """
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
