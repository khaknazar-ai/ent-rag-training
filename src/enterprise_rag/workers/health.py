"""Bounded loopback challenge-response for the supervisor control thread only."""

from __future__ import annotations

import json
import os
import secrets
import socket
import time
from collections.abc import Callable
from contextlib import AbstractContextManager

HEALTH_ENV = "RAG_SUPERVISOR_HEALTH"
MAX_MESSAGES = 32
MAX_HISTORY = 64


def _messages(sock):
    for _ in range(MAX_MESSAGES):
        try:
            data, peer = sock.recvfrom(2048)
        except BlockingIOError:
            return
        except OSError as exc:
            # Windows reports ICMP port-unreachable from a closed peer as a
            # receive reset. Neither that nor an oversized packet is a setup
            # failure or fresh liveness; let the bounded TTL expire normally.
            if getattr(exc, "winerror", None) in {10040, 10054}:
                continue
            raise
        if len(data) > 1024:
            continue
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeError, RecursionError):
            continue
        if isinstance(payload, dict):
            yield payload, peer


def _send(sock, payload, peer):
    sock.sendto(json.dumps(payload, separators=(",", ":")).encode(), peer)


class SupervisorHealth(AbstractContextManager):
    def __init__(self, timeout: float, startup_timeout: float, *, started_at: float):
        self.timeout, self.startup_timeout = timeout, startup_timeout
        self.started_at = started_at
        self.responses = 0
        self.last_ack: float | None = None
        self.peer = None
        self.peer_pid: int | None = None
        self.pending: dict[str, float] = {}
        self.last_probe = float("-inf")
        self.secret = secrets.token_hex(32)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(("127.0.0.1", 0))
            self.socket.setblocking(False)
        except BaseException:
            self.socket.close()
            raise
        self.environment = {
            HEALTH_ENV: json.dumps(
                {
                    "port": self.socket.getsockname()[1],
                    "secret": self.secret,
                }
            )
        }

    def __exit__(self, *args):
        self.socket.close()

    def check(
        self,
        pid: int,
        now: float,
        *,
        pid_validator: Callable[[int], bool] | None = None,
    ) -> str | None:
        # Expiration wins over queued packets; receive time never refreshes TTL.
        if self.last_ack is not None and now - self.last_ack >= self.timeout:
            return "heartbeat_timeout"
        if self.last_ack is None and now - self.started_at >= self.startup_timeout:
            return "startup_timeout"
        for payload, peer in _messages(self.socket):
            if (
                peer[0] != "127.0.0.1"
                or payload.get("secret") != self.secret
                or type(payload.get("pid")) is not int
                or not 0 < payload["pid"] <= 0xFFFFFFFF
            ):
                continue
            reporter = payload["pid"]
            if self.peer is None and payload.get("hello") is not True:
                continue
            if self.peer is not None and (
                peer != self.peer or reporter != self.peer_pid
            ):
                continue
            if reporter != pid and (
                pid_validator is None or not pid_validator(reporter)
            ):
                continue
            if self.peer is None:
                self.peer, self.peer_pid = peer, reporter
            nonce = payload.get("ack")
            issued = self.pending.pop(nonce, None) if isinstance(nonce, str) else None
            if issued is not None and 0 <= now - issued < self.timeout:
                if self.last_ack is None or issued > self.last_ack:
                    self.last_ack = issued
                    self.responses += 1
        self.pending = {
            nonce: issued
            for nonce, issued in self.pending.items()
            if now - issued < self.timeout
        }
        if self.peer is not None and now - self.last_probe >= min(1, self.timeout / 4):
            nonce = secrets.token_hex(16)
            _send(self.socket, {"secret": self.secret, "challenge": nonce}, self.peer)
            self.pending[nonce] = now
            self.last_probe = now
            while len(self.pending) > MAX_HISTORY:
                del self.pending[next(iter(self.pending))]
        return None


class SupervisorPulse(AbstractContextManager):
    """No thread: tick must execute on the supervisor's own control loop."""

    def __init__(self, config: dict | None):
        self.socket = None
        if config is None:
            return  # Explicit direct supervisor, without a watchdog channel.
        if (
            not isinstance(config, dict)
            or type(config.get("port")) is not int
            or not 1 <= config["port"] <= 65535
            or not isinstance(config.get("secret"), str)
            or len(config["secret"]) != 64
        ):
            raise ValueError("Invalid supervisor health channel configuration")
        self.secret = config["secret"]
        self.parent = ("127.0.0.1", config["port"])
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(("127.0.0.1", 0))
            self.socket.setblocking(False)
        except BaseException:
            self.socket.close()
            raise

    @classmethod
    def from_environment(cls):
        raw = os.environ.pop(HEALTH_ENV, None)
        if raw is None:
            return cls(None)
        try:
            config = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise ValueError("Invalid supervisor health channel configuration") from exc
        if config is None:
            raise ValueError("Invalid supervisor health channel configuration")
        return cls(config)

    def tick(self):
        if self.socket is None:
            return
        _send(
            self.socket,
            {"secret": self.secret, "pid": os.getpid(), "hello": True},
            self.parent,
        )
        for payload, peer in _messages(self.socket):
            nonce = payload.get("challenge")
            if (
                peer == self.parent
                and payload.get("secret") == self.secret
                and isinstance(nonce, str)
                and len(nonce) == 32
            ):
                _send(
                    self.socket,
                    {"secret": self.secret, "pid": os.getpid(), "ack": nonce},
                    self.parent,
                )

    def __exit__(self, *args):
        if self.socket is not None:
            self.socket.close()


def responsive_wait(wait, pulse, seconds, *, monotonic=time.monotonic):
    deadline = monotonic() + seconds
    while True:
        pulse()
        remaining = deadline - monotonic()
        if remaining <= 0:
            return False
        if wait(min(0.1, remaining)):
            return True
