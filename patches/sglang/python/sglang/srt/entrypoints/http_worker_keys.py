"""The API keys this process's HTTP server enforces, and how they reach every tokenizer worker.

With --tokenizer-worker-num > 1, uvicorn serves the app from worker processes that it starts with
multiprocessing's spawn method. Each worker imports sglang.srt.entrypoints.http_server afresh, so the
auth middleware sits on that module's app and reads its keys from here (get_auth_keys) per request.
The keys reach a worker inside its pickled uvicorn worker target (AuthKeysWorkerTarget), in memory
through the spawn pipe, and never through the multi-tokenizer shared-memory args segment (a file in
/dev/shm that outlives a killed server) or the environment.

This module imports only the standard library: a uvicorn worker unpickles AuthKeysWorkerTarget before
it starts answering uvicorn's health pings (5 s by default), so importing it must be quick.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class AuthKeys:
    """The keys an HTTP server enforces (None = not set). repr() never shows them."""

    api_key: Optional[str] = field(default=None, repr=False)
    admin_api_key: Optional[str] = field(default=None, repr=False)


# None until set_auth_keys() runs in this process; the auth middleware then answers 503.
_auth_keys: Optional[AuthKeys] = None


def set_auth_keys(api_key: Optional[str], admin_api_key: Optional[str]) -> None:
    """Set the keys this process's HTTP server enforces.

    One tokenizer worker: http_server._setup_and_run_http_server calls it before uvicorn starts.
    Several workers: AuthKeysWorkerTarget calls it in every worker process before that worker serves.
    """
    global _auth_keys
    _auth_keys = AuthKeys(api_key=api_key, admin_api_key=admin_api_key)


def get_auth_keys() -> Optional[AuthKeys]:
    """The keys set by set_auth_keys() in this process, or None if it has not run yet."""
    return _auth_keys


class AuthKeysWorkerTarget:
    """uvicorn worker target: sets the auth keys in the worker process, then runs uvicorn's server.

    uvicorn's multi-worker supervisor pickles its target to every worker it spawns, also to a worker it
    restarts, so every worker of the server enforces the same keys.
    """

    def __init__(self, run: Callable[..., Any], keys: AuthKeys):
        self.run = run
        self.keys = keys

    def __call__(self, sockets=None):
        set_auth_keys(self.keys.api_key, self.keys.admin_api_key)
        return self.run(sockets=sockets)
