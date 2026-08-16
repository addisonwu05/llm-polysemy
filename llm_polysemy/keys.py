"""Thread-safe round-robin API-key pools, gathered from env vars by prefix.

Extracted verbatim from the original eval/classify scripts. Any env var whose name
starts with a pool's prefix (case-insensitive) is added to that pool, so capacity
is increased by adding suffixed keys (OPENAI_API_KEY, OPENAI_API_KEY_7, ...).
"""

import itertools
import os
import threading


class KeyPool:
    """Thread-safe round-robin pool over a list of API keys."""
    def __init__(self, keys: list[str], name: str):
        if not keys:
            raise RuntimeError(f"No API keys found for pool '{name}'")
        self._keys = keys
        self._cycle = itertools.cycle(keys)
        self._lock = threading.Lock()
        print(f"  [keys] {name}: {len(keys)} key(s)")

    def __len__(self) -> int:
        return len(self._keys)

    def next(self) -> str:
        with self._lock:
            return next(self._cycle)


def gather_keys(prefix: str) -> list[str]:
    p = prefix.lower()
    return [v.strip() for k, v in os.environ.items()
            if k.lower().startswith(p) and v.strip()]


def build_pools(prefixes: dict[str, str]) -> dict[str, KeyPool]:
    """Build a KeyPool per {pool_name: env_prefix}. Raises if any pool is empty."""
    return {name: KeyPool(gather_keys(prefix), name) for name, prefix in prefixes.items()}
