"""
Token-bucket rate limiter for SwarmIQ v2.

Enforces per-minute request (RPM) and token (TPM) limits using a
sliding 60-second window backed by a thread-safe deque.
"""
from __future__ import annotations

import time
import threading
from collections import deque


class TokenBucketRateLimiter:
    """Thread-safe sliding-window rate limiter for LLM API calls.

    Args:
        rpm_limit: Maximum requests per 60-second window.
        tpm_limit: Maximum tokens per 60-second window.
    """

    _WINDOW_SECONDS: float = 60.0

    def __init__(self, rpm_limit: int, tpm_limit: int) -> None:
        self._rpm_limit = rpm_limit
        self._tpm_limit = tpm_limit
        self._lock = threading.Lock()
        # Each entry: (timestamp: float, tokens: int)
        self._requests: deque[tuple[float, int]] = deque()

    def wait_if_needed(self, estimated_tokens: int = 1500) -> None:
        """Block until the next request can be made within rate limits.

        Checks both RPM and TPM ceilings against the sliding 60-second window.
        If RPM would be exceeded, sleeps until the oldest request falls outside
        the window. If TPM would be exceeded, sleeps 3 seconds.
        After waiting, records the new request.

        Args:
            estimated_tokens: Estimated token count for the upcoming request.
        """
        with self._lock:
            self._enforce_limits(estimated_tokens)
            self._requests.append((time.monotonic(), estimated_tokens))

    def _enforce_limits(self, estimated_tokens: int) -> None:
        """Purge stale entries and sleep if limits would be exceeded."""
        while True:
            now = time.monotonic()
            self._purge_old(now)

            rpm_ok = len(self._requests) < self._rpm_limit
            tpm_ok = self._window_tokens() + estimated_tokens <= self._tpm_limit

            if rpm_ok and tpm_ok:
                break

            if not rpm_ok and self._requests:
                # Sleep until the oldest request falls outside the window
                oldest_ts = self._requests[0][0]
                sleep_for = self._WINDOW_SECONDS - (now - oldest_ts) + 0.01
                if sleep_for > 0:
                    self._lock.release()
                    try:
                        time.sleep(sleep_for)
                    finally:
                        self._lock.acquire()
                continue

            if not tpm_ok:
                self._lock.release()
                try:
                    time.sleep(3)
                finally:
                    self._lock.acquire()
                continue

            break

    def _purge_old(self, now: float) -> None:
        """Remove entries older than the sliding window."""
        cutoff = now - self._WINDOW_SECONDS
        while self._requests and self._requests[0][0] <= cutoff:
            self._requests.popleft()

    def _window_tokens(self) -> int:
        """Sum of tokens in the current window."""
        return sum(tokens for _, tokens in self._requests)


# Module-level singleton for Groq API rate limits
groq_limiter = TokenBucketRateLimiter(rpm_limit=25, tpm_limit=10_000)
