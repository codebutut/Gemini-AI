import asyncio
import threading
import time


class RateLimiter:
    """
    A thread-safe Token Bucket rate limiter with async support.
    """

    def __init__(self, max_requests: int, period: float, auto_refill: bool = False):
        """
        Initializes the RateLimiter.

        Args:
            max_requests (int): Maximum number of tokens (requests) allowed.
            period (float): The time period in seconds for the rate limit.
            auto_refill (bool): If True, starts a background thread to refill tokens.
        """
        self.max_requests = max_requests
        self.period = period
        self.tokens = max_requests
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.auto_refill = auto_refill
        self._stop_event = threading.Event()

        # Calculate refill rate
        self.refill_interval = self.period / self.max_requests if self.max_requests > 0 else self.period

        if self.auto_refill:
            self._refill_thread = threading.Thread(target=self._refill_loop, daemon=True)
            self._refill_thread.start()

    def _refill_loop(self) -> None:
        """Background loop to refill tokens at a steady rate."""
        while not self._stop_event.is_set():
            time.sleep(self.refill_interval)
            with self.condition:
                if self.tokens < self.max_requests:
                    self.tokens += 1
                    self.condition.notify_all()

    def stop(self) -> None:
        """Stops the refill thread."""
        self._stop_event.set()

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        """
        Attempts to acquire a token (synchronous).

        Args:
            blocking (bool): Whether to block and wait for a token.
            timeout (Optional[float]): Max time to wait if blocking.

        Returns:
            bool: True if token acquired, False otherwise.
        """
        start_time = time.monotonic()
        with self.condition:
            while self.tokens <= 0:
                if not blocking:
                    return False

                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return False
                    self.condition.wait(timeout=remaining)
                else:
                    self.condition.wait()

            self.tokens -= 1
            return True

    async def acquire_async(self) -> bool:
        """
        Attempts to acquire a token (asynchronous).
        Does not block the event loop.
        """
        while True:
            with self.lock:
                if self.tokens > 0:
                    self.tokens -= 1
                    return True

            # Wait for a bit before retrying if no tokens available
            await asyncio.sleep(self.refill_interval / 2)

    def release(self) -> None:
        """Manually releases a token back to the pool."""
        with self.condition:
            self.tokens = min(self.tokens + 1, self.max_requests)
            self.condition.notify_all()

    def remaining(self) -> int:
        """Returns the number of remaining tokens."""
        with self.lock:
            return self.tokens

    def update_limits(self, remaining: int, limit: int) -> None:
        """Updates the rate limiter state from external telemetry."""
        with self.condition:
            self.tokens = remaining
            self.max_requests = limit
            # Recalculate refill interval if limit changed
            self.refill_interval = self.period / self.max_requests if self.max_requests > 0 else self.period
            self.condition.notify_all()
