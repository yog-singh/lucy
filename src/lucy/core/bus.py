"""In-process async publish/subscribe for events (ADR 0003)."""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Final, Self

from lucy.core.events import Event


class _Closed:
    """Sentinel queued to end a subscription."""


_CLOSED: Final = _Closed()


class Subscription:
    """A stream of events matching the subscribed types.

    Iterate it with `async for`; the loop ends when the subscription or the bus
    is closed, after any events already queued have been delivered. Use it as
    an async context manager to unsubscribe on exit.
    """

    def __init__(self, bus: EventBus, event_types: tuple[type[Event], ...]) -> None:
        self._bus = bus
        self._types = event_types
        self._queue: asyncio.Queue[Event | _Closed] = asyncio.Queue()
        self._closed = False

    def matches(self, event: Event) -> bool:
        """True if this subscription wants `event` (always, when no types were given)."""
        return not self._types or isinstance(event, self._types)

    def _deliver(self, event: Event) -> None:
        if not self._closed:
            self._queue.put_nowait(event)

    def close(self) -> None:
        """Unsubscribe. Events already queued are still delivered, then iteration stops."""
        if self._closed:
            return
        self._closed = True
        self._bus._unsubscribe(self)
        self._queue.put_nowait(_CLOSED)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Event:
        item = await self._queue.get()
        if isinstance(item, _Closed):
            self._queue.put_nowait(item)  # keep later __anext__ calls ending too
            raise StopAsyncIteration
        return item

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class EventBus:
    """Fan-out of published events to every matching subscription.

    Each subscription has its own unbounded queue, so `publish` never waits on
    a slow subscriber. The cost is that a subscriber that stops reading grows
    its queue without limit; bounded queues with a drop or backpressure policy
    can be added if that becomes a problem.
    """

    def __init__(self) -> None:
        self._subscriptions: list[Subscription] = []
        self._closed = False

    def subscribe(self, *event_types: type[Event]) -> Subscription:
        """Subscribe to events of the given types (and subclasses), or to all if none given."""
        if self._closed:
            raise RuntimeError("cannot subscribe to a closed EventBus")
        subscription = Subscription(self, event_types)
        self._subscriptions.append(subscription)
        return subscription

    async def publish(self, event: Event) -> None:
        """Deliver `event` to every matching subscription without waiting for them."""
        if self._closed:
            raise RuntimeError("cannot publish to a closed EventBus")
        for subscription in list(self._subscriptions):
            if subscription.matches(event):
                subscription._deliver(event)

    def close(self) -> None:
        """Close every subscription so their `async for` loops end. Idempotent."""
        self._closed = True
        for subscription in list(self._subscriptions):
            subscription.close()

    @property
    def closed(self) -> bool:
        """Whether `close` has been called."""
        return self._closed

    def _unsubscribe(self, subscription: Subscription) -> None:
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)
