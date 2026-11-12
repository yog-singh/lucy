import asyncio

import pytest

from lucy.core.bus import EventBus
from lucy.core.events import Event, Interrupt, TextDelta, UserMessage


async def collect(sub: object) -> list[Event]:
    return [event async for event in sub]  # type: ignore[attr-defined]


async def test_delivers_to_unfiltered_subscriber() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    events = [UserMessage(session_id="s", text="hi"), TextDelta(session_id="s", text="a")]
    for event in events:
        await bus.publish(event)
    bus.close()
    assert await collect(sub) == events


async def test_filters_by_type() -> None:
    bus = EventBus()
    sub = bus.subscribe(TextDelta, Interrupt)
    await bus.publish(UserMessage(session_id="s", text="hi"))
    delta = TextDelta(session_id="s", text="a")
    interrupt = Interrupt(session_id="s")
    await bus.publish(delta)
    await bus.publish(interrupt)
    bus.close()
    assert await collect(sub) == [delta, interrupt]


async def test_multiple_subscribers_each_get_events() -> None:
    bus = EventBus()
    first, second = bus.subscribe(), bus.subscribe()
    event = UserMessage(session_id="s", text="hi")
    await bus.publish(event)
    bus.close()
    assert await collect(first) == [event]
    assert await collect(second) == [event]


async def test_close_ends_waiting_loops() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    task = asyncio.create_task(collect(sub))
    await asyncio.sleep(0)
    event = Interrupt(session_id="s")
    await bus.publish(event)
    bus.close()
    assert await asyncio.wait_for(task, timeout=1) == [event]


async def test_slow_subscriber_does_not_block_publisher() -> None:
    bus = EventBus()
    bus.subscribe()  # never read
    for i in range(1000):
        await asyncio.wait_for(bus.publish(TextDelta(session_id="s", text=str(i))), timeout=1)


async def test_context_manager_unsubscribes() -> None:
    bus = EventBus()
    async with bus.subscribe() as sub:
        await bus.publish(Interrupt(session_id="s"))
    await bus.publish(Interrupt(session_id="s"))  # not delivered: unsubscribed
    assert len(await collect(sub)) == 1


async def test_closed_bus_rejects_publish_and_subscribe() -> None:
    bus = EventBus()
    bus.close()
    bus.close()  # idempotent
    with pytest.raises(RuntimeError):
        await bus.publish(Interrupt(session_id="s"))
    with pytest.raises(RuntimeError):
        bus.subscribe()
