"""Deadlines must be scheduled, never spawned, from the event loop.

``threading.Timer`` here meant one ``Thread.start()`` per bounded call, executed by the event loop
itself — the very stall these deadlines exist to witness (a loop blocked in a synchronous
``Thread.start()`` cannot process its own expiry). A busy gateway runs a few hundred threads, so
that spawn also contends with every other spawn in the process.

Deadlines are now pushed onto one module-level daemon timer thread that owns a deadline heap, so
``run_bounded_async`` only pushes an entry. The companion executor-side invariant lives in
``tests/gateway/test_turn_executor_spawn_off_loop.py``.
"""

from __future__ import annotations

import asyncio
import threading
import time

from agent.deadline import run_bounded_async


def test_bounded_async_starts_no_thread_from_the_loop(monkeypatch):
    """The invariant: no thread creation on the loop thread, whatever the deadline path does."""

    async def warm():  # the one shared timer thread is created on first use, outside the window
        await run_bounded_async(asyncio.sleep(0), 5.0, label="warm")

    asyncio.run(warm())
    starts = []
    real_start = threading.Thread.start

    def recording_start(self):
        starts.append(threading.get_ident())
        real_start(self)

    async def drive() -> int:
        loop_ident = threading.get_ident()
        monkeypatch.setattr(threading.Thread, "start", recording_start)
        try:
            await run_bounded_async(asyncio.sleep(0), 5.0, label="complete")
            await run_bounded_async(asyncio.sleep(0), 0.2, label="complete-under-short-deadline")
        finally:
            monkeypatch.setattr(threading.Thread, "start", real_start)
        return loop_ident

    loop_ident = asyncio.run(drive())
    assert starts.count(loop_ident) == 0, "a deadline still started a thread on the event loop"


def test_expired_deadline_still_times_out_from_the_shared_timer_thread():
    """One timer thread serves every deadline; an expired one must still be processed by the loop."""

    async def never():
        await asyncio.Event().wait()

    async def drive() -> float:
        began = time.monotonic()
        result = await run_bounded_async(never(), 0.2, label="shared-timer", dump_on_blocked_loop=False)
        assert result.timed_out
        return time.monotonic() - began

    elapsed = asyncio.run(drive())
    assert 0.15 <= elapsed < 5.0, elapsed


def test_loop_callback_latency_stays_bounded_in_a_large_process():
    """Acceptance bound: with >150 live threads, neither a deadline push nor a turn-pool submit may
    occupy the loop long enough to delay a ``call_soon_threadsafe`` callback past 50 ms."""
    from gateway.turn_executor import _UnboundedThreadExecutor

    release = threading.Event()
    fillers = [threading.Thread(target=release.wait, args=(30.0,), daemon=True) for _ in range(160)]
    for thread in fillers:
        thread.start()
    executor = _UnboundedThreadExecutor(thread_name_prefix="latency-probe")
    try:
        assert threading.active_count() > 150

        async def drive() -> list[float]:
            loop = asyncio.get_running_loop()
            latencies = []
            for _ in range(10):
                ran_at = []
                began = time.monotonic()
                loop.call_soon_threadsafe(lambda: ran_at.append(time.monotonic() - began))
                await run_bounded_async(asyncio.sleep(0), 5.0, label="latency")
                await loop.run_in_executor(executor, lambda: None)
                while not ran_at:  # armed before the work above, so its wait is the measurement
                    await asyncio.sleep(0)
                latencies.append(ran_at[0])
            return latencies

        latencies = asyncio.run(drive())
        assert max(latencies) < 0.05, f"loop callback delayed by {max(latencies) * 1000:.0f} ms: {latencies}"
    finally:
        executor.shutdown(wait=True)
        release.set()
