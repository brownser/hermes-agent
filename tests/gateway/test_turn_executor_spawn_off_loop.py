"""``submit`` must not spawn a thread while holding the executor lock.

``submit`` arrives on the gateway event loop (``loop.run_in_executor``) and starts a new thread
whenever no parked worker is available. ``Thread.start()`` waits until the new thread gets the
GIL, so holding the executor lock across it made every other submitter — and therefore the loop
itself — queue behind that spawn. Harvested loop-thread stacks from a busy gateway showed this
critical section as the blocking frame in the majority of them.

The critical section still has to guarantee what it always did: a concurrent ``shutdown`` must
either refuse an item or see its thread, and no unstarted thread may be left registered.
"""

from __future__ import annotations

import threading

import pytest

from gateway.turn_executor import _UnboundedThreadExecutor


def _gate_the_first_spawn(monkeypatch, gate: threading.Event, entered: threading.Event, prefix: str) -> None:
    """Make the executor's first thread spawn block until ``gate`` is set."""
    real_start = threading.Thread.start

    def start(self):
        if self.name.startswith(f"{prefix}_") and not entered.is_set():
            entered.set()
            assert gate.wait(10.0), "the test never released the gated spawn"
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)


def test_a_slow_spawn_does_not_serialise_another_submitter(monkeypatch):
    """The regression pin: a spawn in flight no longer holds the lock other submitters need."""
    executor = _UnboundedThreadExecutor(thread_name_prefix="hermes-test")
    gate, entered = threading.Event(), threading.Event()
    _gate_the_first_spawn(monkeypatch, gate, entered, "hermes-test")
    try:
        first = threading.Thread(target=lambda: executor.submit(lambda: "first").result(10.0), daemon=True)
        first.start()
        assert entered.wait(10.0), "the first submit never reached its spawn"

        outcome = {}
        second = threading.Thread(target=lambda: outcome.setdefault("future", executor.submit(lambda: "second")), daemon=True)
        second.start()
        second.join(2.0)
        blocked = second.is_alive()
        gate.set()

        assert not blocked, (
            "a second submit blocked behind another submitter's thread spawn: submit() still holds "
            "the executor lock across Thread.start(), so the event loop queues behind every spawn"
        )
        assert outcome["future"].result(10.0) == "second"
        first.join(10.0)
    finally:
        gate.set()
        executor.shutdown(wait=True)


def test_a_failed_spawn_leaves_no_unstarted_thread_registered(monkeypatch):
    """At the OS thread limit start() raises; nothing unstarted may stay in ``_threads``.

    The thread is registered BEFORE it is started, so a spawn that fails has to deregister itself —
    otherwise ``_stop_pool`` hands ``shutdown`` a thread it cannot join.
    """
    executor = _UnboundedThreadExecutor(thread_name_prefix="hermes-test")
    real_start = threading.Thread.start
    fail_spawn = {"on": True}

    def start(self):
        if self.name.startswith("hermes-test_") and fail_spawn["on"]:
            raise RuntimeError("can't start new thread")
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with pytest.raises(RuntimeError):
            executor.submit(lambda: None)
        assert executor._threads == set(), "an unstarted thread stayed registered; it could not be joined"
    finally:
        fail_spawn["on"] = False
    assert executor.submit(lambda: "after").result(10.0) == "after"
    executor.shutdown(wait=True)


def test_shutdown_never_exposes_a_spawn_it_cannot_join(monkeypatch):
    """``_stop_pool`` joins ``_threads`` after shutdown returns, so every exposed thread must be started."""
    executor = _UnboundedThreadExecutor(thread_name_prefix="hermes-test")
    gate, entered = threading.Event(), threading.Event()
    _gate_the_first_spawn(monkeypatch, gate, entered, "hermes-test")
    hold = threading.Event()  # keeps the item's thread BUSY, so it stays in _threads
    submitter = threading.Thread(target=lambda: executor.submit(hold.wait, 10.0), daemon=True)
    snapshot = {}
    try:
        submitter.start()
        assert entered.wait(10.0), "the submit never reached its spawn"

        def close():
            executor.shutdown(wait=False)
            snapshot["threads"] = list(executor._threads)

        closer = threading.Thread(target=close, daemon=True)
        closer.start()
        closer.join(0.3)
        still_waiting = closer.is_alive()
        gate.set()
        closer.join(10.0)

        assert still_waiting, "shutdown returned while a thread was still being spawned"
        assert snapshot["threads"], "the in-flight item's thread was not exposed for the quiesce join"
        assert all(thread.ident is not None for thread in snapshot["threads"])
    finally:
        gate.set()
        hold.set()
        submitter.join(10.0)
        executor.shutdown(wait=True)
