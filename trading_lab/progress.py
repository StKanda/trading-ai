from __future__ import annotations

import sys
import time


def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def estimate_remaining(elapsed, completed, total):
    if completed <= 0 or total <= completed:
        return 0.0 if total <= completed else None
    return max(0.0, elapsed / completed * (total - completed))


class TerminalProgress:
    """Throttled progress output; callbacks never participate in calculations."""

    def __init__(self, enabled=False, stream=None, refresh_seconds=0.4, log_seconds=12.0):
        self.enabled = bool(enabled)
        self.stream = stream or sys.stdout
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.refresh_seconds = float(refresh_seconds)
        self.log_seconds = float(log_seconds)
        self.last_update = 0.0
        self.last_width = 0

    def update(self, message, force=False):
        if not self.enabled:
            return
        now = time.monotonic()
        interval = self.refresh_seconds if self.is_tty else self.log_seconds
        if not force and now - self.last_update < interval:
            return
        text = str(message)
        if self.is_tty:
            padded = text.ljust(self.last_width)
            self.stream.write("\r" + padded)
            self.last_width = len(text)
            if force:
                self.stream.write("\n")
                self.last_width = 0
        else:
            self.stream.write(text + "\n")
        self.stream.flush()
        self.last_update = now

    def epoch_callback(self, label, max_epochs):
        if not self.enabled:
            return None
        started = time.monotonic()

        def report(event):
            now = time.monotonic()
            if event.get("event") == "finished":
                run = int(event["epochs_run"])
                best = int(event["best_epoch"])
                ending = "early stop" if event["early_stopped"] else "completed"
                self.update(
                    f"{label} · {ending} {run}/{max_epochs} epochs · best {best} · "
                    f"elapsed {format_duration(now - started)}",
                    force=True,
                )
                return
            epoch = int(event["epoch"])
            elapsed = now - started
            remaining = estimate_remaining(elapsed, epoch, int(max_epochs))
            ending = " · early stop" if event.get("early_stopping") else ""
            self.update(
                f"{label} · epoch {epoch}/{max_epochs} · best {int(event['best_epoch'])}"
                f" · elapsed {format_duration(elapsed)} · ETA {format_duration(remaining)}{ending}",
                force=bool(event.get("early_stopping")),
            )

        return report

    def item_callback(self, label):
        if not self.enabled:
            return None
        started = time.monotonic()

        def report(event):
            completed, total = int(event["completed"]), int(event["total"])
            elapsed = time.monotonic() - started
            remaining = estimate_remaining(elapsed, completed, total)
            suffix = event.get("detail", "")
            self.update(
                f"{label} · {completed}/{total}{' · ' + str(suffix) if suffix else ''}"
                f" · elapsed {format_duration(elapsed)} · ETA {format_duration(remaining)}",
                force=completed >= total,
            )

        return report

    def bar_callback(self, label):
        if not self.enabled:
            return None
        return self.item_callback(label)