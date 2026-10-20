"""Copy of the console output in a log file, for long unattended runs."""

import sys
from pathlib import Path
from typing import TextIO


class LineLog:
    """A log file fed with console text, line by line.

    Progress bars (tqdm) redraw their line with carriage returns many times a second; the log
    gets only each line's final state, so it stays readable. stdout and stderr share one
    LineLog, since they share the console's current line."""

    def __init__(self, file: TextIO):
        self.file, self.line = file, ""

    def write(self, text: str) -> None:
        *lines, line = (self.line + text).split("\n")
        for finished in lines:
            self.file.write(_last_state(finished) + "\n")
        redraw = line.rstrip("\r").rfind("\r")  # keep only the latest redraw, to bound memory
        self.line = line[redraw:] if redraw >= 0 else line

    def close_line(self) -> None:
        """Write the unfinished line, e.g. a progress bar when a step ends."""
        if _last_state(self.line):
            self.file.write(_last_state(self.line) + "\n")
        self.line = ""
        self.file.flush()


class Tee:
    """A text stream that writes to `stream` (the console) and to a LineLog."""

    def __init__(self, stream: TextIO, log: LineLog):
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log.file.flush()

    def __getattr__(self, name: str):  # isatty, fileno, encoding, ... of the console
        return getattr(self.stream, name)


def _last_state(line: str) -> str:
    """What a line redrawn with carriage returns finally shows."""
    return line.rstrip("\r").rsplit("\r", 1)[-1].rstrip()


def log_console(path: Path) -> LineLog:
    """Copy stdout and stderr to `path` (appended) from now on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log = LineLog(open(path, "a", encoding="utf-8", buffering=1))  # noqa: SIM115 (process-long)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    return log
