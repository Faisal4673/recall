"""Terminal cosmetics: welcome banner, animated prompt, thinking indicator.

Animations run on daemon threads and redraw a single line in place with ANSI
escapes.
"""

import sys
import threading

RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
SAVE_CURSOR = "\0337"       # DEC save/restore: keeps typed text in place while
RESTORE_CURSOR = "\0338"    # the animator rewrites the prefix beneath it

AGENT_LABEL = f"{RED}agent: {RESET}"

# One lit cell sweeping a strip of dim cells; every frame is the same width.
_SCANNER = ["▰▱▱▱", "▱▰▱▱", "▱▱▰▱", "▱▱▱▰", "▱▱▰▱", "▱▰▱▱"]

_BANNER = r"""
        ▗▄▄▄▄▄▄▄▖
        ▌ ▣  ▣ ▐     ██████╗ ███████╗ ██████╗ █████╗ ██╗     ██╗
        ▌  ▀▀▀ ▐     ██╔══██╗██╔════╝██╔════╝██╔══██╗██║     ██║
        ▝▀▀╤╤▀▀▘     ██████╔╝█████╗  ██║     ███████║██║     ██║
          ▟▙ ▟▙      ██╔══██╗██╔══╝  ██║     ██╔══██║██║     ██║
                     ██║  ██║███████╗╚██████╗██║  ██║███████╗███████╗
                     ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝
"""


def welcome():
    """Print the pixel-art banner and a short welcome message."""
    print(f"{RED}{_BANNER}{RESET}")
    print(f"  {DIM}a deepseek agent with persistent recall memory{RESET}")
    print(f"  {DIM}type your message and press enter · Ctrl-D to exit{RESET}\n")


class _Animation:
    """A single-line animation driven by a daemon thread.

    `render(frame)` draws each tick; `clear` is written once on stop.
    """

    def __init__(self, render, interval, clear=""):
        self._render = render
        self._interval = interval
        self._clear = clear
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        def loop():
            frame = 0
            while not self._stop.is_set():
                self._render(frame)
                sys.stdout.flush()
                frame += 1
                if self._stop.wait(self._interval):  # interruptible sleep
                    break
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._clear:
            sys.stdout.write(self._clear)
            sys.stdout.flush()


def thinking_indicator():
    """Started animation for while the model streams `reasoning_content`.

    Call `.stop()` once the answer (or a tool call) starts arriving.
    """
    def render(frame):
        cells = _SCANNER[frame % len(_SCANNER)]
        sys.stdout.write(f"\r{HIDE_CURSOR}{RED}thinking {cells}{RESET}")
    clear = "\r" + " " * 24 + "\r" + SHOW_CURSOR
    return _Animation(render, 0.16, clear=clear).start()


def animated_input(label="You"):
    """Prompt for input via input() so line editing works, return the line."""
    prompt = f"{RED}{label} ❯ {RESET}"
    return input(prompt)
