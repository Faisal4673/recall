"""Terminal cosmetics: welcome banner, animated red prompt, thinking indicator.

Pure presentation -- no model or memory logic lives here. The animations run on
daemon threads and update a single line in place with ANSI escapes; if a
terminal doesn't support them the worst case is a little visual noise, never a
crash. The prompt animation redraws only its own prefix (save/restore cursor),
so whatever the user is typing is left untouched.
"""

import sys
import threading

# --- ANSI escapes ------------------------------------------------------------
RED = "\033[91m"          # bright red, for the "You" / "agent" labels
DIM = "\033[2m"
RESET = "\033[0m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
SAVE_CURSOR = "\0337"      # DEC save/restore: leaves typed text in place while
RESTORE_CURSOR = "\0338"   # the animator rewrites the line prefix beneath it

# The label printed before each model reply.
AGENT_LABEL = f"{RED}agent: {RESET}"

# A little "pixel" scanner -- one lit cell sweeping a strip of dim cells. Each
# frame is the same display width, so redrawing in place never shifts the line.
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
    print(f"  {DIM}a deepseek agent with persistent, tool-aware recall memory{RESET}")
    print(f"  {DIM}type your message and press enter · Ctrl-D to exit{RESET}\n")


class _Animation:
    """A single-line animation driven by a daemon thread.

    `render(frame)` is called each tick to draw frame number `frame`; `clear`
    (optional) is written once on stop to wipe the line.
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
    """Return a started animation showing a red 'thinking' pixel scanner.

    Use while the model streams `reasoning_content`; call `.stop()` the moment
    the actual answer (or a tool call) starts arriving.
    """
    def render(frame):
        cells = _SCANNER[frame % len(_SCANNER)]
        sys.stdout.write(f"\r{HIDE_CURSOR}{RED}thinking {cells}{RESET}")
    # Wipe the line and bring the cursor back when we're done thinking.
    clear = "\r" + " " * 24 + "\r" + SHOW_CURSOR
    return _Animation(render, 0.16, clear=clear).start()


def animated_input(label="You"):
    """Prompt for input and return the typed line.

    This uses the normal Python input prompt so terminal line editing
    (including backspace) behaves correctly.
    """
    prompt = f"{RED}{label} ❯ {RESET}"
    return input(prompt)
