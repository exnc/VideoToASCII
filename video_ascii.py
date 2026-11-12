#!/usr/bin/env python3
"""Play a video in color, as ASCII art, directly in the terminal.

Usage:
    python video_ascii.py path/to/video.mp4
    python video_ascii.py video.mp4 --chars "+"      # "+" characters only
    python video_ascii.py video.mp4 --width 160 --loop

Controls:
    Left / Right     -/+ 10 seconds
    Up / Down        +/- 1 minute
    PgUp / PgDn      +/- 10 minutes
    Home / End       start / end
    0-9              jump to 0%, 10%, ... 90%
    Space            pause / resume
    q / Esc          quit

Dependencies: numpy, opencv-python (video decoding).
The terminal must support 24-bit color (Windows Terminal, iTerm2, most
modern Linux terminals).
"""

import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np

# Ramp from darkest to brightest.
DEFAULT_RAMP = " .:-=+*#%@"

# Height/width ratio of a terminal cell (~2:1).
CELL_ASPECT = 0.5

# Lines reserved at the bottom: progress bar + status line.
UI_ROWS = 2

BAR_FILL = (230, 180, 80)
PARTIAL_BLOCKS = " ▏▎▍▌▋▊▉"

# --- Vectorized ANSI encoding of a frame -----------------------------------
# Each cell is 20 bytes of fixed width:
#   ESC[38;2;RRR;GGG;BBBm<char>
# Digits are zero-padded (terminals accept that), so everything can be
# built with numpy without a Python loop.
_PREFIX = np.frombuffer(b"\x1b[38;2;", dtype=np.uint8)
_DIGITS = np.array([list(f"{i:03d}".encode()) for i in range(256)], dtype=np.uint8)
CELL_BYTES = 20


def encode_frame(rgb: np.ndarray, chars: np.ndarray) -> bytes:
    """rgb: (h, w, 3) uint8, chars: (h, w) uint8 (ASCII codes)."""
    h, w, _ = rgb.shape
    cells = np.empty((h, w, CELL_BYTES), dtype=np.uint8)
    cells[..., 0:7] = _PREFIX
    cells[..., 7:10] = _DIGITS[rgb[..., 0]]
    cells[..., 10] = ord(";")
    cells[..., 11:14] = _DIGITS[rgb[..., 1]]
    cells[..., 14] = ord(";")
    cells[..., 15:18] = _DIGITS[rgb[..., 2]]
    cells[..., 18] = ord("m")
    cells[..., 19] = chars

    # Only emit the color code when it differs from the previous cell on
    # the same row -> far fewer bytes to write.
    changed = np.ones((h, w), dtype=bool)
    changed[:, 1:] = np.any(rgb[:, 1:] != rgb[:, :-1], axis=2)
    # A space needs no color; the cell after it must re-emit its own.
    space = chars == ord(" ")
    changed[:, 1:] |= space[:, :-1]
    changed &= ~space

    mask = np.empty((h, w, CELL_BYTES), dtype=bool)
    mask[..., :19] = changed[..., None]
    mask[..., 19] = True

    # Add a '\n' at the end of each row (except the last one).
    flat = np.empty((h, w * CELL_BYTES + 1), dtype=np.uint8)
    flat[:, :-1] = cells.reshape(h, -1)
    flat[:, -1] = ord("\n")
    fmask = np.empty((h, w * CELL_BYTES + 1), dtype=bool)
    fmask[:, :-1] = mask.reshape(h, -1)
    fmask[:, -1] = True
    fmask[-1, -1] = False

    return flat[fmask].tobytes()


# --- Image processing ------------------------------------------------------
def frame_to_cells(frame_bgr, cols, rows, ramp, quant, boost):
    small = cv2.resize(frame_bgr, (cols, rows), interpolation=cv2.INTER_AREA)
    rgb = small[..., ::-1]  # BGR -> RGB

    # Perceived luminance -> index into the character ramp.
    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    idx = (lum * (len(ramp) - 1) / 255.0 + 0.5).astype(np.intp)
    chars = ramp[idx]

    color = rgb.astype(np.float32)
    if boost != 1.0:
        # Thin glyphs cover few pixels: brighten a little.
        color = color * boost
    color = np.clip(color, 0, 255).astype(np.uint8)
    if quant > 1:
        # Quantization: more identical neighbor cells -> lighter output.
        color = (color // quant) * quant + quant // 2
        color = np.minimum(color, 255).astype(np.uint8)
    return color, chars


def fit_size(vw, vh, term_cols, term_rows, width=None):
    max_rows = max(1, term_rows - UI_ROWS)
    cols = min(width or term_cols, term_cols)
    rows = int(round(cols * vh / vw * CELL_ASPECT))
    if rows > max_rows:
        rows = max_rows
        cols = int(round(rows * vw / vh / CELL_ASPECT))
    return max(1, cols), max(1, rows)


# --- Progress bar ----------------------------------------------------------
def fmt_time(seconds):
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def progress_bar(fraction, width):
    """Bar of `width` cells, with 1/8 cell precision on the leading edge."""
    fraction = min(max(fraction, 0.0), 1.0)
    eighths = int(round(fraction * width * 8))
    full, part = divmod(eighths, 8)
    filled = "█" * full
    edge = PARTIAL_BLOCKS[part] if full < width and part else ""
    empty = "░" * (width - full - len(edge))
    r, g, b = BAR_FILL
    return f"\x1b[38;2;{r};{g};{b}m{filled}{edge}\x1b[90m{empty}"


# --- Non-blocking keyboard -------------------------------------------------
class Keyboard:
    """Reads keypresses without blocking; returns key names."""

    WIN_SPECIAL = {"H": "up", "P": "down", "K": "left", "M": "right",
                   "I": "pgup", "Q": "pgdn", "G": "home", "O": "end"}
    VT_SEQ = {"[A": "up", "[B": "down", "[C": "right", "[D": "left",
              "[5~": "pgup", "[6~": "pgdn", "[H": "home", "[F": "end",
              "[1~": "home", "[4~": "end", "OH": "home", "OF": "end"}

    def __enter__(self):
        if os.name == "nt":
            import msvcrt
            self._msvcrt = msvcrt
        elif sys.stdin.isatty():
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if os.name != "nt" and getattr(self, "_old", None) is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read(self):
        if os.name == "nt":
            return self._read_windows()
        if getattr(self, "_old", None) is None:
            return []
        return self._read_posix()

    def _read_windows(self):
        keys = []
        m = self._msvcrt
        while m.kbhit():
            ch = m.getwch()
            if ch in ("\x00", "\xe0"):
                name = self.WIN_SPECIAL.get(m.getwch())
                if name:
                    keys.append(name)
            else:
                keys.append(self._plain(ch))
        return keys

    def _read_posix(self):
        import select
        if not select.select([self._fd], [], [], 0)[0]:
            return []
        data = os.read(self._fd, 64).decode("utf-8", "ignore")
        keys, i = [], 0
        while i < len(data):
            if data[i] == "\x1b":
                for seq, name in self.VT_SEQ.items():
                    if data.startswith(seq, i + 1):
                        keys.append(name)
                        i += 1 + len(seq)
                        break
                else:
                    keys.append("esc")
                    i += 1
            else:
                keys.append(self._plain(data[i]))
                i += 1
        return keys

    @staticmethod
    def _plain(ch):
        if ch == " ":
            return "space"
        if ch == "\x1b":
            return "esc"
        if ch == "\x03":
            return "q"  # Ctrl+C in raw mode
        return ch.lower()


# --- Player ----------------------------------------------------------------
class Player:
    def __init__(self, path, ramp_str, width, loop, quant, boost):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            sys.exit(f"Cannot open video: {path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps = fps if 1 <= fps <= 240 else 30.0
        self.frame_time = 1.0 / self.fps
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ramp = np.frombuffer(ramp_str.encode("ascii"), dtype=np.uint8)
        self.name = os.path.basename(path)
        self.width, self.loop, self.quant, self.boost = width, loop, quant, boost

        self.out = sys.stdout.buffer
        self.pos = 0            # index of the next frame to read
        self.paused = False
        self.t0 = time.perf_counter()  # wall-clock time of frame 0
        self.last_frame = None
        self.last_size = None
        self.real_fps = 0.0
        self._fps_t0, self._fps_n = time.perf_counter(), 0

    # -- timing / seeking --
    @property
    def duration(self):
        return self.total / self.fps if self.total else 0.0

    def current_time(self):
        return max(0, self.pos - 1) / self.fps

    def resync(self):
        self.t0 = time.perf_counter() - self.pos * self.frame_time

    def seek_frame(self, target):
        last = max(0, self.total - 1) if self.total else max(0, target)
        target = int(min(max(target, 0), last))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self.pos = target
        self.resync()
        self.show_next()  # show the new position right away, even when paused

    def seek_seconds(self, delta):
        self.seek_frame(self.pos - 1 + delta * self.fps)

    def handle_key(self, key):
        if key in ("q", "esc"):
            return False
        if key == "space":
            self.paused = not self.paused
            if not self.paused:
                self.resync()
            self.render()
        elif key == "left":
            self.seek_seconds(-10)
        elif key == "right":
            self.seek_seconds(10)
        elif key == "up":
            self.seek_seconds(60)
        elif key == "down":
            self.seek_seconds(-60)
        elif key == "pgup":
            self.seek_seconds(600)
        elif key == "pgdn":
            self.seek_seconds(-600)
        elif key == "home":
            self.seek_frame(0)
        elif key == "end" and self.total:
            self.seek_frame(self.total - 1)
        elif key.isdigit() and self.total:
            self.seek_frame(self.total * int(key) / 10)
        return True

    # -- decoding --
    def show_next(self):
        """Decode the next frame and draw it. Returns False at end of video."""
        ok, frame = self.cap.read()
        if not ok:
            return False
        self.pos += 1
        self.last_frame = frame
        self.render()
        return True

    # -- drawing --
    def render(self):
        if self.last_frame is None:
            return
        term = shutil.get_terminal_size((120, 40))
        cols, rows = fit_size(self.vw, self.vh, term.columns, term.lines, self.width)
        out = []
        if (cols, rows, term.columns, term.lines) != self.last_size:
            out.append(b"\x1b[0m\x1b[2J")
            self.last_size = (cols, rows, term.columns, term.lines)

        color, chars = frame_to_cells(self.last_frame, cols, rows, self.ramp,
                                      self.quant, self.boost)
        pad = (term.columns - cols) // 2
        body = encode_frame(color, chars)
        if pad:
            body = body.replace(b"\n", b"\n\x1b[%dG" % (pad + 1))
        out.append(b"\x1b[H" + (b"\x1b[%dG" % (pad + 1) if pad else b"") + body + b"\x1b[0m")

        now = time.perf_counter()
        self._fps_n += 1
        if now - self._fps_t0 >= 1.0:
            self.real_fps = self._fps_n / (now - self._fps_t0)
            self._fps_t0, self._fps_n = now, 0

        # Progress bar line: "▶ [████▌░░░░] 01:00:29 / 03:00:59"
        icon = "❚❚" if self.paused else "▶ "
        times = f" {fmt_time(self.current_time())} / {fmt_time(self.duration)}"
        bar_w = max(5, term.columns - len(icon) - len(times) - 4)
        frac = (self.pos - 1) / max(1, self.total - 1) if self.total else 0.0
        bar_line = f" {icon}[{progress_bar(frac, bar_w)}\x1b[0m]{times}"

        status = (f" {self.name}  ·  {cols}x{rows}  ·  {self.real_fps:4.1f}/{self.fps:.0f} fps"
                  f"  ·  ←→ 10s  ↑↓ 1min  PgUp/PgDn 10min  0-9 jump  Space pause  q quit")
        status = status[: term.columns - 1]

        out.append(b"\x1b[%d;1H\x1b[2K" % (term.lines - 1) + bar_line.encode("utf-8"))
        out.append(b"\x1b[%d;1H\x1b[2K\x1b[90m" % term.lines
                   + status.encode("utf-8", "replace") + b"\x1b[0m")
        self.out.write(b"".join(out))
        self.out.flush()

    # -- main loop --
    def run(self):
        if os.name == "nt":
            os.system("")  # enable ANSI escape processing on Windows
        self.out.write(b"\x1b[?1049h\x1b[?25l\x1b[2J")  # alt screen, hide cursor
        self.out.flush()
        try:
            with Keyboard() as kb:
                self.resync()
                while True:
                    if not all(self.handle_key(k) for k in kb.read()):
                        break

                    if self.paused:
                        size = shutil.get_terminal_size((120, 40))
                        if self.last_size and size != self.last_size[2:]:
                            self.render()
                        time.sleep(0.03)
                        continue

                    # Behind schedule? Skip frames without decoding them.
                    due = int((time.perf_counter() - self.t0) / self.frame_time)
                    while self.pos < due - 1 and self.cap.grab():
                        self.pos += 1

                    if not self.show_next():
                        if self.loop and self.pos > 0:
                            self.seek_frame(0)
                            continue
                        break

                    delay = self.t0 + self.pos * self.frame_time - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        except KeyboardInterrupt:
            pass
        finally:
            self.cap.release()
            self.out.write(b"\x1b[0m\x1b[?25h\x1b[?1049l")
            self.out.flush()


def main():
    p = argparse.ArgumentParser(
        description="Colored ASCII video player for the terminal.",
        epilog="Controls: Left/Right -/+10s, Up/Down +/-1min, PgUp/PgDn +/-10min, "
               "Home/End, 0-9 jump to 0-90%%, Space pause, q/Esc quit.")
    p.add_argument("path", help="path to the video")
    p.add_argument("--chars", default=DEFAULT_RAMP,
                   help='characters from darkest to brightest (default: "%(default)s"). '
                        'Use --chars "+" for "+" only.')
    p.add_argument("--width", type=int, default=None, help="max width in columns")
    p.add_argument("--loop", action="store_true", help="loop playback")
    p.add_argument("--quant", type=int, default=4,
                   help="color quantization step (1 = none, default 4)")
    p.add_argument("--boost", type=float, default=None,
                   help="color brightness multiplier (default: 1.0, or 1.4 with a single character)")
    a = p.parse_args()

    if not os.path.isfile(a.path):
        sys.exit(f"File not found: {a.path}")
    if not a.chars or not a.chars.isascii():
        sys.exit("--chars must contain at least one ASCII character")
    boost = a.boost if a.boost is not None else (1.4 if len(a.chars) == 1 else 1.0)
    Player(a.path, a.chars, a.width, a.loop, max(1, a.quant), boost).run()


if __name__ == "__main__":
    main()
