"""
NutriCare TUI -- Gemini-Inspired Gradient Logo & Wordmark
Author: Senior Python TUI Engineer & Terminal UI/UX Designer
Date: 2026-10-12

Features:
  - Original Bold Pixel-Art / Block Wordmark for 'NUTRICARE' (73 chars wide)
  - Horizontal Blue -> Indigo -> Violet -> Purple -> Pink 24-bit gradient
  - Subtitles: 'AI-POWERED BIOMARKER ANALYSIS' & 'LOCAL ML DEMONSTRATION SYSTEM'
  - Compact persistent header brand for active screens
"""

import sys
from rich.text import Text

# Color stops for the Gemini-inspired horizontal gradient:
# Blue (#5294E2) -> Indigo (#7585E8) -> Violet (#A879D5) -> Purple (#BD71B8) -> Pink (#D46A9B)
GRADIENT_STOPS = [
    (0.00, (82, 148, 226)),    # #5294E2 Blue
    (0.25, (117, 133, 232)),   # #7585E8 Indigo
    (0.50, (168, 121, 213)),   # #A879D5 Violet
    (0.75, (189, 113, 184)),   # #BD71B8 Purple
    (1.00, (212, 106, 155)),   # #D46A9B Pink
]

# Bold solid pixel-art block wordmark (73 chars wide, 6 lines high)
# Uses clean solid block Unicode characters (█ and space) for maximum clarity and cross-platform fidelity
WORDMARK_LINES = [
    r"███   ██ ██    ██ ████████ ██████  ████  ██████   ██████  ██████  ███████",
    r"████  ██ ██    ██    ██    ██   ██  ██  ██    ██ ██    ██ ██   ██ ██     ",
    r"██ ██ ██ ██    ██    ██    ██████   ██  ██       ████████ ██████  █████  ",
    r"██  ████ ██    ██    ██    ██   ██  ██  ██       ██    ██ ██   ██ ██     ",
    r"██   ███ ██    ██    ██    ██   ██  ██  ██    ██ ██    ██ ██   ██ ██     ",
    r"██    ██  ██████     ██    ██   ██ ████  ██████  ██    ██ ██   ██ ███████",
]

# Compact wordmark for constrained terminal views (35 chars wide, 3 lines high)
COMPACT_WORDMARK_LINES = [
    r"█▄  █ █   █ ▀█▀ █▀▄  █ █▀▀ ▄▀█ █▀▄  █▀▀",
    r"█ ▀▄█ █   █  █  █▀▀▄ █ █   █▀█ █▀▀▄ ██▄",
    r"▀   ▀  ▀▀▀   ▀  ▀  ▀ ▀ ▀▀▀ ▀ ▀ ▀  ▀ ▀▀▀",
]


def interpolate_rgb(t: float) -> tuple[int, int, int]:
    """Interpolate RGB color along the 5 gradient stops for parameter t in [0.0, 1.0]."""
    t = max(0.0, min(1.0, t))
    for i in range(len(GRADIENT_STOPS) - 1):
        t0, c0 = GRADIENT_STOPS[i]
        t1, c1 = GRADIENT_STOPS[i + 1]
        if t0 <= t <= t1:
            segment_t = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r = int(c0[0] + (c1[0] - c0[0]) * segment_t)
            g = int(c0[1] + (c1[1] - c0[1]) * segment_t)
            b = int(c0[2] + (c1[2] - c0[2]) * segment_t)
            return r, g, b
    return GRADIENT_STOPS[-1][1]


def render_gradient_text(lines: list[str]) -> Text:
    """Renders multi-line text with a continuous horizontal true-color gradient."""
    max_len = max(len(line) for line in lines) if lines else 1
    result = Text()

    for row_idx, line in enumerate(lines):
        for col_idx, ch in enumerate(line):
            if ch == " ":
                result.append(" ")
            else:
                t = col_idx / (max_len - 1) if max_len > 1 else 0.0
                r, g, b = interpolate_rgb(t)
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                result.append(ch, style=f"bold {hex_color}")
        if row_idx < len(lines) - 1:
            result.append("\n")

    return result


def get_logo_text(compact: bool = False) -> Text:
    """Builds the full decorative NutriCare brand banner with decorations and subtitles."""
    lines = COMPACT_WORDMARK_LINES if compact else WORDMARK_LINES
    logo = render_gradient_text(lines)

    result = Text()
    if compact:
        result.append("  :: ───────────────────────────── ::\n", style="#43D9D0")
        result.append_text(logo)
        result.append("\n")
        result.append("  :: ───────────────────────────── ::\n", style="#7585E8")
        result.append("   AI-POWERED BIOMARKER ANALYSIS\n", style="bold #43D9D0")
        result.append("   LOCAL ML DEMONSTRATION SYSTEM", style="bold #9299AA")
    else:
        # Centered geometric border for 73-char wordmark
        result.append("  :: ─────────────────────────────────────────────────────────────────── ::\n", style="#43D9D0")
        result.append_text(logo)
        result.append("\n")
        result.append("  :: ─────────────────────────────────────────────────────────────────── ::\n", style="#7585E8")
        result.append("                      AI-POWERED BIOMARKER ANALYSIS\n", style="bold #43D9D0")
        result.append("                      LOCAL ML DEMONSTRATION SYSTEM", style="bold #9299AA")

    return result


def get_compact_header_brand() -> Text:
    """Returns a sleek single-line header wordmark for the dashboard bar."""
    text = Text()
    text.append("✦ ", style="#43D9D0")
    word = "NUTRICARE"
    for i, ch in enumerate(word):
        t = i / (len(word) - 1)
        r, g, b = interpolate_rgb(t)
        text.append(ch, style=f"bold #{r:02x}{g:02x}{b:02x}")
    text.append(" ML", style="bold #E6E8F0")
    text.append(" v2.1", style="#9299AA")
    return text


if __name__ == "__main__":
    from rich.console import Console
    console = Console()
    console.print(get_logo_text())
    console.print()
    console.print(get_compact_header_brand())
