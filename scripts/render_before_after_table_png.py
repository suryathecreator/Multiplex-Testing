#!/usr/bin/env python3
"""Render the before/after budget-match table to PNG without external deps."""

from __future__ import annotations

import csv
import struct
import zlib
from pathlib import Path


OUT_DIR = Path(
    "final_eval_outputs/"
    "aime-train100-ablation4096-r3-stable3b-20260501-103533/"
    "shared2_budget_backfill_graphs"
)
CSV_PATH = OUT_DIR / "before_after_budget_match.csv"
PNG_PATH = OUT_DIR / "before_after_budget_match.png"


FONT = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11110", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "a": ["00000", "00000", "01110", "00001", "01111", "10001", "01111"],
    "c": ["00000", "00000", "01110", "10000", "10000", "10001", "01110"],
    "e": ["00000", "00000", "01110", "10001", "11111", "10000", "01110"],
    "f": ["00110", "01000", "01000", "11100", "01000", "01000", "01000"],
    "h": ["10000", "10000", "10110", "11001", "10001", "10001", "10001"],
    "k": ["10000", "10010", "10100", "11000", "10100", "10010", "10001"],
    "m": ["00000", "00000", "11010", "10101", "10101", "10101", "10101"],
    "o": ["00000", "00000", "01110", "10001", "10001", "10001", "01110"],
    "p": ["00000", "00000", "11110", "10001", "11110", "10000", "10000"],
    "r": ["00000", "00000", "10110", "11001", "10000", "10000", "10000"],
    "s": ["00000", "00000", "01111", "10000", "01110", "00001", "11110"],
    "t": ["01000", "01000", "11100", "01000", "01000", "01001", "00110"],
    "u": ["00000", "00000", "10001", "10001", "10001", "10011", "01101"],
}


def png_write(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = []
    stride = width * 3
    for y in range(height):
        rows.append(b"\x00" + bytes(pixels[y * stride : (y + 1) * stride]))
    data = zlib.compress(b"".join(rows), level=9)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", data)
        + chunk(b"IEND", b"")
    )


def set_px(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        idx = (y * width + x) * 3
        pixels[idx : idx + 3] = bytes(color)


def rect(pixels: bytearray, width: int, height: int, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            set_px(pixels, width, height, xx, yy, color)


def text_width(text: str, scale: int) -> int:
    return sum((len(FONT.get(ch, FONT[" "])[0]) + 1) * scale for ch in text) - scale


def draw_text(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    scale: int,
    color: tuple[int, int, int],
) -> None:
    cursor = x
    for ch in text:
        glyph = FONT.get(ch, FONT[" "])
        for gy, line in enumerate(glyph):
            for gx, bit in enumerate(line):
                if bit == "1":
                    rect(pixels, width, height, cursor + gx * scale, y + gy * scale, scale, scale, color)
        cursor += (len(glyph[0]) + 1) * scale


def draw_centered(
    pixels: bytearray,
    width: int,
    height: int,
    cx: int,
    y: int,
    text: str,
    scale: int,
    color: tuple[int, int, int],
) -> None:
    draw_text(pixels, width, height, cx - text_width(text, scale) // 2, y, text, scale, color)


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open()))
    width, height = 1120, 430
    pixels = bytearray([255, 255, 255] * width * height)
    black = (15, 15, 15)
    gray = (230, 230, 230)
    left, right = 100, 1020
    col_x = [170, 405, 670, 925]
    top = 70
    scale = 5
    small = 4

    draw_centered(pixels, width, height, width // 2, 24, "Shared every 2 before and after top-up", small, black)
    rect(pixels, width, height, left, top, right - left, 4, black)
    draw_centered(pixels, width, height, col_x[0], top + 28, "k", scale, black)
    draw_centered(pixels, width, height, col_x[1], top + 28, "Fixed", scale, black)
    draw_centered(pixels, width, height, col_x[2], top + 28, "Before top-up", scale, black)
    draw_centered(pixels, width, height, col_x[3], top + 28, "After top-up", scale, black)
    rect(pixels, width, height, left, top + 82, right - left, 3, black)

    y = top + 118
    for row in rows:
        draw_centered(pixels, width, height, col_x[0], y, row["pass_k"], scale, black)
        draw_centered(pixels, width, height, col_x[1], y, f"{float(row['fixed']):.4f}", scale, black)
        draw_centered(pixels, width, height, col_x[2], y, f"{float(row['before']):.4f}", scale, black)
        draw_centered(pixels, width, height, col_x[3], y, f"{float(row['after']):.4f}", scale, black)
        rect(pixels, width, height, left, y + 44, right - left, 1, gray)
        y += 58
    rect(pixels, width, height, left, y - 12, right - left, 4, black)
    png_write(PNG_PATH, width, height, pixels)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
