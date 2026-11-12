#!/usr/bin/env python3
"""Lit une vidéo en couleur directement dans le terminal, en ASCII.

Usage:
    python video_ascii.py chemin/vers/video.mp4
    python video_ascii.py video.mp4 --chars "+"      # uniquement des "+"
    python video_ascii.py video.mp4 --width 160 --loop

Dépendances : numpy, opencv-python (pour décoder la vidéo).
Le terminal doit supporter la couleur 24 bits (Windows Terminal, iTerm2,
la plupart des terminaux Linux récents).
"""

import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np

# Rampe du plus sombre au plus clair (même esprit que la capture d'écran).
DEFAULT_RAMP = " .:-=+*#%@"

# Proportion hauteur/largeur d'une cellule de terminal (~2:1).
CELL_ASPECT = 0.5

# --- Encodage vectorisé d'une image en séquences ANSI --------------------
# Chaque cellule = 20 octets de largeur fixe :
#   ESC[38;2;RRR;GGG;BBBm<char>
# Les chiffres sont complétés par des zéros (les terminaux l'acceptent),
# ce qui permet de tout construire avec numpy sans boucle Python.
_PREFIX = np.frombuffer(b"\x1b[38;2;", dtype=np.uint8)
_DIGITS = np.array([list(f"{i:03d}".encode()) for i in range(256)], dtype=np.uint8)
CELL_BYTES = 20


def encode_frame(rgb: np.ndarray, chars: np.ndarray) -> bytes:
    """rgb : (h, w, 3) uint8, chars : (h, w) uint8 (codes ASCII)."""
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

    # On n'émet le code couleur que s'il change par rapport à la cellule
    # précédente de la même ligne -> beaucoup moins d'octets à écrire.
    changed = np.ones((h, w), dtype=bool)
    changed[:, 1:] = np.any(rgb[:, 1:] != rgb[:, :-1], axis=2)
    # Un espace n'a pas besoin de couleur ; la cellule qui le suit doit
    # donc réémettre la sienne.
    space = chars == ord(" ")
    changed[:, 1:] |= space[:, :-1]
    changed &= ~space

    mask = np.empty((h, w, CELL_BYTES), dtype=bool)
    mask[..., :19] = changed[..., None]
    mask[..., 19] = True

    # Ajout d'un '\n' en fin de ligne (sauf la dernière).
    flat = np.empty((h, w * CELL_BYTES + 1), dtype=np.uint8)
    flat[:, :-1] = cells.reshape(h, -1)
    flat[:, -1] = ord("\n")
    fmask = np.empty((h, w * CELL_BYTES + 1), dtype=bool)
    fmask[:, :-1] = mask.reshape(h, -1)
    fmask[:, -1] = True
    fmask[-1, -1] = False

    return flat[fmask].tobytes()


# --- Traitement d'image ---------------------------------------------------
def frame_to_cells(frame_bgr, cols, rows, ramp, quant, boost):
    small = cv2.resize(frame_bgr, (cols, rows), interpolation=cv2.INTER_AREA)
    rgb = small[..., ::-1]  # BGR -> RGB

    # Luminance perçue -> index dans la rampe de caractères.
    lum = rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    idx = (lum * (len(ramp) - 1) / 255.0 + 0.5).astype(np.intp)
    chars = ramp[idx]

    color = rgb.astype(np.float32)
    if boost != 1.0:
        # Les glyphes fins couvrent peu de pixels : on éclaircit un peu.
        color = color * boost
    color = np.clip(color, 0, 255).astype(np.uint8)
    if quant > 1:
        # Quantification : plus de cellules voisines identiques -> sortie plus légère.
        color = (color // quant) * quant + quant // 2
        color = np.minimum(color, 255).astype(np.uint8)
    return color, chars


def fit_size(vw, vh, term_cols, term_rows, width=None):
    max_rows = max(1, term_rows - 1)  # une ligne pour la barre d'état
    cols = min(width or term_cols, term_cols)
    rows = int(round(cols * vh / vw * CELL_ASPECT))
    if rows > max_rows:
        rows = max_rows
        cols = int(round(rows * vw / vh / CELL_ASPECT))
    return max(1, cols), max(1, rows)


# --- Boucle principale ----------------------------------------------------
def play(path, ramp_str, width, loop, quant, boost):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir la vidéo : {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1 <= fps <= 240):
        fps = 30.0
    frame_time = 1.0 / fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ramp = np.frombuffer(ramp_str.encode("ascii"), dtype=np.uint8)
    name = os.path.basename(path)

    out = sys.stdout.buffer
    if os.name == "nt":
        os.system("")  # active le traitement des séquences ANSI sous Windows

    out.write(b"\x1b[?1049h\x1b[?25l\x1b[2J")  # écran alternatif, curseur caché
    out.flush()

    last_size = None
    start = time.perf_counter()
    frame_no = 0
    shown = 0
    fps_t0, fps_n, real_fps = time.perf_counter(), 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if loop and frame_no > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start, frame_no = time.perf_counter(), 0
                    continue
                break
            frame_no += 1

            # En retard ? On saute des images pour rester synchro.
            now = time.perf_counter()
            target = start + frame_no * frame_time
            if now > target + frame_time:
                continue

            term = shutil.get_terminal_size((120, 40))
            cols, rows = fit_size(vw, vh, term.columns, term.lines, width)
            if (cols, rows, term.columns, term.lines) != last_size:
                out.write(b"\x1b[0m\x1b[2J")
                last_size = (cols, rows, term.columns, term.lines)

            color, chars = frame_to_cells(frame, cols, rows, ramp, quant, boost)
            pad = (term.columns - cols) // 2
            body = encode_frame(color, chars)
            if pad:
                body = body.replace(b"\n", b"\n\x1b[%dG" % (pad + 1))

            fps_n += 1
            if now - fps_t0 >= 1.0:
                real_fps, fps_t0, fps_n = fps_n / (now - fps_t0), now, 0
            pos = f"{frame_no}/{total}" if total else str(frame_no)
            status = f" {name}  ·  {cols}x{rows}  ·  image {pos}  ·  {real_fps:4.1f}/{fps:.0f} fps  ·  Ctrl+C pour quitter"
            status = status[: term.columns - 1]

            out.write(
                b"\x1b[H" + (b"\x1b[%dG" % (pad + 1) if pad else b"") + body
                + b"\x1b[0m\x1b[%d;1H\x1b[2K\x1b[90m" % term.lines
                + status.encode("utf-8", "replace") + b"\x1b[0m"
            )
            out.flush()
            shown += 1

            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        out.write(b"\x1b[0m\x1b[?25h\x1b[?1049l")
        out.flush()


def main():
    p = argparse.ArgumentParser(description="Lecteur vidéo ASCII en couleur pour le terminal.")
    p.add_argument("path", help="chemin de la vidéo")
    p.add_argument("--chars", default=DEFAULT_RAMP,
                   help='caractères du plus sombre au plus clair (défaut : "%(default)s"). '
                        'Utiliser --chars "+" pour uniquement des +.')
    p.add_argument("--width", type=int, default=None, help="largeur max en colonnes")
    p.add_argument("--loop", action="store_true", help="rejouer en boucle")
    p.add_argument("--quant", type=int, default=4,
                   help="pas de quantification des couleurs (1 = aucune, défaut 4)")
    p.add_argument("--boost", type=float, default=None,
                   help="multiplicateur de luminosité des couleurs (défaut : 1.0, ou 1.4 avec un seul caractère)")
    a = p.parse_args()

    if not os.path.isfile(a.path):
        sys.exit(f"Fichier introuvable : {a.path}")
    if not a.chars or not a.chars.isascii():
        sys.exit("--chars doit contenir au moins un caractère ASCII")
    boost = a.boost if a.boost is not None else (1.4 if len(a.chars) == 1 else 1.0)
    play(a.path, a.chars, a.width, a.loop, max(1, a.quant), boost)


if __name__ == "__main__":
    main()
