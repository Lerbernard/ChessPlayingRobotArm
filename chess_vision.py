"""
CHESS MOVE DETECTION FROM A WEBCAM

The reliable way to do this is NOT to recognise pieces. It is to watch which
SQUARES changed. A move always empties one square and fills another, so
comparing a frame against the previous position gives you the move directly.
No training data, no model, and it does not care what the pieces look like.

    pip install opencv-python numpy

    python chess_vision.py

HOW TO USE
    1. Point the webcam at the board. The whole board must be in frame.
    2. Click the FOUR CORNERS of the playing area, in this order:
           a1, h1, h8, a8      (that is, going around the edge)
       Click the OUTER corners of the corner squares, not their centres.
    3. Press  b  to capture the baseline (do this with the start position set up).
    4. Make a move. Press  d  to detect it.

    KEYS
       c   re-click the corners from scratch
       r   RESET: forget the saved corners and start calibration again
       n   nudge mode: fine-tune a corner with the arrow keys
       f   flip / rotate the board 90 degrees (if a1 ends up in the wrong place)
       b   capture baseline (call the current position "known")
       d   detect what changed since the baseline
       [ ] shrink / grow the sampled area inside each square
       - + how strict the overhang filter is
       e/E raise / lower the edge weight (helps same-tone pieces)
       ARROWS  drag every sampling box until it sits on the piece BASE
       0   reset that offset back to zero
       ; ' shift sampling towards the BASE of the piece (away from the top)
       k   click where the camera looks straight down (sets the lean direction)
       s   save a snapshot
       q   quit
"""

import sys
import json
import os

import cv2
import numpy as np

CAM_INDEX   = 1            # try 1, 2... if the wrong camera opens

# A tall piece leans over the edge of its square and bleeds into the neighbour.
# Two defences:
#   INSET     - only look at the middle of each square, ignoring the edges
#   RATIO     - a second square only counts if it changed nearly as much as
#               the first. Overhang scores far lower than a real move.
INSET       = 0.30         # fraction of the square trimmed off each side

# The camera is not perfectly overhead, so pieces LEAN AWAY from the point
# directly beneath it. Their tops splay into neighbouring squares while their
# BASES stay in the right square. The warp is only exact at board level, so
# the base is the honest part to look at.
#
# BASE_BIAS pulls each sampling window from the square centre back towards
# the nadir (the spot the camera looks straight down on), which is where the
# base of the piece is relative to its leaning top.
BASE_BIAS   = 0.30         # 0 = square centre, 1 = shifted a whole half-square
NADIR       = None         # (x, y) in warped pixels; None = centre of the board

# A blunt, direct offset applied to EVERY sampling box, in warped pixels.
# Use the arrow keys to drag the boxes until they sit on the piece BASES.
# Much easier to get right by eye than the nadir maths.
OFFSET_X    = 0
OFFSET_Y    = 0
RATIO       = 0.40         # 2nd square must score >= this * the 1st
THRESHOLD   = 12.0         # below this, treat a square as unchanged
WARP_SIZE   = 640          # the board gets flattened to this many pixels square
CALIB_FILE  = "board_corners.json"

FILES = "abcdefgh"
RANKS = "12345678"


# ---------------------------------------------------------------- corners
def load_corners():
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                pts = json.load(f)
            if len(pts) == 4:
                print(f"[Cal] loaded corners from {CALIB_FILE}")
                return np.float32(pts)
        except Exception as e:
            print(f"[Cal] could not read {CALIB_FILE}: {e}")
    return None


def save_corners(pts):
    with open(CALIB_FILE, "w") as f:
        json.dump([[float(x), float(y)] for x, y in pts], f, indent=2)
    print(f"[Cal] saved to {CALIB_FILE}")


def forget_corners():
    """Throw away the saved calibration so the next run starts fresh."""
    if os.path.exists(CALIB_FILE):
        os.remove(CALIB_FILE)
        print(f"[Cal] deleted {CALIB_FILE}")
    else:
        print("[Cal] nothing saved to delete")


def rotate_corners(pts):
    """Spin the corner order by one, rotating the board 90 degrees."""
    return np.float32([pts[1], pts[2], pts[3], pts[0]])


def nudge_corners(cap, corners):
    """
    Fine-tune one corner at a time with the arrow keys. Far easier than
    re-clicking when the grid is only a few pixels out.
        TAB   next corner        arrows  move by 1 px
        SHIFT+arrows via W A S D move by 10 px
        ENTER save,  ESC cancel
    """
    labels = ["a1", "h1", "h8", "a8"]
    pts = corners.copy()
    idx = 0
    print("\n[Nudge] TAB = next corner, arrows = 1px, wasd = 10px,"
          " ENTER = save, ESC = cancel")

    while True:
        ok, frame = cap.read()
        if not ok:
            return None

        view = frame.copy()
        for i, p in enumerate(pts):
            c = (0, 0, 255) if i == idx else (0, 255, 0)
            cv2.circle(view, (int(p[0]), int(p[1])), 8, c, -1)
            cv2.putText(view, labels[i], (int(p[0]) + 12, int(p[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        cv2.polylines(view, [np.int32(pts)], True, (0, 200, 255), 2)
        cv2.putText(view, f"nudging {labels[idx]}   TAB=next  ENTER=save  ESC=cancel",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("camera", view)
        cv2.imshow("board", draw_grid(warp_board(frame, pts)))

        k = cv2.waitKeyEx(20)
        if k == -1:
            continue
        key = k & 0xFF

        if key == 9:                       # TAB
            idx = (idx + 1) % 4
        elif key in (13, 10):              # ENTER
            save_corners(pts)
            return pts
        elif key == 27:                    # ESC
            return None
        # arrow keys: 1 px
        elif k in (2424832, 65361): pts[idx][0] -= 1      # left
        elif k in (2555904, 65363): pts[idx][0] += 1      # right
        elif k in (2490368, 65362): pts[idx][1] -= 1      # up
        elif k in (2621440, 65364): pts[idx][1] += 1      # down
        # wasd: 10 px
        elif key == ord("a"): pts[idx][0] -= 10
        elif key == ord("d"): pts[idx][0] += 10
        elif key == ord("w"): pts[idx][1] -= 10
        elif key == ord("s"): pts[idx][1] += 10


def pick_corners(cap):
    """Click a1, h1, h8, a8 in that order."""
    labels = ["a1 (outer corner)", "h1", "h8", "a8"]
    pts = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            print(f"  {labels[len(pts) - 1]} -> ({x}, {y})")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    print("\n[Cal] Click the four OUTER corners: a1, h1, h8, a8")
    print("      (going around the edge of the playing area)")
    print("      u = undo last,  ENTER = done,  ESC = cancel\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Cam] lost the camera")
            return None
        view = frame.copy()

        for i, p in enumerate(pts):
            cv2.circle(view, p, 7, (0, 255, 0), -1)
            cv2.putText(view, labels[i].split()[0], (p[0] + 10, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(pts) > 1:
            cv2.polylines(view, [np.int32(pts)], len(pts) == 4, (0, 200, 255), 2)

        msg = ("all four set - press ENTER" if len(pts) == 4
               else f"click {labels[len(pts)]}")
        cv2.putText(view, msg, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.imshow("calibrate", view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k in (13, 10) and len(pts) == 4:
            cv2.destroyWindow("calibrate")
            arr = np.float32(pts)
            save_corners(arr)
            return arr
        elif k == 27:
            cv2.destroyWindow("calibrate")
            return None


def warp_board(frame, corners):
    """Flatten the board to a square, top-down, a8 at the top-left."""
    # a1 h1 h8 a8  ->  bottom-left, bottom-right, top-right, top-left
    dst = np.float32([[0, WARP_SIZE], [WARP_SIZE, WARP_SIZE],
                      [WARP_SIZE, 0], [0, 0]])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))


# ---------------------------------------------------------------- squares
def nadir_point(nadir=None):
    """Where the camera looks straight down, in warped pixels."""
    if nadir is not None:
        return nadir
    if NADIR is not None:
        return NADIR
    return (WARP_SIZE / 2.0, WARP_SIZE / 2.0)


def cell_rect(f, r, inset_frac=None, bias=None, nadir=None, off=(0, 0)):
    """
    The box to sample for one square, shifted towards the BASE of any piece
    standing there rather than its leaning top.
    """
    s = WARP_SIZE // 8
    if inset_frac is None:
        inset_frac = INSET
    if bias is None:
        bias = BASE_BIAS
    inset = int(s * inset_frac)

    cx, cy = (f + 0.5) * s, (r + 0.5) * s
    nx, ny = nadir_point(nadir)

    # unit vector from this square towards the nadir; the base lies that way
    dx, dy = nx - cx, ny - cy
    mag = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    shift = bias * (s / 2.0)
    ox, oy = dx / mag * shift, dy / mag * shift

    ox += off[0]; oy += off[1]          # blunt manual nudge on top

    x0 = int(f * s + inset + ox); x1 = int((f + 1) * s - inset + ox)
    y0 = int(r * s + inset + oy); y1 = int((r + 1) * s - inset + oy)

    # keep it on the board
    x0 = max(0, min(WARP_SIZE - 2, x0)); x1 = max(x0 + 2, min(WARP_SIZE, x1))
    y0 = max(0, min(WARP_SIZE - 2, y0)); y1 = max(y0 + 2, min(WARP_SIZE, y1))
    return x0, y0, x1, y1


def square_cells(warped, inset_frac=None, bias=None, nadir=None, off=(0, 0)):
    """Split the warped board into 64 sampling patches, keyed by square name."""
    cells = {}
    for r in range(8):             # r=0 is rank 8 (top of the warped image)
        for f in range(8):
            x0, y0, x1, y1 = cell_rect(f, r, inset_frac, bias, nadir, off)
            cells[FILES[f] + RANKS[7 - r]] = warped[y0:y1, x0:x1]
    return cells


SIG_SIZE = (32, 32)        # every cell is resized to this before comparing

def cell_signature(cell):
    """
    A fingerprint of one square that survives the hard case:
    a WHITE piece on a WHITE square, or a BLACK piece on a BLACK square.

    Plain brightness fails there - a white pawn on a light square barely
    shifts the average. So the signature stacks three views, each of which
    catches something the others miss:

      1. LOCALLY NORMALISED BRIGHTNESS
         CLAHE stretches the contrast inside each square on its own, so the
         faint shading on a white-on-white piece gets amplified into
         something measurable instead of being lost in the average.

      2. EDGE ENERGY (Sobel gradient magnitude)
         This is the important one. A piece has a SILHOUETTE and curved,
         shaded sides. Those edges exist no matter how well the piece's tone
         matches the square underneath it. An empty square is flat and has
         almost no gradient; an occupied one has a lot.

      3. COLOUR (a and b from Lab)
         Real "white" and "black" pieces are usually slightly warm or cool
         compared to the board. Lab keeps that separate from lightness, so it
         still helps when brightness alone does not.
    """
    c = cv2.resize(cell, SIG_SIZE, interpolation=cv2.INTER_AREA)
    c = cv2.GaussianBlur(c, (3, 3), 0)

    lab = cv2.cvtColor(c, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)

    # 1. local contrast: pulls detail out of same-tone-on-same-tone
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    Ln = clahe.apply(L)

    # 2. edges: present whenever a piece is there, tone-independent
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    edge = np.clip(edge * EDGE_GAIN, 0, 255)

    return np.dstack([
        Ln.astype(np.float32),
        edge.astype(np.float32),
        A.astype(np.float32),
        B.astype(np.float32),
    ])


# How much each view counts towards the final score.
# Raise EDGE if same-tone pieces are still being missed.
# Tuned by grid search so all four tone combinations score about the SAME.
# Before: white-on-white scored 6, black-on-white scored 63 - a 10x spread,
# which meant the ratio filter threw away real white-on-white moves.
# After: 29 / 29 / 53 / 29 - a 1.8x spread, all comfortably detected.
W_BRIGHT, W_EDGE, W_COLOUR = 0.2, 2.0, 1.0
EDGE_GAIN = 2.5


def cell_score(a, b):
    """How different are two signatures? Weighted across the three views."""
    d = np.abs(a - b)
    bright = float(d[:, :, 0].mean())
    edge   = float(d[:, :, 1].mean())
    colour = float(d[:, :, 2].mean() + d[:, :, 3].mean()) / 2.0
    return (W_BRIGHT * bright + W_EDGE * edge + W_COLOUR * colour) / \
           (W_BRIGHT + W_EDGE + W_COLOUR)


def diff_squares(base, now, threshold=None, ratio=None):
    """
    Which squares changed between two boards?

    Returns (kept, rejected). A square is only kept if it changed by at least
    `ratio` of the biggest change - that is what throws out the faint bleed
    from a tall piece leaning into the square next door, which typically
    scores 6x lower than the square that actually changed.
    """
    if ratio is None:
        ratio = RATIO
    if threshold is None:
        threshold = THRESHOLD

    scored = []
    for name in base:
        a, b = base[name], now[name]
        if a is None or b is None or a.shape != b.shape:
            continue
        score = cell_score(a, b)
        if score > threshold:
            scored.append((name, score))
    scored.sort(key=lambda t: -t[1])

    if not scored:
        return [], []

    cutoff = scored[0][1] * ratio
    kept     = [t for t in scored if t[1] >= cutoff]
    rejected = [t for t in scored if t[1] <  cutoff]
    return kept, rejected


def guess_move(changed):
    """
    Two squares changed = a move. We cannot tell direction from the diff
    alone, so we report both and let the game state decide which is the
    origin (the one that HAD a piece before).
    """
    if len(changed) < 2:
        return None
    return changed[0][0], changed[1][0]


# ---------------------------------------------------------------- overlay
def draw_grid(warped, highlight=(), inset_frac=None, bias=None, nadir=None, off=(0, 0)):
    out = warped.copy()
    s = WARP_SIZE // 8
    if inset_frac is None:
        inset_frac = INSET
    ins = int(s * inset_frac)

    for i in range(9):
        cv2.line(out, (i * s, 0), (i * s, WARP_SIZE), (0, 180, 0), 1)
        cv2.line(out, (0, i * s), (WARP_SIZE, i * s), (0, 180, 0), 1)

    # the actual sampled area - shifted towards each piece's BASE
    for r in range(8):
        for f in range(8):
            x0, y0, x1, y1 = cell_rect(f, r, inset_frac, bias, nadir, off)
            cv2.rectangle(out, (x0, y0), (x1, y1), (90, 90, 200), 1)

    # mark the nadir: pieces lean directly away from this point
    nx, ny = nadir_point(nadir)
    cv2.drawMarker(out, (int(nx), int(ny)), (0, 220, 255),
                   cv2.MARKER_CROSS, 22, 2)
    cv2.circle(out, (int(nx), int(ny)), 13, (0, 220, 255), 1)

    for r in range(8):
        for f in range(8):
            name = FILES[f] + RANKS[7 - r]
            cv2.putText(out, name, (f * s + 4, r * s + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)

    for name in highlight:
        f = FILES.index(name[0])
        r = 7 - RANKS.index(name[1])
        cv2.rectangle(out, (f * s + 2, r * s + 2),
                      ((f + 1) * s - 2, (r + 1) * s - 2), (0, 0, 255), 3)
    return out


def pick_nadir(cap, corners):
    """
    Click the square the camera is looking straight down on - the one whose
    piece looks like it is standing UPRIGHT rather than leaning. Pieces lean
    directly away from that point, so it is what tells us where each base is.
    """
    picked = {}

    def on_click(e, x, y, flags, param):
        if e == cv2.EVENT_LBUTTONDOWN:
            picked["xy"] = (float(x), float(y))

    cv2.namedWindow("nadir")
    cv2.setMouseCallback("nadir", on_click)
    print("\n[Nadir] Click the spot the camera looks STRAIGHT DOWN on -")
    print("        the place where a piece looks upright, not leaning.")
    print("        Pieces lean away from it. ESC to cancel.")

    while "xy" not in picked:
        ok, frame = cap.read()
        if not ok:
            return None
        w = warp_board(frame, corners)
        cv2.putText(w, "click where pieces look UPRIGHT (not leaning)",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2)
        cv2.imshow("nadir", w)
        if (cv2.waitKey(1) & 0xFF) == 27:
            cv2.destroyWindow("nadir")
            return None

    cv2.destroyWindow("nadir")
    print(f"        nadir set to {picked['xy']}")
    return picked["xy"]


# ---------------------------------------------------------------- main
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)   # DSHOW: faster on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[Cam] could not open camera {CAM_INDEX}.")
        print("      Try changing CAM_INDEX to 1 or 2 at the top of this file.")
        sys.exit(1)

    corners = load_corners()
    if corners is None:
        corners = pick_corners(cap)
        if corners is None:
            cap.release()
            return

    baseline = None
    highlight = ()
    inset  = INSET
    ratio  = RATIO
    bias   = BASE_BIAS
    nadir  = NADIR
    off    = [OFFSET_X, OFFSET_Y]
    status = "press b to set the baseline"

    print("\n  b = baseline    d = detect")
    print("  c = re-click    r = full reset    n = nudge a corner    f = rotate 90")
    print("  ARROWS = drag the sample boxes onto the piece bases,  0 = reset")
    print("  ;/' base bias   k = set nadir   [ ] inset   - + ratio")
    print("  s = snapshot    q = quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        warped = warp_board(frame, corners)
        view = draw_grid(warped, highlight, inset, bias, nadir, tuple(off))

        cv2.rectangle(view, (0, WARP_SIZE - 52), (WARP_SIZE, WARP_SIZE),
                      (0, 0, 0), -1)
        cv2.putText(view, status, (8, WARP_SIZE - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(view, f"ARROWS move boxes ({off[0]:+d},{off[1]:+d}) 0=reset  "
                          f"inset {inset:.2f}[]  ratio {ratio:.2f}-+  |  b base  d detect",
                    (8, WARP_SIZE - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

        # show the raw camera with the board outline, so you can see framing
        raw = frame.copy()
        cv2.polylines(raw, [np.int32(corners)], True, (0, 200, 255), 2)
        cv2.imshow("camera", raw)
        cv2.imshow("board", view)

        kx = cv2.waitKeyEx(1)
        k = kx & 0xFF if kx != -1 else 255

        # arrow keys drag the WHOLE sampling grid onto the piece bases
        if kx in (2424832, 65361):   off[0] -= 1; baseline = None
        elif kx in (2555904, 65363): off[0] += 1; baseline = None
        elif kx in (2490368, 65362): off[1] -= 1; baseline = None
        elif kx in (2621440, 65364): off[1] += 1; baseline = None
        if kx in (2424832, 65361, 2555904, 65363, 2490368, 65362, 2621440, 65364):
            status = f"sample offset ({off[0]:+d},{off[1]:+d}) - press b to re-baseline"

        if k == ord("q"):
            break

        elif k == ord("b"):
            baseline = {n: cell_signature(c)
                        for n, c in square_cells(warped, inset, bias, nadir, tuple(off)).items()}
            highlight = ()
            status = "baseline set - make a move, then press d"
            print("[Base] captured")

        elif k == ord("d"):
            if baseline is None:
                status = "no baseline yet - press b first"
                continue
            now = {n: cell_signature(c)
                   for n, c in square_cells(warped, inset, bias, nadir,
                                            tuple(off)).items()}
            kept, rejected = diff_squares(baseline, now, ratio=ratio)

            if not kept:
                status = "nothing changed"
                highlight = ()
            else:
                print("\n[Diff] real changes:")
                for n, sc in kept[:4]:
                    print(f"    {n}   score {sc:6.1f}")
                if rejected:
                    print("       ignored as overhang/shadow:")
                    for n, sc in rejected[:4]:
                        print(f"         {n}   score {sc:6.1f}")

                mv = guess_move(kept)
                if mv:
                    highlight = (mv[0], mv[1])
                    status = f"move between {mv[0]} and {mv[1]}"
                    print(f"[Move] {mv[0]} <-> {mv[1]}")
                elif len(kept) == 1:
                    highlight = (kept[0][0],)
                    status = f"only {kept[0][0]} changed (capture?)"
                else:
                    highlight = tuple(n for n, _ in kept[:4])
                    status = f"{len(kept)} squares changed - ambiguous"

                baseline = now

        elif k == ord("c"):
            new = pick_corners(cap)
            if new is not None:
                corners = new
                baseline = None
                highlight = ()
                status = "recalibrated - press b for a new baseline"

        elif k == ord("r"):
            # full reset: bin the saved file and click again from scratch
            forget_corners()
            new = pick_corners(cap)
            if new is not None:
                corners = new
                baseline = None
                highlight = ()
                status = "reset - press b for a new baseline"
            else:
                status = "reset cancelled - old corners still in use"

        elif k == ord("n"):
            new = nudge_corners(cap, corners)
            if new is not None:
                corners = new
                baseline = None
                highlight = ()
                status = "nudged - press b for a new baseline"
            else:
                status = "nudge cancelled"

        elif k == ord("f"):
            corners = rotate_corners(corners)
            save_corners(corners)
            baseline = None
            highlight = ()
            status = "rotated 90 - press f again if still wrong"
            print("[Cal] rotated the board 90 degrees")

        elif k == ord("["):
            inset = max(0.05, inset - 0.03)
            baseline = None
            status = f"inset {inset:.2f} - press b to re-baseline"
        elif k == ord("]"):
            inset = min(0.45, inset + 0.03)
            baseline = None
            status = f"inset {inset:.2f} (tighter = less overhang) - press b"

        elif k == ord(";"):
            bias = max(0.0, bias - 0.05)
            baseline = None
            status = f"base bias {bias:.2f} (0 = square centre) - press b"
        elif k == ord("'"):
            bias = min(0.9, bias + 0.05)
            baseline = None
            status = f"base bias {bias:.2f} (more = further towards the base) - press b"

        elif k == ord("k"):
            nn = pick_nadir(cap, corners)
            if nn is not None:
                nadir = nn
                baseline = None
                status = "nadir set - press b to re-baseline"

        elif k == ord("e"):
            ns = globals()
            ns["W_EDGE"] = min(6.0, W_EDGE + 0.5)
            baseline = None
            status = f"edge weight {W_EDGE:.1f} (higher = better same-tone) - press b"
        elif k == ord("E"):
            ns = globals()
            ns["W_EDGE"] = max(0.5, W_EDGE - 0.5)
            baseline = None
            status = f"edge weight {W_EDGE:.1f} - press b"

        elif k == ord("-"):
            ratio = max(0.05, ratio - 0.05)
            status = f"ratio {ratio:.2f} (lower = more squares accepted)"
        elif k in (ord("="), ord("+")):
            ratio = min(0.95, ratio + 0.05)
            status = f"ratio {ratio:.2f} (higher = rejects overhang harder)"

        elif k == ord("0"):
            off = [0, 0]
            baseline = None
            status = "sample offset reset to (0,0)"

        elif k == ord("s"):
            cv2.imwrite("snapshot_board.png", view)
            cv2.imwrite("snapshot_raw.png", frame)
            print("[Save] snapshot_board.png, snapshot_raw.png")
            status = "saved"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()