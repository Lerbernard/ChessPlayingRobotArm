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
       s   save a snapshot
       q   quit
"""

import sys
import json
import os

import cv2
import numpy as np

CAM_INDEX   = 1            # try 1, 2... if the wrong camera opens
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
def square_cells(warped):
    """Split the warped board into 64 cells, keyed by square name."""
    s = WARP_SIZE // 8
    inset = int(s * 0.18)          # ignore the edges: piece bases overhang
    cells = {}
    for r in range(8):             # r=0 is rank 8 (top of the warped image)
        for f in range(8):
            y0, x0 = r * s + inset, f * s + inset
            y1, x1 = (r + 1) * s - inset, (f + 1) * s - inset
            name = FILES[f] + RANKS[7 - r]
            cells[name] = warped[y0:y1, x0:x1]
    return cells


def cell_signature(cell):
    """A small, lighting-tolerant fingerprint of one square."""
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return g.astype(np.float32)


def diff_squares(base, now, threshold=12.0):
    """
    Which squares changed between two boards?
    Returns [(square, score), ...] sorted by how much it changed.
    """
    changed = []
    for name in base:
        a, b = base[name], now[name]
        if a.shape != b.shape:
            continue
        score = float(np.mean(np.abs(a - b)))
        if score > threshold:
            changed.append((name, score))
    changed.sort(key=lambda t: -t[1])
    return changed


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
def draw_grid(warped, highlight=()):
    out = warped.copy()
    s = WARP_SIZE // 8
    for i in range(9):
        cv2.line(out, (i * s, 0), (i * s, WARP_SIZE), (0, 180, 0), 1)
        cv2.line(out, (0, i * s), (WARP_SIZE, i * s), (0, 180, 0), 1)

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
    status = "press b to set the baseline"

    print("\n  b = baseline    d = detect")
    print("  c = re-click    r = full reset    n = nudge a corner    f = rotate 90")
    print("  s = snapshot    q = quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        warped = warp_board(frame, corners)
        view = draw_grid(warped, highlight)

        cv2.rectangle(view, (0, WARP_SIZE - 52), (WARP_SIZE, WARP_SIZE),
                      (0, 0, 0), -1)
        cv2.putText(view, status, (8, WARP_SIZE - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(view, "b base  d detect  |  r reset  n nudge  f rotate  c reclick",
                    (8, WARP_SIZE - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        # show the raw camera with the board outline, so you can see framing
        raw = frame.copy()
        cv2.polylines(raw, [np.int32(corners)], True, (0, 200, 255), 2)
        cv2.imshow("camera", raw)
        cv2.imshow("board", view)

        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break

        elif k == ord("b"):
            baseline = {n: cell_signature(c)
                        for n, c in square_cells(warped).items()}
            highlight = ()
            status = "baseline set - make a move, then press d"
            print("[Base] captured")

        elif k == ord("d"):
            if baseline is None:
                status = "no baseline yet - press b first"
                continue
            now = {n: cell_signature(c)
                   for n, c in square_cells(warped).items()}
            changed = diff_squares(baseline, now)

            if not changed:
                status = "nothing changed"
                highlight = ()
            else:
                top = changed[:4]
                print("\n[Diff] squares that changed:")
                for n, sc in top:
                    print(f"    {n}   score {sc:6.1f}")

                mv = guess_move(changed)
                if mv:
                    highlight = (mv[0], mv[1])
                    status = f"move between {mv[0]} and {mv[1]}"
                    print(f"[Move] {mv[0]} <-> {mv[1]}")
                else:
                    highlight = tuple(n for n, _ in top)
                    status = f"only {len(changed)} square changed"

                # re-baseline so the next detect compares against now
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

        elif k == ord("s"):
            cv2.imwrite("snapshot_board.png", view)
            cv2.imwrite("snapshot_raw.png", frame)
            print("[Save] snapshot_board.png, snapshot_raw.png")
            status = "saved"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
