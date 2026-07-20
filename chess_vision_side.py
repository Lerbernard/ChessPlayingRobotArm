"""
CHESS MOVE DETECTION  -  SIDE / ANGLED CAMERA

A top-down camera can look straight at the flat square. A side camera cannot:
the piece stands UP, so it appears ABOVE its square in the image, and it hides
part of whatever is behind it.

So this version does not sample the flat square. It samples the COLUMN OF SPACE
above each square, where the piece body actually shows up. You calibrate that
by clicking the board corners twice: once at board level, once at the top of a
piece standing on those corners.

    pip install opencv-python numpy
    python chess_vision_side.py

CALIBRATION (two passes, 8 clicks total)
    Pass 1  - the four BOARD corners:   a1, h1, h8, a8
    Pass 2  - put a piece (a KING, the tallest) on each of those four corners
              in turn and click the TOP of it, same order.
    That second pass tells the code how tall a piece looks at each depth,
    which is what makes the sampling work.

KEYS
    b   baseline      d   detect
    r   full reset    n   nudge a corner    f   rotate 90
    h   adjust piece height globally (+/-)
    v   toggle: show the sampling boxes
    s   snapshot      q   quit

HONEST LIMITS OF A SIDE VIEW
    - The rank furthest from the camera is partly hidden by the ranks in front.
      Detection there is less reliable. A HIGH, steep angle helps a lot: the
      closer to overhead, the less occlusion.
    - Shoot along the ranks (from behind White or Black), not from a corner.
      Corner angles occlude diagonally and are much worse.
"""

import os
import sys
import json

import cv2
import numpy as np

CAM_INDEX  = 1
WARP_W     = 640          # warped board width
WARP_H     = 640
CALIB_FILE = "board_corners_side.json"

FILES = "abcdefgh"
RANKS = "12345678"


# ================================================================ calibration
def load_calib():
    if os.path.exists(CALIB_FILE):
        try:
            d = json.load(open(CALIB_FILE))
            base = np.float32(d["base"])
            top  = np.float32(d["top"])
            if len(base) == 4 and len(top) == 4:
                print(f"[Cal] loaded {CALIB_FILE}")
                return base, top
        except Exception as e:
            print(f"[Cal] could not read {CALIB_FILE}: {e}")
    return None, None


def save_calib(base, top):
    json.dump({"base": [[float(a), float(b)] for a, b in base],
               "top":  [[float(a), float(b)] for a, b in top]},
              open(CALIB_FILE, "w"), indent=2)
    print(f"[Cal] saved {CALIB_FILE}")


def forget_calib():
    if os.path.exists(CALIB_FILE):
        os.remove(CALIB_FILE)
        print(f"[Cal] deleted {CALIB_FILE}")
    else:
        print("[Cal] nothing saved to delete")


def click_four(cap, labels, title, hint):
    pts = []

    def on_click(e, x, y, flags, param):
        if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            print(f"    {labels[len(pts) - 1]} -> ({x}, {y})")

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_click)
    print(f"\n[Cal] {hint}")
    print("      u = undo,  ENTER = done,  ESC = cancel")

    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        view = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 7, (0, 255, 0), -1)
            cv2.putText(view, labels[i], (p[0] + 10, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(pts) > 1:
            cv2.polylines(view, [np.int32(pts)], len(pts) == 4, (0, 200, 255), 2)

        msg = "all four set - ENTER" if len(pts) == 4 else f"click {labels[len(pts)]}"
        cv2.rectangle(view, (0, 0), (view.shape[1], 66), (0, 0, 0), -1)
        cv2.putText(view, hint, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 1)
        cv2.putText(view, msg, (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.imshow(title, view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k in (13, 10) and len(pts) == 4:
            cv2.destroyWindow(title)
            return np.float32(pts)
        elif k == 27:
            cv2.destroyWindow(title)
            return None


def calibrate(cap):
    labels = ["a1", "h1", "h8", "a8"]

    base = click_four(cap, labels, "calibrate: board",
                      "PASS 1/2  click the four BOARD corners (a1, h1, h8, a8)")
    if base is None:
        return None, None

    top = click_four(cap, labels, "calibrate: piece height",
                     "PASS 2/2  stand a KING on each corner, click its TOP")
    if top is None:
        return None, None

    save_calib(base, top)
    return base, top


# ================================================================ geometry
def board_to_image(base, f, r):
    """
    Where does the point (f, r) on the board plane land in the image?
    f, r are 0..8 in board units (0 = a-file / rank 1 edge, 8 = far edge).
    Uses the homography from the four clicked corners.
    """
    src = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])      # a1 h1 h8 a8
    M = cv2.getPerspectiveTransform(src, base)
    p = np.float32([[[f, r]]])
    return cv2.perspectiveTransform(p, M)[0][0]


def piece_lift(base, top, f, r):
    """
    How many pixels UP does a piece appear at this board position?
    Interpolated from the four corner measurements, so it shrinks correctly
    with distance from the camera.
    """
    lifts = [top[i][1] - base[i][1] for i in range(4)]        # negative = up
    u, v = f / 8.0, r / 8.0
    return ((1 - u) * (1 - v) * lifts[0] +
            (    u) * (1 - v) * lifts[1] +
            (    u) * (    v) * lifts[2] +
            (1 - u) * (    v) * lifts[3])


def square_boxes(base, top, height_scale=1.0):
    """
    For every square, the image-space box where a piece standing there
    would appear. This is what we sample instead of the flat square.
    """
    boxes = {}
    for r in range(8):
        for f in range(8):
            name = FILES[f] + RANKS[r]

            # centre of the square on the board plane
            cx, cy = board_to_image(base, f + 0.5, r + 0.5)
            # and its two horizontal edges, to size the box
            lx, _ = board_to_image(base, f + 0.15, r + 0.5)
            rx, _ = board_to_image(base, f + 0.85, r + 0.5)
            w = max(6.0, abs(rx - lx))

            lift = piece_lift(base, top, f + 0.5, r + 0.5) * height_scale

            # from just below the base up to the top of a piece
            y0 = cy + abs(lift) * 0.10
            y1 = cy + lift
            top_y, bot_y = sorted((y0, y1))

            boxes[name] = (int(cx - w / 2), int(top_y),
                           int(cx + w / 2), int(bot_y))
    return boxes


def crop(frame, box):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h, y1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return frame[y0:y1, x0:x1]


# ================================================================ detection
SAMPLE = (24, 40)          # every square's patch is resized to this

def signature(frame, box):
    c = crop(frame, box)
    if c is None:
        return None
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, SAMPLE, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    # normalise: shrugs off overall brightness changes, keeps the shape
    g = cv2.equalizeHist(g)
    return g.astype(np.float32)


def snapshot(frame, boxes):
    return {n: signature(frame, b) for n, b in boxes.items()}


def diff(base_snap, now_snap, threshold=18.0):
    out = []
    for n, a in base_snap.items():
        b = now_snap.get(n)
        if a is None or b is None or a.shape != b.shape:
            continue
        score = float(np.mean(np.abs(a - b)))
        if score > threshold:
            out.append((n, score))
    out.sort(key=lambda t: -t[1])
    return out


# ================================================================ drawing
def draw(frame, base, boxes, changed=(), show_boxes=True):
    view = frame.copy()

    # board outline
    cv2.polylines(view, [np.int32(base)], True, (0, 200, 255), 2)

    # grid on the board plane
    for i in range(9):
        p0 = board_to_image(base, i, 0); p1 = board_to_image(base, i, 8)
        cv2.line(view, tuple(np.int32(p0)), tuple(np.int32(p1)), (0, 120, 0), 1)
        p0 = board_to_image(base, 0, i); p1 = board_to_image(base, 8, i)
        cv2.line(view, tuple(np.int32(p0)), tuple(np.int32(p1)), (0, 120, 0), 1)

    if show_boxes:
        for n, b in boxes.items():
            cv2.rectangle(view, (b[0], b[1]), (b[2], b[3]), (60, 60, 60), 1)

    for n in changed:
        b = boxes[n]
        cv2.rectangle(view, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
        cv2.putText(view, n, (b[0], b[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

    return view


def nudge(cap, base, top):
    """Fine-tune the board corners with the keyboard."""
    labels = ["a1", "h1", "h8", "a8"]
    pts = base.copy()
    idx = 0
    print("\n[Nudge] TAB = next, arrows = 1px, wasd = 10px, ENTER = save, ESC = cancel")

    while True:
        ok, frame = cap.read()
        if not ok:
            return None
        boxes = square_boxes(pts, top)
        view = draw(frame, pts, boxes, show_boxes=False)
        for i, p in enumerate(pts):
            c = (0, 0, 255) if i == idx else (0, 255, 0)
            cv2.circle(view, (int(p[0]), int(p[1])), 8, c, -1)
            cv2.putText(view, labels[i], (int(p[0]) + 12, int(p[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        cv2.putText(view, f"nudging {labels[idx]}  TAB next  ENTER save  ESC cancel",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("camera", view)

        k = cv2.waitKeyEx(20)
        if k == -1:
            continue
        key = k & 0xFF
        if key == 9:
            idx = (idx + 1) % 4
        elif key in (13, 10):
            return pts
        elif key == 27:
            return None
        elif k in (2424832, 65361): pts[idx][0] -= 1
        elif k in (2555904, 65363): pts[idx][0] += 1
        elif k in (2490368, 65362): pts[idx][1] -= 1
        elif k in (2621440, 65364): pts[idx][1] += 1
        elif key == ord("a"): pts[idx][0] -= 10
        elif key == ord("d"): pts[idx][0] += 10
        elif key == ord("w"): pts[idx][1] -= 10
        elif key == ord("s"): pts[idx][1] += 10


# ================================================================ main
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"[Cam] could not open camera {CAM_INDEX}")
        sys.exit(1)

    base, top = load_calib()
    if base is None:
        base, top = calibrate(cap)
        if base is None:
            cap.release()
            return

    height_scale = 1.0
    show_boxes   = True
    baseline     = None
    changed      = ()
    status       = "press b to set the baseline"

    print("\n  b baseline   d detect   |   r reset   n nudge   f rotate")
    print("  h/H piece height   v boxes   s snapshot   q quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes = square_boxes(base, top, height_scale)
        view = draw(frame, base, boxes, changed, show_boxes)

        h_, w_ = view.shape[:2]
        cv2.rectangle(view, (0, h_ - 52), (w_, h_), (0, 0, 0), -1)
        cv2.putText(view, status, (10, h_ - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        cv2.putText(view, f"height x{height_scale:.2f}   "
                          "b base  d detect  h/H height  v boxes  r reset  n nudge",
                    (10, h_ - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (150, 150, 150), 1)
        cv2.imshow("camera", view)

        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break

        elif k == ord("b"):
            baseline = snapshot(frame, boxes)
            changed = ()
            status = "baseline set - make a move, then press d"
            print("[Base] captured")

        elif k == ord("d"):
            if baseline is None:
                status = "no baseline yet - press b first"
                continue
            now = snapshot(frame, boxes)
            ch = diff(baseline, now)
            if not ch:
                status = "nothing changed"
                changed = ()
            else:
                print("\n[Diff]")
                for n, sc in ch[:5]:
                    print(f"    {n}   {sc:6.1f}")
                if len(ch) >= 2:
                    changed = (ch[0][0], ch[1][0])
                    status = f"move between {ch[0][0]} and {ch[1][0]}"
                    print(f"[Move] {ch[0][0]} <-> {ch[1][0]}")
                else:
                    changed = (ch[0][0],)
                    status = f"only {ch[0][0]} changed (capture?)"
                baseline = now

        elif k == ord("h"):
            height_scale = max(0.2, height_scale - 0.05)
            status = f"piece height x{height_scale:.2f}"
        elif k == ord("H"):
            height_scale = min(2.5, height_scale + 0.05)
            status = f"piece height x{height_scale:.2f}"

        elif k == ord("v"):
            show_boxes = not show_boxes

        elif k == ord("f"):
            base = np.float32([base[1], base[2], base[3], base[0]])
            top  = np.float32([top[1], top[2], top[3], top[0]])
            save_calib(base, top)
            baseline = None
            status = "rotated 90 - press f again if still wrong"

        elif k == ord("n"):
            new = nudge(cap, base, top)
            if new is not None:
                base = new
                save_calib(base, top)
                baseline = None
                status = "nudged - press b for a new baseline"

        elif k == ord("r"):
            forget_calib()
            nb, nt = calibrate(cap)
            if nb is not None:
                base, top = nb, nt
                baseline = None
                changed = ()
                status = "reset - press b for a new baseline"

        elif k == ord("s"):
            cv2.imwrite("side_snapshot.png", view)
            print("[Save] side_snapshot.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
