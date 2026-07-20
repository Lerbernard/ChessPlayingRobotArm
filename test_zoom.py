"""
WEBCAM ZOOM / FIELD-OF-VIEW TESTER

Most webcams have a FIXED lens - "zoom" only crops in, never out. But some
default to a cropped mode, so you may have field of view you are not using.
Two things are worth testing:

  1. CAP_PROP_ZOOM        - does the camera expose a zoom control at all?
  2. RESOLUTION           - many webcams crop the sensor at lower resolutions
                            and only give the FULL field of view at max res.
                            This is usually where the missing width is hiding.

    python test_zoom.py

Keys:
    z / x   zoom out / in   (if supported)
    0       reset zoom to widest
    r       cycle through resolutions
    s       save a snapshot of the current setting
    p       print every property the camera reports
    q       quit
"""

import cv2

CAM_INDEX = 1        # change to whatever find_camera.py reported

# Widest first. If the field of view changes between these, the camera is
# cropping the sensor at the smaller sizes.
RESOLUTIONS = [
    (1920, 1080),
    (1600, 896),
    (1280, 720),
    (1024, 576),
    (800, 600),
    (640, 480),
]

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"Could not open camera {CAM_INDEX}. Try a different CAM_INDEX.")
    raise SystemExit


# ---------------------------------------------------------------- probing
def probe_zoom():
    """Does this camera actually respond to zoom commands?"""
    original = cap.get(cv2.CAP_PROP_ZOOM)
    print(f"\n  CAP_PROP_ZOOM reads: {original}")

    if original == 0 and not cap.set(cv2.CAP_PROP_ZOOM, 100):
        print("  -> camera does not expose a zoom control (normal for most webcams)")
        return False

    # try to move it and see if the value follows
    for test in (100, 200, 150):
        cap.set(cv2.CAP_PROP_ZOOM, test)
        got = cap.get(cv2.CAP_PROP_ZOOM)
        print(f"     set {test:>4}  ->  reads back {got}")

    cap.set(cv2.CAP_PROP_ZOOM, original)
    moved = got != original
    print("  -> zoom control " + ("WORKS" if moved else "is not responding"))
    return moved


def print_props():
    props = {
        "FRAME_WIDTH":  cv2.CAP_PROP_FRAME_WIDTH,
        "FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
        "FPS":          cv2.CAP_PROP_FPS,
        "ZOOM":         cv2.CAP_PROP_ZOOM,
        "FOCUS":        cv2.CAP_PROP_FOCUS,
        "AUTOFOCUS":    cv2.CAP_PROP_AUTOFOCUS,
        "BRIGHTNESS":   cv2.CAP_PROP_BRIGHTNESS,
        "CONTRAST":     cv2.CAP_PROP_CONTRAST,
        "SATURATION":   cv2.CAP_PROP_SATURATION,
        "EXPOSURE":     cv2.CAP_PROP_EXPOSURE,
        "AUTO_EXPOSURE":cv2.CAP_PROP_AUTO_EXPOSURE,
        "GAIN":         cv2.CAP_PROP_GAIN,
        "PAN":          cv2.CAP_PROP_PAN,
        "TILT":         cv2.CAP_PROP_TILT,
    }
    print("\n  --- what this camera reports ---")
    for name, prop in props.items():
        print(f"    {name:<14} {cap.get(prop)}")
    print()


def set_resolution(w, h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (aw, ah) != (w, h):
        print(f"  asked for {w}x{h}, camera gave {aw}x{ah}")
    return aw, ah


# ---------------------------------------------------------------- main
print("=" * 62)
print("  WEBCAM ZOOM / FIELD-OF-VIEW TEST")
print("=" * 62)

has_zoom = probe_zoom()
print_props()

print("  IMPORTANT: cycle resolutions with 'r' and watch the edges of the")
print("  picture. If MORE of the room appears at 1920x1080 than at 640x480,")
print("  the camera crops at lower resolutions - always use the biggest.\n")
print("  z/x = zoom out/in   0 = widest   r = resolution   s = save   q = quit\n")

res_i = 0
w, h = set_resolution(*RESOLUTIONS[res_i])
zoom = cap.get(cv2.CAP_PROP_ZOOM) or 100

while True:
    ok, frame = cap.read()
    if not ok:
        print("Lost the camera.")
        break

    fh, fw = frame.shape[:2]
    view = frame.copy()

    # centre crosshair, so you can judge the field of view consistently
    cv2.line(view, (fw // 2, 0), (fw // 2, fh), (0, 180, 255), 1)
    cv2.line(view, (0, fh // 2), (fw, fh // 2), (0, 180, 255), 1)

    bar = f"{fw}x{fh}"
    if has_zoom:
        bar += f"   zoom={zoom:.0f}"
    else:
        bar += "   (no zoom control)"

    cv2.rectangle(view, (0, 0), (fw, 40), (0, 0, 0), -1)
    cv2.putText(view, bar, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.putText(view, "z/x zoom   r resolution   s save   q quit",
                (10, fh - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)

    cv2.imshow("zoom test", view)
    k = cv2.waitKey(1) & 0xFF

    if k == ord("q"):
        break

    elif k == ord("r"):
        res_i = (res_i + 1) % len(RESOLUTIONS)
        w, h = set_resolution(*RESOLUTIONS[res_i])
        print(f"  resolution -> {w}x{h}")

    elif k == ord("z") and has_zoom:
        zoom = max(0, zoom - 20)
        cap.set(cv2.CAP_PROP_ZOOM, zoom)
        zoom = cap.get(cv2.CAP_PROP_ZOOM)

    elif k == ord("x") and has_zoom:
        zoom += 20
        cap.set(cv2.CAP_PROP_ZOOM, zoom)
        zoom = cap.get(cv2.CAP_PROP_ZOOM)

    elif k == ord("0") and has_zoom:
        cap.set(cv2.CAP_PROP_ZOOM, 0)
        zoom = cap.get(cv2.CAP_PROP_ZOOM)
        print("  zoom reset to widest")

    elif k == ord("p"):
        print_props()

    elif k == ord("s"):
        name = f"fov_{fw}x{fh}.png"
        cv2.imwrite(name, frame)
        print(f"  saved {name}")

cap.release()
cv2.destroyAllWindows()

print("\n  If none of that widened the view, the lens is fixed. Options:")
print("    - raise the camera (doubling the height roughly doubles coverage)")
print("    - the maker's own utility may have a field-of-view setting that")
print("      OpenCV cannot reach (Logi Tune for Logitech, for example)")
