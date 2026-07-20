"""
FIND YOUR WEBCAM

Windows numbers cameras arbitrarily, and index 0 is usually the laptop's
built-in one. This opens each camera it can find so you can see which is which.

    python find_camera.py

Press SPACE to move to the next camera, q to quit.
The index you want gets printed when you quit.
"""

import cv2

MAX_INDEX = 6          # how many indices to probe

print("Probing camera indices...\n")

found = []
for i in range(MAX_INDEX):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  index {i}:  working  ({w}x{h})")
            found.append(i)
        else:
            print(f"  index {i}:  opens but gives no frames")
        cap.release()
    else:
        print(f"  index {i}:  nothing")

if not found:
    print("\nNo cameras found. Check it's plugged in, and that no other app")
    print("(Teams, Zoom, Camera) is holding it open.")
    raise SystemExit

print(f"\nWorking cameras: {found}")
print("\nShowing each one. SPACE = next, q = quit.\n")

for i in found:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        cv2.putText(frame, f"CAM_INDEX = {i}   ({w}x{h})", (14, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE = next camera,  q = quit", (14, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("which camera is this?", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            break
        if k == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nPut this at the top of chess_vision.py:   CAM_INDEX = {i}")
            raise SystemExit

    cap.release()

cv2.destroyAllWindows()
print("\nThat was all of them. Re-run to look again.")
