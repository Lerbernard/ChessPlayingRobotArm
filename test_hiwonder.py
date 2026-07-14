"""
xArm 1S -- write-only servo sweep test.

This board's firmware does not answer HID queries, so there is no read-back.
Everything here is command-and-observe. Designed so that a jammed joint and a
dead joint look DIFFERENT, which the +/-30 twitch test could not do.

Method:
  * all servos unloaded (limp) except the one under test -> nothing fights
  * slow, FULL-RANGE sweep -> visible even on the gripper, and pulls a joint
    away from a hard stop instead of pushing it into one
  * you report what you saw; the map is printed at the end

pip install hidapi

BEFORE RUNNING: power the arm OFF for ~10 min, then back on. Hiwonder LX servos
latch a stall/over-temp alarm and refuse to move until power-cycled. That alarm
is what a lit LED usually means -- it is not a "power is fine" light.
"""

import time
import hid

VID, PID = 0x0483, 0x5750

SIG        = 0x55
CMD_MOVE   = 3
CMD_UNLOAD = 20

ALL = [1, 2, 3, 4, 5, 6]

# Conservative sweep windows. Wide enough to be unmistakable, short of the
# hard stops. Gripper gets its own range because its travel is small.
SWEEP = {
    1: (250, 600),   # gripper  -- open <-> closed
    2: (300, 700),
    3: (300, 700),
    4: (300, 700),
    5: (350, 650),   # shoulder carries the most load; keep it tighter
    6: (250, 750),
}


class XArm:
    def __init__(self):
        self.dev = hid.device()
        self.dev.open(VID, PID)

    def _send(self, cmd, params=()):
        params = list(params)
        report = [0x00, SIG, SIG, len(params) + 2, cmd] + params
        self.dev.write(report + [0] * (65 - len(report)))

    def move(self, sid, pos, ms):
        self._send(CMD_MOVE, [1, ms & 0xFF, (ms >> 8) & 0xFF,
                              sid, pos & 0xFF, (pos >> 8) & 0xFF])

    def unload(self, ids=ALL):
        self._send(CMD_UNLOAD, [len(ids)] + list(ids))

    def close(self):
        self.dev.close()


def sweep(arm, sid):
    lo, hi = SWEEP[sid]
    mid = (lo + hi) // 2

    # everything limp except the servo under test
    arm.unload([i for i in ALL if i != sid])
    time.sleep(0.3)

    print(f"    center {mid} ...");  arm.move(sid, mid, 1500); time.sleep(1.8)
    print(f"    -> {lo} ...");       arm.move(sid, lo, 2000);  time.sleep(2.3)
    print(f"    -> {hi} ...");       arm.move(sid, hi, 2500);  time.sleep(2.8)
    print(f"    -> {mid} ...");      arm.move(sid, mid, 2000); time.sleep(2.3)


def main():
    arm = XArm()
    print("Connected.\n")
    print("Lay the arm on its side or hold it -- the untested joints go LIMP.")
    print("Watch and LISTEN. A whine or click with no motion = stripped gears.")
    print("Dead silence with no motion = dead electronics.\n")
    input("Press Enter to begin...")

    arm.unload()
    print("\nAll servos unloaded. Pose the arm into a relaxed, folded position")
    print("with room to move -- NOT standing straight up.")
    input("Press Enter when it's posed...\n")

    results = {}
    for sid in ALL:
        print(f"--- ID {sid}  (range {SWEEP[sid][0]}-{SWEEP[sid][1]}) ---")
        input("    Press Enter to sweep...")
        sweep(arm, sid)
        ans = input("    What moved? (joint name, or 'none', or 'noise'): ").strip()
        results[sid] = ans or "none"
        print()

    arm.unload()
    print("\n=========== RESULT ===========")
    for sid in ALL:
        print(f"  ID {sid}  ->  {results[sid]}")
    print("==============================")
    print("\nServos left unloaded. Power cycle to restore torque.")
    arm.close()


if __name__ == "__main__":
    main()