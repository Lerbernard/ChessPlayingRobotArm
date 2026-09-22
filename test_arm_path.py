"""
Dry run of the arm's motion path - NO hardware, NO camera, NO Stockfish.

It imports chess_pi.py against fake serial/Dobot/OpenCV/chess modules, plays a
quiet move and a capture, and prints every point the arm would be sent to. Use
it to see what the arm will do (and to count the moves) before touching the
real thing:

    python3 test_arm_path.py

A quiet move should be exactly 6 arm moves: up, across, down, (grip), up,
across, down, (release), up - i.e. one trip per piece, no detours.
"""

import sys
import types
from unittest import mock

MOVES = []          # everything the arm is told to do, in order


# ---------------------------------------------------------------- fake serial
class FakeSerial:
    def __init__(self, *a, **k):
        self.port = self.baudrate = self.timeout = self.dtr = self.rts = None

    def open(self): pass
    def reset_input_buffer(self): pass
    def write(self, b): pass
    def readline(self): return b""
    def close(self): pass


# ---------------------------------------------------------------- fake Dobot
class FakeResponse:
    params = None          # no queue index -> chess_pi falls back to timing


class FakeDobot:
    """A perfect arm: it lands exactly where it is sent."""

    def __init__(self, *a, **k):
        self.pos = [205.1, -5.4, 120.0]
        self.ser = types.SimpleNamespace(timeout=0.5)

    def speed(self, *a): pass
    def grip(self, v): MOVES.append(("grip", v)); return FakeResponse()
    def suck(self, v): MOVES.append(("suck", v)); return FakeResponse()
    def pose(self): return tuple(self.pos) + (0, 0, 0, 0, 0)

    def _set_ptp_cmd(self, x, y, z, r, mode=None, wait=False):
        MOVES.append(("move", x, y, z))
        self.pos = [x, y, z]
        return FakeResponse()

    def move_to(self, x, y, z, r, wait=False):
        MOVES.append(("MOVJ", x, y, z))       # joint move: arcs, should not happen
        self.pos = [x, y, z]
        return FakeResponse()

    def _send_command(self, m): pass
    def _get_queued_cmd_current_index(self): return 10 ** 9
    def close(self): pass


def install_fakes():
    serial = types.ModuleType("serial")
    serial.Serial = FakeSerial
    sys.modules["serial"] = serial

    pydobot = types.ModuleType("pydobot")
    pydobot.Dobot = FakeDobot
    message = types.ModuleType("pydobot.message")
    message.Message = type("Message", (), {"id": None, "ctrl": None})
    enums = types.ModuleType("pydobot.enums")
    enums.PTPMode = types.SimpleNamespace(MOVL_XYZ=2)
    ids = types.ModuleType("pydobot.enums.CommunicationProtocolIDs")
    ids.CommunicationProtocolIDs = types.SimpleNamespace(CLEAR_ALL_ALARMS_STATE=20)
    ctrl = types.ModuleType("pydobot.enums.ControlValues")
    ctrl.ControlValues = types.SimpleNamespace(ONE=1)
    enums.CommunicationProtocolIDs = ids.CommunicationProtocolIDs
    enums.ControlValues = ctrl.ControlValues
    for name, mod in (("pydobot", pydobot), ("pydobot.message", message),
                      ("pydobot.enums", enums),
                      ("pydobot.enums.CommunicationProtocolIDs", ids),
                      ("pydobot.enums.ControlValues", ctrl)):
        sys.modules[name] = mod

    sys.modules["cv2"] = mock.MagicMock()
    numpy = mock.MagicMock()
    numpy.float32 = lambda x: x
    sys.modules["numpy"] = numpy
    chess = mock.MagicMock()
    sys.modules["chess"] = chess
    sys.modules["chess.engine"] = chess.engine


class ShellDobot(FakeDobot):
    """
    A realistic arm: its workspace is a shell, so the higher it goes the less
    it can fold in toward its base. Asked for a point inside that limit it
    stops short, on the same line, exactly as the real one does.

    This is the arm from the g8 failure: asked for (120.2, -86.4, 65.0) - a
    radius of 148 mm - it stopped at a radius of 181.
    """

    @staticmethod
    def min_radius(z):
        return 148.0 + max(0.0, z) * 0.5        # 148 at board height, 181 at z=65

    def _set_ptp_cmd(self, x, y, z, r, mode=None, wait=False):
        import math
        MOVES.append(("move", x, y, z))
        lo = self.min_radius(z)
        d = math.hypot(x, y)
        if d < lo:                               # refused - stop short on the line
            x, y = x * lo / d, y * lo / d
        self.pos = [x, y, z]
        return FakeResponse()


def report(title):
    print(f"\n=== {title} ===")
    for m in MOVES:
        if m[0] in ("move", "MOVJ"):
            print(f"   {m[0]:5s} x={m[1]:7.1f}  y={m[2]:7.1f}  z={m[3]:7.1f}")
        else:
            print(f"   {m[0]}({m[1]})")
    moves = [m for m in MOVES if m[0] in ("move", "MOVJ")]
    grips = [m for m in MOVES if m[0] == "grip"]
    print(f"   -> {len(moves)} arm moves, {len(grips)} gripper commands")
    return moves


def main():
    install_fakes()
    import chess_pi

    chess_pi.time.sleep = lambda *a: None      # no waiting in a dry run
    print(f"\ntravel height {chess_pi.TRAVEL_Z:.1f} mm, "
          f"square pitch {chess_pi.PITCH:.1f} mm")

    chess_pi.go_park()
    MOVES.clear()
    chess_pi.arm_move_piece("e2", "e4", "P")
    quiet = report("quiet move: e2 -> e4")

    MOVES.clear()
    chess_pi.arm_remove_piece("d5", "p")       # the captured piece first
    chess_pi.arm_move_piece("e4", "d5", "P")
    capture = report("capture: exd5")

    # ---- the same move on an arm that cannot fold in up high ----
    MOVES.clear()
    chess_pi.arm = ShellDobot()
    chess_pi.LANE_R["min"] = chess_pi.LANE_R["max"] = None
    chess_pi._last[:] = [205.1, -5.4, 120.0]
    chess_pi._claw["closed"] = None
    near = min(("a8", "h8"),
               key=lambda sq: (board := chess_pi.board_config.SQUARE_MAP[sq]["p"])
               and (board["x"] ** 2 + board["y"] ** 2))
    e = chess_pi.board_config.SQUARE_MAP[near]["p"]
    print(f"\n(now on an arm whose travel lane folds in no closer than "
          f"{ShellDobot.min_radius(chess_pi.TRAVEL_Z):.0f} mm; {near} sits at "
          f"{(e['x'] ** 2 + e['y'] ** 2) ** 0.5:.0f} mm)")
    picked = chess_pi.pick(e["x"], e["y"], e["z"], near)
    shell = report(f"far-rank pick: {near}")

    failures = []
    if not picked:
        failures.append(f"{near} could not be picked on the shell arm")
    if chess_pi.LANE_R["min"] is None:
        failures.append("the lane's fold-in limit was never learned")
    if not any(m[0] == "grip" and m[1] is True for m in MOVES):
        failures.append("the claw never closed on the piece")
    if len(quiet) != 6:
        failures.append(f"quiet move took {len(quiet)} arm moves, expected 6")
    if len(capture) != 12:
        failures.append(f"capture took {len(capture)} arm moves, expected 12")
    for m in quiet + capture:
        if m[0] == "MOVJ":
            failures.append("a joint move (MOVJ) was used - it arcs through pieces")
            break

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("OK: one trip per piece, all moves linear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
