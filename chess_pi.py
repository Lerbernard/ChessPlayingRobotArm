"""
================================================================================
  ROBOTIC CHESS  -  everything on the Raspberry Pi, one script
================================================================================

  Camera (ov5647) reads the human's move, Stockfish replies, the Dobot plays it
  and then parks clear of the board so the camera can see it again. The ESP32
  touchscreen drives the whole thing over serial:

      board calibration  ->  colour training  ->  play  ->  END TURN

  START HERE:
    1. Plug the Dobot and the ESP32 into the Pi's USB ports.
    2. Set DOBOT_PORT / ESP_PORT below (ls /dev/serial/by-id or /dev/ttyUSB*).
    3. Set STOCKFISH_PATH (on the Pi: sudo apt install stockfish, then the path
       is usually /usr/games/stockfish).
    4. Run at the Pi's OWN screen (the camera windows need a display):
           python3 chess_pi.py

  This file drives the ARM and ENGINE and CAMERA. The board geometry and colour
  hues come from board_corners.json (written by the on-screen calibration).
  The arm's square coordinates come from board_config.py (SQUARE_MAP), written
  by the on-screen ARM calibration.
================================================================================
"""

import os
import json
import math
import time
import threading

import cv2
import numpy as np
import chess
import chess.engine
from pydobot import Dobot

import board_config

# ------------------------------------------------------------------ CONFIG
# Stable by-id paths so a reboot can't swap ttyUSB0/ttyUSB1.
DOBOT_PORT     = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
ESP_PORT       = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
STOCKFISH_PATH = "/usr/games/stockfish"  # confirmed with `which stockfish`

# --- camera / board geometry ---
WARP_SIZE   = 640
CAM_CALIB   = "board_corners.json"       # corners + k1 + hues
FILES, RANKS = "abcdefgh", "12345678"
INSET, RATIO, THRESHOLD = 0.30, 0.40, 12.0
SIG_SIZE = (32, 32)
W_BRIGHT, W_EDGE, W_COLOUR = 0.2, 2.0, 1.0
EDGE_GAIN = 2.5

# --- arm motion ---
CLEARANCE      = 50.0    # claw fingers hang below the Z point - clear the kings
TRAVEL_Z_FLOOR = 65.0
MAX_REACH      = 320.0
MOVE_SPEED     = 80.0    # conservative: real speed with accel is below the setting
MOVE_BASE      = 0.9     # generous per-move floor so we never outrun the arm
SETTLE         = 0.15    # pause after a move so the arm stops shaking before
                         # the claw opens or closes
THINK_TIME     = 0.5
# The arm self-test at startup is a motion the game does not need. It is off
# by default so the arm stays parked until there is a real move to play; the
# ARM TEST button on the touchscreen still runs it whenever you want it.
SELFTEST_AT_START = False
# The arm's workspace is a SHELL, not a cylinder: up in the travel lane it
# cannot fold in as close to its base - nor stretch as far out - as it can
# down at board height. A square just inside that limit is reachable at PIECE
# height but NOT in the lane above it, and the arm simply stops short and
# trips its alarm. These are the lane's limits, in mm from the base. Leave
# them None and they are learned (and printed) the first time a move stops
# short; paste the printed numbers here to skip that one learning move.
# 184 is what YOUR arm measured: asked for a point 148mm from its base at
# travel height it stopped at 181. Set it back to None to measure again (the
# first far-rank move of the session then costs one wasted reach).
LANE_MIN_R = 184.0       # closest the arm folds in up in the travel lane
LANE_MAX_R = None        # furthest it reaches out up there (None = learn it)
# Where the arm parks (high, clear) before the camera reads the board. Set it
# yourself on the ARM calibration screen (HIGH PARK row); that writes PARK into
# board_config.py. This is just the fallback if you haven't set one.
PARK           = getattr(board_config, "PARK", (205.1, -5.4, 120.0))

# captured-piece piles
GRAVE_W = getattr(board_config, "GRAVEYARD_WHITE", None)
GRAVE_B = getattr(board_config, "GRAVEYARD_BLACK", None)
FLOOR_Z      = getattr(board_config, "FLOOR_Z", None)   # captured table surface
FLOOR_MARGIN = 40.0    # travel lane stays at least this far above the floor

# ---- global board nudge ----
# If the claw lands slightly off-centre on EVERY square in the same direction,
# set these to shift the whole board. Units are mm, ADDED to every square:
#   claw lands short of the square (toward the arm's base)  -> increase X
#   claw lands past the square (away from the base)         -> decrease X
#   claw lands to one side                                  -> adjust Y
# Watch one descent (e.g. on e4), measure the miss with a ruler, set, re-run.
BOARD_OFFSET_X = 0.0
BOARD_OFFSET_Y = 0.0

OVERRIDES = {}


# ================================================================ SERIAL
import serial

esp = serial.Serial()
esp.port, esp.baudrate, esp.timeout = ESP_PORT, 115200, 0.1
esp.dtr = esp.rts = False
esp.open()
time.sleep(0.3)
esp.reset_input_buffer()

def send(msg):
    esp.write((msg + "\n").encode())
    print(f"  -> ESP: {msg}")


# ================================================================ ARM
def _check_gripper(a):
    """The pneumatic gripper needs both grip() (valve) and suck() (pump)."""
    ok = hasattr(a, "grip") and hasattr(a, "suck")
    if ok:
        print("[Gripper] pneumatic gripper ready (grip valve + suck pump)")
    else:
        print("[Gripper] WARNING: arm.grip/arm.suck missing - gripper may not work")
    return ok

print(f"[Arm] connecting on {DOBOT_PORT} ...")
arm = Dobot(port=DOBOT_PORT, verbose=False)
# pydobot reads the arm's reply with a short serial timeout. If the reply is
# late it returns None -> None.params -> AttributeError. Give it more time.
# (This is exactly what the working chess_bridge.py did.)
for attr in ("ser", "_serial", "serial"):
    sp = getattr(arm, attr, None)
    if sp is not None and hasattr(sp, "timeout"):
        sp.timeout = 2.0
        print(f"[Serial] Dobot read timeout raised to 2.0 s (arm.{attr})")
        break
arm.speed(100, 100)
_check_gripper(arm)

def clear_alarms():
    """
    Clear the Dobot's alarm state (the RED light). It trips when the arm is
    told to reach a point outside its ~320 mm radius - which the far rank
    sits right at. After clearing, the arm accepts moves again.
    """
    try:
        from pydobot.message import Message
        from pydobot.enums.CommunicationProtocolIDs import CommunicationProtocolIDs
        from pydobot.enums.ControlValues import ControlValues
        msg = Message()
        msg.id = CommunicationProtocolIDs.CLEAR_ALL_ALARMS_STATE
        msg.ctrl = ControlValues.ONE
        arm._send_command(msg)
        print("[Arm] alarms cleared")
    except Exception as e:
        print(f"[Arm] could not clear alarms: {e}")

clear_alarms()

def dobot(fn, *a, tries=4, **kw):
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except (AttributeError, TypeError) as e:
            print(f"  [Serial] Dobot didn't answer ({e}) - retry {i+1}/{tries}")
            time.sleep(0.5)
    raise RuntimeError("Dobot stopped responding. Power-cycle the arm.")

_MOVL = None
if hasattr(arm, "_set_ptp_cmd"):
    for getter in (
        lambda: __import__("pydobot.enums", fromlist=["PTPMode"]).PTPMode.MOVL_XYZ,
        lambda: 2,
    ):
        try:
            _MOVL = getter()
            print(f"[Motion] linear MOVL_XYZ enabled (mode={_MOVL})")
            break
        except Exception:
            continue
if _MOVL is None:
    print("[Motion] WARNING: MOVL unavailable, falling back to MOVJ (arcs)")

def _claw_cmd(fn, val, dwell):
    """
    Send a gripper command and WAIT for the arm to report it executed (same
    queue confirmation as motion), then dwell for the air to actually move.
    Without the queue wait the valve can still be pending while the arm has
    already started its next move.
    """
    try:
        resp = dobot(fn, val)
        idx = None
        try:
            if resp is not None and getattr(resp, "params", None):
                idx = _struct.unpack_from("L", resp.params, 0)[0]
        except Exception:
            idx = None
        if idx is not None:
            _wait_queue(idx, timeout=3.0)
        time.sleep(dwell)
    except Exception as e:
        print(f"[Gripper] command failed: {e}")

# What we last COMMANDED the claw to do. None = unknown (at startup we have no
# idea where the fingers are, so the first command is always sent). Knowing the
# state lets us skip a repeat command, and each one costs a full second of dwell.
_claw = {"closed": None}

def claw_open(force=False):
    """
    OPEN the claw. Only grip() drives the fingers - suck() is NOT sent here.
    Sending suck(True) alongside was actuating the gripper itself, so the claw
    snapped open then shut again ("opens and closes fast"). One command, one
    action, with a dwell long enough for the fingers to actually travel.
    """
    if not force and _claw["closed"] is False:
        return                      # already open - don't burn a second on it
    _claw_cmd(arm.grip, False, 0.9)
    _claw["closed"] = False

def claw_close(force=False):
    """CLOSE on the piece. Again grip() only."""
    if not force and _claw["closed"] is True:
        return
    _claw_cmd(arm.grip, True, 1.0)
    _claw["closed"] = True

def claw_pump_off():
    """Stop the air pump once the arm is clear (quiet between moves)."""
    _claw_cmd(arm.suck, False, 0.2)

def control_claw(grab):
    """Close or open the claw (always sent - used by the gripper self-test)."""
    claw_close(force=True) if grab else claw_open(force=True)


def coords(square, piece_symbol):
    if square in OVERRIDES:
        return OVERRIDES[square]
    e = board_config.SQUARE_MAP[square][piece_symbol.lower()]
    return e["x"] + BOARD_OFFSET_X, e["y"] + BOARD_OFFSET_Y, e["z"]

def _board_pitch():
    """Square spacing measured from the SAVED calibration (no fixed number)."""
    try:
        a = board_config.SQUARE_MAP["a1"]["p"]; h = board_config.SQUARE_MAP["h1"]["p"]
        return math.dist((a["x"], a["y"]), (h["x"], h["y"])) / 7.0
    except Exception:
        return 25.0

# ============================================================ MOTION
# Rewritten clean. One primitive (_goto) sends the arm to an absolute point,
# waits for the arm to confirm it executed, then checks where it really is.
# Everything else is built from it. Each phase logs the SQUARE it is acting on
# and the exact coordinate used, so a wrong target is visible immediately.

import struct as _struct

_last = [999.0, 999.0, 999.0]      # last known position; 999 = unknown

# How close counts as "on target". If EVERY move reports a few mm off in the
# log, the arm's pose readback simply is not that repeatable - raise this
# rather than letting good moves be reported as failures.
VERIFY_TOL  = 5.0                  # mm: closer than this counts as on target
CORRECT_MAX = 30.0                 # mm: nudge back on target up to this much
HOVER       = 20.0                 # mm above the grip point: the height the
                                   # claw slides in at when it has to come in
                                   # along a file (see "the two operations")

def _wait_queue(expected_idx, timeout):
    """Block until the arm reports it executed command `expected_idx`."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if arm._get_queued_cmd_current_index() >= expected_idx:
                return True
        except Exception:
            pass
        time.sleep(0.08)
    print(f"  [Queue] timeout waiting for command {expected_idx}")
    return False

def _read_pose():
    """Where the arm says it is, or None."""
    try:
        pr = arm.pose()
        return (pr[0], pr[1], pr[2]) if pr else None
    except Exception:
        return None

def _send_move(x, y, z):
    """Issue ONE linear move and wait for the arm to finish executing it."""
    dist = (math.dist(tuple(_last), (x, y, z)) if _last[0] < 900 else 250.0)
    est  = MOVE_BASE + dist / MOVE_SPEED
    # Always MOVL: a joint move arcs, and an arc goes through the pieces.
    if _MOVL is not None:
        resp = dobot(arm._set_ptp_cmd, x, y, z, 0, mode=_MOVL, wait=False)
    else:
        resp = dobot(arm.move_to, x, y, z, 0, wait=False)
    idx = None
    try:
        if resp is not None and getattr(resp, "params", None):
            idx = _struct.unpack_from("L", resp.params, 0)[0]
    except Exception:
        idx = None
    if idx is not None:
        _wait_queue(idx, timeout=max(3.0, est * 3))
        time.sleep(0.05)
    else:
        time.sleep(est)

def _miss(x, y, z):
    """
    Update _last from the arm and return how far it is from (x, y, z).
    Returns None if the arm will not say where it is - then we trust the
    command, exactly as before.
    """
    pos = _read_pose()
    if pos is None:
        _last[0], _last[1], _last[2] = x, y, z
        return None
    _last[0], _last[1], _last[2] = pos
    return math.dist(pos, (x, y, z))

_last_fail = {"reason": None}      # why the last _goto returned False

def _goto(x, y, z, label="", nudge=True):
    """
    Send the arm to ONE absolute point and confirm it got there.
    Returns True if the arm is within tolerance of (x, y, z).
    """
    start = (_last[0], _last[1], _last[2]) if _last[0] < 900 else None
    _last_fail["reason"] = None
    try:
        _send_move(x, y, z)
    except Exception as e:
        print(f"  [Move] {label} command failed: {e}")
        clear_alarms()
        _last_fail["reason"] = "command"
        return False

    off = _miss(x, y, z)
    if off is None or off <= VERIFY_TOL:
        return True

    # Did the arm travel, but stop at a different distance from its base? Then
    # it went as far along the line as its workspace allows: that is the reach
    # limit at this height, not a hiccup. Nudging and clearing alarms cannot
    # conjure reach, so give up on the point at once and let the caller find
    # another way in. (An arm that did NOT move was ignoring the command -
    # that IS worth a retry, so it falls through below.)
    moved = math.dist(start, (_last[0], _last[1], _last[2])) if start else 999.0
    r_want, r_got = math.hypot(x, y), math.hypot(_last[0], _last[1])
    if moved > 2.0 and abs(r_got - r_want) > VERIFY_TOL:
        print(f"  [Move] {label}: stopped {r_got:.0f}mm from the base, asked "
              f"for {r_want:.0f}mm - the arm cannot reach that far "
              f"{'in' if r_got > r_want else 'out'} at z={z:.0f}")
        _last_fail["reason"] = "reach"
        return False

    if nudge and off <= CORRECT_MAX:
        print(f"  [Move] {label}: {off:.1f}mm off - nudging onto target")
        try:
            _send_move(x, y, z)
            off = _miss(x, y, z)
        except Exception:
            pass
        if off is None or off <= VERIFY_TOL:
            return True

    if nudge:
        # The usual reason the arm ignores a move is the ALARM state (the RED
        # light): it trips on a point at the edge of the reach and from then on
        # every command is dropped silently. Clear it and try the point once
        # more before giving up on the move.
        print(f"  [Move] {label}: still {off:.1f}mm off - clearing alarms, retrying")
        clear_alarms()
        time.sleep(0.3)
        try:
            _send_move(x, y, z)
            off = _miss(x, y, z)
        except Exception:
            pass
        if off is None or off <= VERIFY_TOL:
            return True

    print(f"  [Move] {label}: asked ({x:.1f},{y:.1f},{z:.1f}) but arm is at "
          f"({_last[0]:.1f},{_last[1]:.1f},{_last[2]:.1f}) - off {off:.1f}mm")
    _last_fail["reason"] = "off_target"
    return False

# ---- board-derived heights -------------------------------------------------
_BOARD_TOP_Z = 0.0
TRAVEL_Z     = TRAVEL_Z_FLOOR
PITCH        = 25.0
FILE_AXIS    = (1.0, 0.0)

def _recompute_geometry():
    """
    Re-derive everything measured from the saved board map: the top of the
    board, the travel lane above it and the square pitch. Called at startup and
    again after anything rewrites board_config, so a fresh calibration can
    never leave a stale travel height (or pitch) behind.
    """
    global _BOARD_TOP_Z, TRAVEL_Z, PITCH, FILE_AXIS
    _BOARD_TOP_Z = max(sq["p"]["z"] for sq in board_config.SQUARE_MAP.values())
    TRAVEL_Z = max(_BOARD_TOP_Z + CLEARANCE, TRAVEL_Z_FLOOR)
    if FLOOR_Z is not None:
        TRAVEL_Z = max(TRAVEL_Z, FLOOR_Z + FLOOR_MARGIN)
    PITCH = _board_pitch()
    FILE_AXIS = _file_axis()

def _file_axis():
    """
    Unit vector along a FILE of the board (rank 8 -> rank 1), measured from
    the saved calibration. Sliding into a square along its own file is what
    lets pieces on that file pass BETWEEN the open fingers instead of being
    knocked over.
    """
    try:
        g = lambda sq: board_config.SQUARE_MAP[sq]["p"]
        vx = ((g("a1")["x"] - g("a8")["x"]) + (g("h1")["x"] - g("h8")["x"])) / 2.0
        vy = ((g("a1")["y"] - g("a8")["y"]) + (g("h1")["y"] - g("h8")["y"])) / 2.0
        n = math.hypot(vx, vy)
        if n > 1.0:
            return vx / n, vy / n
    except Exception:
        pass
    return 1.0, 0.0        # fall back to straight out from the arm's base

_recompute_geometry()
print(f"[Motion] travel height {TRAVEL_Z:.1f} mm")

def lift(label=""):
    """Rise to the travel lane, keeping X and Y."""
    if _last[0] > 900:
        pos = _read_pose()     # position unknown - ask before moving anywhere
        if pos is None:
            return
        _last[0], _last[1], _last[2] = pos
    if _last[2] < TRAVEL_Z - 1.0:
        _goto(_last[0], _last[1], TRAVEL_Z, label or "lift")

MIN_GRAB = 148.0    # the arm cannot fold in closer than this

def reach_ok(x, y):
    """
    Keep the target inside the arm's reachable ring, but NEVER move it far
    enough to land on a different square (half a pitch, from calibration).
    Returns (x, y, ok).
    """
    d = math.hypot(x, y)
    limit = PITCH / 2.0
    if d > MAX_REACH:
        shift = d - (MAX_REACH - 2.0)
        if shift > limit:
            print(f"  [Reach] {d:.0f}mm is {shift:.0f}mm past reach (> half a "
                  f"square) - out of reach")
            return x, y, False
        sc = (MAX_REACH - 2.0) / d
        return x * sc, y * sc, True
    if d < MIN_GRAB:
        shift = MIN_GRAB - d
        if shift > limit:
            print(f"  [Reach] {d:.0f}mm is {shift:.0f}mm inside the fold limit "
                  f"(> half a square) - out of reach")
            return x, y, False
        sc = MIN_GRAB / max(d, 1.0)
        return x * sc, y * sc, True
    return x, y, True

# ---- the two operations ----------------------------------------------------
# One square = one trip: UP into the travel lane, ACROSS to the square, DOWN,
# act, UP again. Three moves, nothing in between.
#
# The one thing that complicates it is the arm's reach. Its workspace is a
# SHELL, not a cylinder: up in the travel lane it cannot fold in as close to
# its base - nor stretch as far out - as it can down at board height. The rank
# nearest the base sits right on that limit, so those squares are reachable at
# PIECE height but not in the lane above them: the arm stops short and trips
# the alarm, which looks like it wandering out over the board and coming back.
#
# So for a square the lane cannot reach, the arm goes as far along that
# square's OWN FILE as the lane does reach, drops to HOVER above the pieces
# there, and slides in along the file - pieces on the file pass between the
# open fingers. It backs out the same way before it rises. Every other square
# is still the straight three moves, and nothing ever detours via the middle
# of the board.

LANE_R = {"min": LANE_MIN_R, "max": LANE_MAX_R}

def _at_radius(x, y, r):
    """The point at radius r straight out from the base through (x, y)."""
    d = math.hypot(x, y) or 1.0
    return x * r / d, y * r / d

def _along_file_to_radius(x, y, want_r):
    """
    Walk along the square's file until the radius from the base reaches
    want_r, and return that point. (Solves |P + t*A| = want_r for the smaller
    |t|.) If the file never reaches that radius, fall back to the point
    straight out from the base.
    """
    ax, ay = FILE_AXIS
    b = x * ax + y * ay
    c = x * x + y * y - want_r * want_r
    disc = b * b - c
    if disc < 0:
        return _at_radius(x, y, want_r)
    root = math.sqrt(disc)
    t = min((-b + root, -b - root), key=abs)
    return x + t * ax, y + t * ay

def lane_point(x, y):
    """
    Where the arm sits in the TRAVEL LANE when it is working on (x, y): the
    square itself when the lane reaches it, otherwise the nearest point along
    its file that the lane does reach.
    """
    r = math.hypot(x, y)
    if LANE_R["min"] is not None and r < LANE_R["min"] - 0.5:
        return _along_file_to_radius(x, y, LANE_R["min"])
    if LANE_R["max"] is not None and r > LANE_R["max"] + 0.5:
        return _along_file_to_radius(x, y, LANE_R["max"])
    return x, y

def _learn_lane_limit(want_x, want_y):
    """
    A lane move stopped short. If the arm ended at a different RADIUS than it
    was asked for, that radius is the lane's reach limit - remember it, so
    from now on the arm comes in along the file instead of failing first.
    Returns True if a limit was learned.
    """
    want_r = math.hypot(want_x, want_y)
    got_r = math.hypot(_last[0], _last[1])
    if got_r > want_r + VERIFY_TOL and got_r < MAX_REACH:
        LANE_R["min"] = got_r + 3.0
        print(f"  [Reach] up in the travel lane the arm folds in no closer "
              f"than {LANE_R['min']:.0f}mm, so it will come in along the file."
              f" Put LANE_MIN_R = {LANE_R['min']:.0f} in CONFIG to skip this.")
        return True
    if got_r < want_r - VERIFY_TOL and got_r > MIN_GRAB:
        LANE_R["max"] = got_r - 3.0
        print(f"  [Reach] up in the travel lane the arm reaches out no further"
              f" than {LANE_R['max']:.0f}mm, so it will come in along the file."
              f" Put LANE_MAX_R = {LANE_R['max']:.0f} in CONFIG to skip this.")
        return True
    return False

def travel_to(x, y, label=""):
    """
    Rise into the travel lane, then cross to (x, y) - still up high.

    If the straight line is refused, go again in two shorter linear hops
    through the halfway point: a Dobot will not always interpolate one long
    MOVL right across its workspace. Both ends of the hop are at travel
    height, so the halfway point is too - there is nothing up there to hit.
    """
    name = label or "travel"
    lift(f"lift before {name}")
    if _goto(x, y, TRAVEL_Z, f"over {name}"):
        return True
    if _last_fail["reason"] == "reach":
        return False           # splitting the line does not buy any reach
    mx, my = (_last[0] + x) / 2.0, (_last[1] + y) / 2.0
    print(f"  [Move] straight line to {name} refused - going via the halfway "
          f"point ({mx:.1f}, {my:.1f})")
    _goto(mx, my, TRAVEL_Z, "halfway")
    return _goto(x, y, TRAVEL_Z, f"over {name}")

def approach(x, y, z, square="?"):
    """
    Bring the claw down onto (x, y, z), whichever way the arm's reach allows.
    Returns True only if it actually arrived.
    """
    lx, ly = lane_point(x, y)
    if not travel_to(lx, ly, square):
        if _learn_lane_limit(lx, ly):
            clear_alarms()                     # the short stop will have tripped it
            time.sleep(0.2)
            lx, ly = lane_point(x, y)          # limits changed - new entry point
            if not travel_to(lx, ly, square):
                return False
        elif math.dist((_last[0], _last[1]), (lx, ly)) > PITCH / 2.0:
            print(f"  [Reach] {square}: the arm stopped more than half a square"
                  f" from where it should be. Is the RED alarm light on?")
            return False
    if math.dist((lx, ly), (x, y)) > 0.5:      # coming in along the file
        if not _goto(lx, ly, z + HOVER, f"down on {square}'s file"):
            return False
        if not _goto(x, y, z + HOVER, f"in along the file to {square}"):
            return False
        return _goto(x, y, z, f"down onto {square}")
    return _goto(x, y, z, f"down onto {square}")

def retreat(square="?"):
    """
    Leave the square the arm is standing on and rise into the travel lane -
    the exact reverse of approach(), so a square that had to be entered along
    its file is left along it too (the arm cannot lift straight out of one).
    """
    if _last[0] > 900:
        return
    x, y, z = _last[0], _last[1], _last[2]
    lx, ly = lane_point(x, y)
    if math.dist((lx, ly), (x, y)) > 0.5:
        _goto(x, y, z + HOVER, f"lift clear of {square}")
        _goto(lx, ly, _last[2], f"back out along {square}'s file")
    _goto(lx, ly, TRAVEL_Z, f"lift from {square}")

def pick(x, y, z, square="?"):
    """Open the claw, get to the square, close on the piece, come away."""
    print(f"  [Pick] {square} at ({x:.1f}, {y:.1f}, {z:.1f})")
    claw_open()                                   # opens while it travels
    if not approach(x, y, z, square):
        print(f"  [Pick] {square}: could not get to the piece - not grabbing "
              f"(the [Move] line above says where the arm actually stopped)")
        retreat(square)
        return False
    time.sleep(SETTLE)
    claw_close()
    retreat(square)
    return True

def place(x, y, z, square="?"):
    """Carrying a piece: get to the square, lower it in, release, come away."""
    print(f"  [Place] {square} at ({x:.1f}, {y:.1f}, {z:.1f})")
    ok = approach(x, y, z, square)
    if not ok:
        # Holding on is no better - the next move would open the claw over
        # some other square - so let it go here and say exactly where.
        print(f"  [Place] {square}: could not get there. Releasing at "
              f"({_last[0]:.1f}, {_last[1]:.1f}) - put the piece on {square} "
              f"by hand.")
    time.sleep(SETTLE)
    claw_open()
    retreat(square)
    claw_pump_off()
    return ok

def go_park():
    """Park high and clear so the camera sees the whole board."""
    px, py, pz = PARK
    print(f"[Park] to ({px:.1f}, {py:.1f}, {pz:.1f})")
    if _last[0] > 900:
        try:
            dobot(arm.move_to, px, py, pz, 0, wait=False)
            time.sleep(MOVE_BASE + 2.0)
            pos = _read_pose()
            if pos:
                _last[0], _last[1], _last[2] = pos
        except Exception as e:
            print(f"[Park] blind move failed: {e}")
            clear_alarms()
        return
    retreat("park")            # a far-rank square cannot be left straight up
    if not _goto(px, py, pz, "park"):
        clear_alarms()
        time.sleep(0.2)
        _goto(px, py, pz, "park retry")
    time.sleep(0.2)

def fit_square_grid(a1, h1, a8, h8):
    """
    Build the board map by BEST-FITTING a perfect square grid to the four
    tapped corners, instead of interpolating between them.

    Why this is better: bilinear interpolation pins each corner EXACTLY, so
    every millimetre of tap error is pushed into the squares in between, and
    the grid can come out as a rectangle or a parallelogram - shapes a real
    board cannot be. The fit instead finds the ideal square grid (one uniform
    pitch, one rotation) closest to all four taps, spreading the small tap
    errors evenly. Squares stay square and evenly spaced by construction.

    Returns (model, pitch, rot_deg, residuals) where model(u, v) -> (x, y, z)
    for file index u and rank index v (0..7).
    """
    P = np.array([[0, 0], [7, 0], [0, 7], [7, 7]], float)       # a1 h1 a8 h8
    Q = np.array([a1[:2], h1[:2], a8[:2], h8[:2]], float)
    pb, qb = P.mean(0), Q.mean(0)
    Pc, Qc = P - pb, Q - qb
    U, S, Vt = np.linalg.svd(Qc.T @ Pc)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt                                   # pure rotation
    scale = float((S * np.diag(D)).sum() / (Pc ** 2).sum())     # uniform pitch
    t = qb - scale * (R @ pb)

    # Z: least-squares PLANE through the four corner heights (the board may
    # sit slightly tilted; a plane models that honestly).
    A = np.array([[0, 0, 1], [7, 0, 1], [0, 7, 1], [7, 7, 1]], float)
    zc = np.array([a1[2], h1[2], a8[2], h8[2]], float)
    zco, *_ = np.linalg.lstsq(A, zc, rcond=None)

    def model(u, v):
        x, y = scale * (R @ np.array([float(u), float(v)])) + t
        z = zco[0] * u + zco[1] * v + zco[2]
        return float(x), float(y), float(z)

    resid = [float(np.linalg.norm(np.array(model(*P[i])[:2]) - Q[i]))
             for i in range(4)]
    rot = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return model, scale, rot, resid

def _corners_swapped(a1, h1, a8, h8):
    """
    True if a8/h8 were captured SWAPPED: then a1->a8 measures like a DIAGONAL
    (longer than a1->h8, which would be the true a-file edge).
    """
    return math.dist(a1[:2], a8[:2]) > math.dist(a1[:2], h8[:2]) * 1.15

def _startup_swap_check():
    """Detect and repair a swapped a8/h8 in the SAVED board map."""
    try:
        g = lambda sq: (board_config.SQUARE_MAP[sq]["p"]["x"],
                        board_config.SQUARE_MAP[sq]["p"]["y"],
                        board_config.SQUARE_MAP[sq]["p"]["z"])
        a1, h1, a8, h8 = g("a1"), g("h1"), g("a8"), g("h8")
    except Exception:
        return
    if not _corners_swapped(a1, h1, a8, h8):
        return
    print("=" * 60)
    print("[Cal] A8 and H8 in the saved calibration are SWAPPED (a1->a8 is a")
    print("      diagonal). Rebuilding the whole square grid corrected.")
    print("=" * 60)
    model, pitch, rot, resid = fit_square_grid(a1, h1, h8, a8)   # swapped back
    print(f"[Cal] refit: pitch {pitch:.2f}mm, worst corner {max(resid):.1f}mm")
    for r_i, rank in enumerate(RANKS):
        for f_i, file in enumerate(FILES):
            x, y, z = model(f_i, r_i)
            for pc in "prnbqk":
                board_config.SQUARE_MAP[file + rank][pc] = {"x": x, "y": y, "z": z}
    # persist the corrected map so tests and restarts see it too
    try:
        L = ["# corrected by chess_pi.py (a8/h8 were swapped)", "SQUARE_MAP = {"]
        for r_i, rank in enumerate(RANKS):
            for f_i, file in enumerate(FILES):
                e = board_config.SQUARE_MAP[file + rank]["p"]
                L.append(f'    "{file}{rank}": {{')
                for pc in "prnbqk":
                    L.append(f'        "{pc}": {{"x": {e["x"]:.1f}, "y": {e["y"]:.1f}, "z": {e["z"]:.1f}}},')
                L.append("    },")
        L.append("}")
        old = open("board_config.py").read()
        for ln in old.splitlines():
            if ln.startswith(("PARK", "GRAVEYARD_WHITE", "GRAVEYARD_BLACK",
                              "FLOOR_Z", "MIN_GRAB")):
                L.append(ln)
        L.append("")
        open("board_config.py", "w").write("\n".join(L))
        print("[Cal] corrected board_config.py written")
    except Exception as e:
        print(f"[Cal] corrected in memory only ({e})")

_startup_swap_check()
# the repair may have rewritten every square, so re-derive anything measured
# from the board map (harmless if nothing changed)
_recompute_geometry()

def arm_move_piece(from_sq, to_sq, piece_symbol):
    fx, fy, fz = coords(from_sq, piece_symbol)
    tx, ty, tz = coords(to_sq, piece_symbol)
    # print the CALIBRATED coordinates actually being used, so a mapping
    # problem is visible instead of silent
    print(f"  [ARM] {from_sq} ({fx:.1f},{fy:.1f},{fz:.1f}) -> "
          f"{to_sq} ({tx:.1f},{ty:.1f},{tz:.1f})")
    fx, fy, ok_f = reach_ok(fx, fy)
    tx, ty, ok_t = reach_ok(tx, ty)
    if not (ok_f and ok_t):
        print(f"  [ARM] refusing {from_sq}->{to_sq}: a square is out of reach - "
              f"please move that piece by hand")
        send("MSG:square out of reach - move that piece by hand")
        return
    if not pick(fx, fy, fz, from_sq):
        print(f"  [ARM] {from_sq}: pick failed - not placing anything")
        send("MSG:could not pick up the piece - check the board")
        return
    if not place(tx, ty, tz, to_sq):
        send(f"MSG:could not place on {to_sq} - put it there by hand")

def grave_spot(colour):
    return GRAVE_W if colour == "w" else GRAVE_B

_graves = {"w": 0, "b": 0}
def grave_slot(colour):
    # Every captured piece goes to the SAME calibrated drop point - they just
    # pile up there. The old per-slot 28 mm offsets made the drop spot march
    # further away with every capture, eventually leaving the arm's reach.
    return grave_spot(colour)

def arm_remove_piece(square, piece_symbol):
    colour = "w" if piece_symbol.isupper() else "b"
    px, py, pz = coords(square, piece_symbol)
    if grave_spot(colour) is None:
        print(f"  [ARM] no {colour} pile calibrated - leaving {square}")
        return
    gx, gy, gz = grave_slot(colour)
    gz = gz + 25.0          # release above the pile so stacked pieces don't jam
    print(f"  [ARM] capture {square} -> {colour} pile (piece #{_graves[colour]+1})")
    px, py, ok_p = reach_ok(px, py)
    gx, gy, ok_g = reach_ok(gx, gy)
    if not (ok_p and ok_g):
        print("  [ARM] capture square out of reach - remove that piece by hand")
        send("MSG:captured piece out of reach - remove it by hand")
        return
    if not pick(px, py, pz, square):
        print(f"  [ARM] {square}: pick failed - remove the piece by hand")
        send("MSG:could not pick up the captured piece")
        return
    if not place(gx, gy, gz, "pile"):
        send("MSG:could not reach the pile - take the piece off by hand")
    _graves[colour] += 1


# ================================================================ ENGINE
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
board = chess.Board()
human_is_white = True

# What limit engine_play uses - depth-capped for weak levels, timed otherwise.
ENGINE_LIMIT = {"limit": chess.engine.Limit(time=THINK_TIME)}

def set_difficulty(elo):
    """
    Stockfish's own UCI_Elo only goes down to ~1320. Below that, Skill Level
    ALONE is not enough (even Skill 0 with normal thinking time plays ~1300),
    so weak levels ALSO cap the search depth. The old mapping sent Elo 800 to
    Skill 5 - roughly 1700-1800 strength, which is why it felt far too strong.
    """
    opt = engine.options.get("UCI_Elo")
    lo = opt.min if opt else 1320
    hi = opt.max if opt else 3190
    if elo < lo:
        # depth 1-2 makes Stockfish essentially blind: it shuffles edge pawns
        # and repeats itself. Depth 3 is still very beatable but plays chess.
        if   elo <  800: skill, depth = 0, 3     # true beginner
        elif elo < 1000: skill, depth = 0, 4
        elif elo < 1100: skill, depth = 1, 4
        elif elo < 1200: skill, depth = 2, 5
        else:            skill, depth = 3, 6
        engine.configure({"UCI_LimitStrength": False, "Skill Level": skill})
        ENGINE_LIMIT["limit"] = chess.engine.Limit(depth=depth)
        print(f"[Engine] Elo {elo} -> Skill {skill}, depth {depth}")
    else:
        engine.configure({"UCI_LimitStrength": True,
                          "UCI_Elo": max(lo, min(hi, elo))})
        ENGINE_LIMIT["limit"] = chess.engine.Limit(time=THINK_TIME)
        print(f"[Engine] UCI_Elo {max(lo, min(hi, elo))}")

def victim_square(move):
    if board.is_en_passant(move):
        f = chess.square_file(move.to_square)
        r = chess.square_rank(move.from_square)
        return chess.square_name(chess.square(f, r))
    return chess.square_name(move.to_square)

def sync_screen():
    send("FEN:" + board.board_fen())

def report_status():
    if board.is_checkmate():
        send("STATUS:CHECKMATE")
    elif board.is_stalemate():
        send("STATUS:STALEMATE")
    elif board.is_insufficient_material():
        send("STATUS:DRAW")

# ---- camera orientation ----------------------------------------------
# The camera's four tapped corners decide which image corner is a1. If they
# were tapped starting from a different physical corner, EVERY square the
# camera reports is rotated or mirrored - the arm's map is fine, but the game
# state desyncs and the arm then moves squares that make no sense.
CAM_ORIENT = {"t": 0}          # 0 = as calibrated

def _remap_name(name, t):
    """Square name under transform t (0-3 rotations, 4-7 the mirrored set)."""
    fi, ri = ord(name[0]) - 97, int(name[1]) - 1
    if t >= 4:
        fi = 7 - fi
        t -= 4
    for _ in range(t):
        fi, ri = ri, 7 - fi
    return chr(97 + fi) + str(ri + 1)

def orient_snap(snap, t=None):
    """Re-key a camera snapshot into board coordinates."""
    t = CAM_ORIENT["t"] if t is None else t
    if t == 0 or not snap:
        return snap
    return {_remap_name(k, t): v for k, v in snap.items()}

def detect_orientation(snap, chessboard):
    """
    Work out which rotation/mirror of the camera view matches the game state,
    by scoring all eight against the pieces actually on the board.
    Returns (best_t, best_score, second_score) with scores out of 64.
    """
    human_colour = chess.WHITE if human_is_white else chess.BLACK
    expected = {}
    for sq_i in chess.SQUARES:
        pc = chessboard.piece_at(sq_i)
        expected[chess.square_name(sq_i)] = ("none" if pc is None else
                                             ("green" if pc.color == human_colour
                                              else "blue"))
    scores = []
    for t in range(8):
        mapped = orient_snap(snap, t)
        hit = sum(1 for n, exp in expected.items()
                  if piece_colour(mapped.get(n)) == exp)
        # -t so that on a TIE the lower transform wins: identity (0) is
        # preferred, then plain rotations before mirrored ones. The opening
        # position is left-right symmetric, so a mirror scores the same as a
        # rotation - never "correct" to a mirror on a tie.
        scores.append((hit, -t))
    scores.sort(reverse=True)
    return -scores[0][1], scores[0][0], scores[1][0]

def check_camera_orientation():
    """
    Compare the camera against the CURRENT game state and lock in the
    orientation that fits. Run this with the pieces in a known position.
    """
    snap = _baseline.get("snap")
    if not snap:
        print("[Orient] no photo yet - take one first")
        return
    t, best, second = detect_orientation(snap, board)
    names = {0: "as calibrated", 1: "rotated 90", 2: "rotated 180",
             3: "rotated 270", 4: "mirrored", 5: "mirrored + 90",
             6: "mirrored + 180", 7: "mirrored + 270"}
    print(f"[Orient] best match: {names[t]} ({best}/64 squares), "
          f"next best {second}/64")
    if best < 55:
        print("[Orient] nothing fits well - is the board set up as the game "
              "expects, and is the camera calibration current?")
        return
    if t == 0:
        print("[Orient] camera agrees with the board - no change needed")
    else:
        CAM_ORIENT["t"] = t
        print(f"[Orient] camera view is {names[t]} - CORRECTED from now on")
    return t

def board_vs_camera(chessboard, snap):
    """
    Compare the ENTIRE board as the camera sees it against the game state.
    Returns a list of (square, expected, seen) disagreements.
    Human pieces read GREEN, the arm's read BLUE, empty reads none.
    This is the safety net against a phantom detection: once a move that
    never happened is accepted, every later move is played on a board that
    does not exist.
    """
    snap = orient_snap(snap)
    if not snap:
        return []
    bad = []
    human_colour = chess.WHITE if human_is_white else chess.BLACK
    for sq_i in chess.SQUARES:
        name = chess.square_name(sq_i)
        pc = chessboard.piece_at(sq_i)
        if pc is None:
            expected = "none"
        else:
            expected = "green" if pc.color == human_colour else "blue"
        seen = piece_colour(snap.get(name))
        if expected != seen:
            bad.append((name, expected, seen))
    return bad

def state_is_sane(chessboard, snap, limit=1):
    """True if the camera broadly agrees with the game state."""
    bad = board_vs_camera(chessboard, snap)
    if len(bad) > limit:
        print(f"[State] board does NOT match the camera - {len(bad)} squares differ"
              f" (a whole move's worth is 2):")
        for name, exp, seen in bad[:8]:
            print(f"        {name}: expected {exp}, camera sees {seen}")
        return False
    return True

def move_looks_real(uci, snap, chessboard):
    """
    Confirm a DETECTED move against the camera on its own two squares:
    the origin must now be empty, and the destination must hold a human
    (green) piece. A phantom move fails one of these.
    """
    snap = orient_snap(snap)
    if not snap or not uci or len(uci) < 4:
        return False
    f, t = uci[:2], uci[2:4]
    src_seen = piece_colour(snap.get(f))
    dst_seen = piece_colour(snap.get(t))
    if src_seen == "green":
        print(f"[Auto] {uci} rejected: {f} still shows a green piece")
        return False
    if dst_seen != "green":
        print(f"[Auto] {uci} rejected: {t} does not show a green piece "
              f"(camera sees '{dst_seen}')")
        return False
    return True

def check_move_done(from_sq, to_sq, colour="blue"):
    """
    Look at the board AFTER the arm's move and confirm reality matches intent.
    Uses the fresh baseline snapshot (taken with the arm parked).
        "ok"        - origin empty, destination holds the piece
        "not_moved" - piece still sits on the origin (the grab missed)
        "lost"      - both squares empty: the piece was dropped somewhere
        "unknown"   - no snapshot to judge from
    """
    snap = orient_snap(_baseline.get("snap"))
    if not snap:
        return "unknown"
    src_state = piece_colour(snap.get(from_sq))
    dst_state = piece_colour(snap.get(to_sq))
    if dst_state == colour and src_state != colour:
        return "ok"
    if src_state == colour:
        return "not_moved"
    if src_state == "none" and dst_state == "none":
        return "lost"
    return "unknown"

_last_engine_move = {"uci": None, "fen": None}

def engine_play():
    """Engine picks a move, the arm plays it, then parks off-board."""
    # Refuse to touch the pieces if the camera and the game state disagree -
    # moving on a wrong board is how pieces end up in random places.
    if not state_is_sane(board, _baseline.get("snap")):
        print("[Engine] NOT MOVING - fix the board (or re-take a photo) first")
        send("MSG:board does not match the game - check the pieces")
        return
    auto_detect["arm_busy"] = True        # pause auto-detect while the arm moves
    print(f"[Engine] thinking from move {board.fullmove_number}, "
          f"{'white' if board.turn == chess.WHITE else 'black'} to play")
    print(f"[Engine] position: {board.board_fen()}")
    result = engine.play(board, ENGINE_LIMIT["limit"])
    move = result.move
    uci = move.uci()
    print(f"[Engine] plays {uci}")
    if uci == _last_engine_move["uci"] and board.board_fen() == _last_engine_move["fen"]:
        print("[Engine] SAME position and SAME move as last turn - the game is "
              "not advancing. Your move was never registered; check the "
              "[Auto]/[Cam] lines above for a rejected or missed detection.")
    _last_engine_move["uci"] = uci
    _last_engine_move["fen"] = board.board_fen()

    piece = board.piece_at(move.from_square)
    symbol = piece.symbol()
    from_sq = chess.square_name(move.from_square)
    to_sq   = chess.square_name(move.to_square)

    if board.is_capture(move):
        vsq = victim_square(move)
        victim = board.piece_at(chess.parse_square(vsq))
        arm_remove_piece(vsq, victim.symbol() if victim else "p")

    is_castle = board.is_castling(move)
    kingside  = board.is_kingside_castling(move)
    arm_move_piece(from_sq, to_sq, symbol)
    if is_castle:
        rank = from_sq[1]
        if kingside:
            arm_move_piece("h" + rank, "f" + rank, "r")
        else:
            arm_move_piece("a" + rank, "d" + rank, "r")

    board.push(move)
    go_park()                 # <-- clear the camera's view
    time.sleep(0.4)
    snapshot_baseline()       # new "before" with the arm parked

    # Camera check: REPORT what the board shows. No automatic re-move - a
    # repeated move is worse than a reported one.
    if not is_castle:
        state = check_move_done(from_sq, to_sq)
        if state == "ok":
            print(f"  [Check] {uci}: confirmed on the board")
        elif state == "not_moved":
            print(f"  [Check] {uci}: piece still on {from_sq} - the grab missed")
            send("MSG:arm missed the piece - check the board")
        elif state == "lost":
            print(f"  [Check] {uci}: {from_sq} and {to_sq} both empty - dropped")
            send("MSG:piece dropped - please place it by hand")

    auto_detect["arm_busy"] = False       # resume watching for the human
    send(uci)
    sync_screen()
    report_status()

def _rescue_castle(uci):
    """
    If the raw detection is illegal, see whether a legal castling move shares
    the same king-from square. Vision often misses the rook, so a castle can
    look like a short king move - python-chess knows which is real.
    """
    if not uci or len(uci) < 4:
        return None
    frm = uci[:2]
    for m in board.legal_moves:
        if board.is_castling(m) and chess.square_name(m.from_square) == frm:
            return m.uci()
    return None

def apply_human_move(uci):
    """The human already moved the piece by hand; we just record + reply."""
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        move = None
    if move is None or move not in board.legal_moves:
        rescue = _rescue_castle(uci)
        if rescue:
            print(f"[Human] {uci} -> castle {rescue}")
            uci = rescue
            move = chess.Move.from_uci(uci)
        else:
            print(f"[Human] {uci} illegal")
            send("ILLEGAL"); return False
    print(f"[Human] {uci}")
    board.push(move)
    sync_screen()
    report_status()
    if not board.is_game_over():
        engine_play()
    return True


# ================================================================ VISION
# --- barrel undistort ---
_ud = {"k1": 0.0, "m1": None, "m2": None, "shape": None}
def undistort(frame, k1):
    if abs(k1) < 1e-6:
        return frame
    h, w = frame.shape[:2]
    if _ud["m1"] is None or _ud["k1"] != k1 or _ud["shape"] != (h, w):
        fx = fy = w * 0.9
        K = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], np.float32)
        D = np.array([k1, 0, 0, 0], np.float32)
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_16SC2)
        _ud.update(k1=k1, m1=m1, m2=m2, shape=(h, w))
    return cv2.remap(frame, _ud["m1"], _ud["m2"], cv2.INTER_LINEAR)

def warp_board(frame, corners):
    dst = np.float32([[0, WARP_SIZE], [WARP_SIZE, WARP_SIZE], [WARP_SIZE, 0], [0, 0]])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))

def square_cells(warped, inset=INSET):
    s = WARP_SIZE // 8
    i = int(s * inset)
    cells = {}
    for r in range(8):
        for f in range(8):
            cells[FILES[f] + RANKS[7 - r]] = warped[r*s+i:(r+1)*s-i, f*s+i:(f+1)*s-i]
    return cells

def cell_signature(cell):
    c = cv2.resize(cell, SIG_SIZE, interpolation=cv2.INTER_AREA)
    c = cv2.GaussianBlur(c, (3, 3), 0)
    L, A, B = cv2.split(cv2.cvtColor(c, cv2.COLOR_BGR2Lab))
    Ln = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(L)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(cv2.magnitude(gx, gy) * EDGE_GAIN, 0, 255)
    return np.dstack([Ln, edge, A, B]).astype(np.float32)

def cell_score(a, b):
    d = np.abs(a - b)
    bright = float(d[:, :, 0].mean())
    edge   = float(d[:, :, 1].mean())
    colour = float(d[:, :, 2].mean() + d[:, :, 3].mean()) / 2.0
    return (W_BRIGHT*bright + W_EDGE*edge + W_COLOUR*colour) / (W_BRIGHT+W_EDGE+W_COLOUR)

def diff_squares(base, now, threshold=THRESHOLD, ratio=RATIO):
    scored = []
    for name in base:
        a, b = base[name], now[name]
        if a is None or b is None or a.shape != b.shape:
            continue
        sc = cell_score(a, b)
        if sc > threshold:
            scored.append((name, sc))
    scored.sort(key=lambda t: -t[1])
    if not scored:
        return [], []
    cut = scored[0][1] * ratio
    return [t for t in scored if t[1] >= cut], [t for t in scored if t[1] < cut]

def occupancy(sig):
    return 0.0 if sig is None else float(sig[:, :, 1].mean())


def colour_cover(sig):
    """
    How much of the square's CENTRE is covered by green / blue pixels.

    Counting pixels beats averaging: on a one-square move the piece lands
    right next to where it started, and a sliver of it bleeds into the old
    square's crop. An average barely shifts (so the move goes unseen), while
    a coverage FRACTION on a tight centre crop drops decisively.
    Returns (green_fraction, blue_fraction), each 0..1.
    """
    if sig is None:
        return 0.0, 0.0
    h, w = sig.shape[0], sig.shape[1]
    m0, m1 = int(h * 0.22), int(h * 0.78)          # centre ~56% only
    n0, n1 = int(w * 0.22), int(w * 0.78)
    A = sig[m0:m1, n0:n1, 2]
    B = sig[m0:m1, n0:n1, 3]
    total = float(A.size) or 1.0
    green = float((128.0 - A > 14).sum()) / total
    blue  = float((128.0 - B > 14).sum()) / total
    return green, blue

COVER_ON = 0.14      # this much of the centre coloured -> a piece is there

def piece_colour(sig):
    """"green", "blue" or "none" for one square, by centre coverage."""
    g, b = colour_cover(sig)
    if max(g, b) < COVER_ON:
        return "none"
    return "green" if g >= b else "blue"


def moved_colour(kept, base, now):
    """
    Which colour piece made this move? Look at the square that EMPTIED (the
    origin): whatever colour was there in the BASE frame is the mover.
    """
    if base is None or now is None:
        return "unknown"
    best, best_drop = None, 0.0
    for name, _ in kept:
        drop = occupancy(base.get(name)) - occupancy(now.get(name))
        if drop > best_drop:
            best, best_drop = name, drop
    if best is None:
        return "unknown"
    return piece_colour(base.get(best))

CASTLES = {
    frozenset(["e1", "g1", "h1", "f1"]): "O-O white",
    frozenset(["e1", "c1", "a1", "d1"]): "O-O-O white",
    frozenset(["e8", "g8", "h8", "f8"]): "O-O black",
    frozenset(["e8", "c8", "a8", "d8"]): "O-O-O black",
}

def classify_change(changed, base=None, now=None):
    names = [n for n, _ in changed]
    nset = set(names)
    if len(names) < 2:
        return {"kind": "none", "starts": [], "ends": list(names), "uci": None}

    for sqset, lbl in CASTLES.items():
        if nset == set(sqset):
            homes = ("e1", "e8", "a1", "a8", "h1", "h8")
            # king start is always e1/e8; king end is g/c file
            kstart = "e1" if "e1" in sqset else "e8"
            kend   = next(s for s in sqset if s[0] in "gc")
            return {"kind": "castle", "label": lbl,
                    "starts": [s for s in sqset if s in homes],
                    "ends":   [s for s in sqset if s not in homes],
                    "uci": kstart + kend}

    if len(names) == 3:
        files = sorted(set(n[0] for n in names))
        ranks = sorted(set(n[1] for n in names))
        if len(files) == 2 and len(ranks) == 2:
            starts, ends = [], []
            for n in names:
                if base is not None and now is not None:
                    (starts if occupancy(base.get(n)) > occupancy(now.get(n)) else ends).append(n)
            if not starts:
                starts, ends = names[:1], names[1:]
            return {"kind": "enpassant", "starts": starts, "ends": ends,
                    "uci": (starts[0] + ends[0]) if starts and ends else None}

    a, b = names[0], names[1]
    if base is not None and now is not None:
        if (occupancy(base.get(b)) - occupancy(now.get(b))) > (occupancy(base.get(a)) - occupancy(now.get(a))):
            a, b = b, a
    return {"kind": "move", "starts": [a], "ends": [b], "uci": a + b}


def detect_move_by_colour(base, now, chessboard):
    """
    Detect the human's move from per-square COLOUR STATE changes, validated
    against the position's legal moves. Far more robust than raw image diff:
    the pieces are strongly green/blue, and chess rules filter out phantoms
    (shadows, small shifts, the arm's edge in frame).
        origin  : a square that WAS green and is no longer green
        dest    : a square that IS now green and wasn't before
    Castling shows two of each - resolved by matching against legal moves.
    Returns a uci string or None.
    """
    if base is None or now is None or chessboard is None:
        return None
    lost, gained = [], []
    for sq in base:
        b_st = piece_colour(base.get(sq))
        n_st = piece_colour(now.get(sq))
        if b_st == n_st:
            continue
        if b_st == "green" and n_st != "green":
            lost.append(sq)
        if n_st == "green" and b_st != "green":
            gained.append(sq)
    if len(lost) > 3 or len(gained) > 3:
        return None            # scene-wide change (lighting) - not a move

    if not lost or not gained:
        # No square flipped state outright - normal for a ONE-SQUARE move,
        # where the piece still clips the crop it came from. Fall back to the
        # biggest green coverage DROP and RISE, and let chess legality decide.
        drops, rises = {}, {}
        for sq in base:
            gb, _ = colour_cover(base.get(sq))
            gn, _ = colour_cover(now.get(sq))
            if gb - gn > 0.06:
                drops[sq] = gb - gn
            if gn - gb > 0.06:
                rises[sq] = gn - gb
        best = None
        for mv in chessboard.legal_moves:
            f = chess.square_name(mv.from_square)
            t = chess.square_name(mv.to_square)
            if f in drops and t in rises:
                score = drops[f] + rises[t]
                if best is None or score > best[0]:
                    best = (score, mv.uci())
        if best:
            print(f"[Detect] faint one-square change -> {best[1]}")
            return best[1]
        return None

    changed = set(lost) | set(gained)
    # Rank every matching legal move by how many changed squares it EXPLAINS.
    # This matters for castling: 4 squares change, and a plain rook/king move
    # (h1g1, e1d1) also matches 2 of them - the castle, explaining all 4,
    # must win.
    best = None            # (explained_count, uci_or_partial)
    for mv in chessboard.legal_moves:
        f, t = chess.square_name(mv.from_square), chess.square_name(mv.to_square)
        if f not in lost or t not in gained:
            continue
        if chessboard.is_castling(mv):
            rank = f[1]
            if chessboard.is_kingside_castling(mv):
                r_from, r_to = "h" + rank, "f" + rank
            else:
                r_from, r_to = "a" + rank, "d" + rank
            if r_from in lost and r_to in gained:
                cand = (4, mv.uci())          # full castle: explains all 4
            else:
                # king moved, rook not yet - two hand motions. Report partial
                # so the caller WAITS with the same baseline for the rook.
                cand = (3, "partial")
        else:
            cand = (len({f, t} & changed), mv.uci())
        if best is None or cand[0] > best[0]:
            best = cand
    return best[1] if best else None


# --- camera (USB webcam via OpenCV) ---
# The camera is found automatically: any /dev/video* whose card name is NOT
# one of the Pi's own internal video subsystems is treated as the USB webcam.
# Works with any UVC camera - no name to edit when you swap cameras.
PI_INTERNAL = ("rp1-cfe", "pispbe", "rpi-hevc", "bcm2835", "unicam")
CAM_INDEX   = 0            # fallback only, if nothing identifiable is found

def _find_camera_index():
    """Return the lowest /dev/videoN index that is a real USB camera."""
    import glob, subprocess
    for dev in sorted(glob.glob("/dev/video*"),
                      key=lambda d: int(d.replace("/dev/video", ""))):
        try:
            out = subprocess.run(["v4l2-ctl", "-d", dev, "--info"],
                                 capture_output=True, text=True, timeout=3).stdout
        except Exception:
            continue
        low = out.lower()
        if "card type" not in low:
            continue
        if any(tag in low for tag in PI_INTERNAL):
            continue
        name = next((ln.split(":", 1)[1].strip() for ln in out.splitlines()
                     if "Card type" in ln), "?")
        idx = int(dev.replace("/dev/video", ""))
        print(f"[Cam] found USB camera '{name}' at /dev/video{idx}")
        return idx
    print(f"[Cam] no USB camera identified; falling back to index {CAM_INDEX}")
    return CAM_INDEX

class PiCam:
    """Any USB webcam through OpenCV, auto-located (see _find_camera_index)."""
    def __init__(self):
        idx = _find_camera_index()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():                 # fall back to default backend
            self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open USB camera (index {idx})")
        # MJPG gives full quality/framerate over USB on most UVC cameras.
        # Try 1080p, then step down until the camera actually delivers frames.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except Exception:
            pass
        for w, h in ((1920, 1080), (1280, 720), (640, 480)):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            time.sleep(0.3)
            ok, _ = self.cap.read()
            if ok:
                aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[Cam] running at {aw}x{ah}")
                break
        else:
            raise RuntimeError("camera opened but delivers no frames at any size")
        time.sleep(0.3)
    def read(self):
        ok, frame = self.cap.read()                 # already BGR from OpenCV
        return ok, frame
    def release(self):
        self.cap.release()


# --- vision calibration state (loaded / saved to CAM_CALIB) ---
vision = {"corners": None, "k1": 0.0}   # sensible barrel default

def load_vision():
    if os.path.exists(CAM_CALIB):
        try:
            d = json.load(open(CAM_CALIB))
            if isinstance(d, dict):
                vision["corners"] = np.float32(d["corners"])
                vision["k1"] = float(d.get("k1", 0.0))
            else:
                vision["corners"] = np.float32(d)
            print(f"[Vision] loaded corners from {CAM_CALIB}")
        except Exception as e:
            print(f"[Vision] could not read {CAM_CALIB}: {e}")

def save_vision():
    json.dump({"corners": [[float(x), float(y)] for x, y in vision["corners"]],
               "k1": float(vision["k1"])}, open(CAM_CALIB, "w"), indent=2)
    print(f"[Vision] saved {CAM_CALIB}")

load_vision()


# ================================================================ CAMERA THREAD
# The camera preview and all OpenCV windows live in ONE thread. The serial loop
# asks it to do things (calibrate, snapshot a baseline, detect a move) through
# these simple flags, and reads the result back.
cam_cmd = {"action": None, "result": None, "done": threading.Event()}
_baseline = {"snap": None}

# Automatic detection: when "on", the camera watches the board and appends any
# detected human move (as UCI) to "queue". The main loop drains the queue and
# plays each move - no END TURN button needed.
auto_detect = {"on": False, "queue": [], "arm_busy": False}

def _request(action, timeout=120):
    cam_cmd["result"] = None
    cam_cmd["done"].clear()
    cam_cmd["action"] = action
    if not cam_cmd["done"].wait(timeout):
        print(f"[Cam] '{action}' timed out")
        return None
    return cam_cmd["result"]

def snapshot_baseline():
    """Store the current board as the 'before' position."""
    _request("baseline")

def detect_move():
    """Compare against the baseline, return a UCI string or None."""
    return _request("detect")

# Camera calibration now happens ENTIRELY on the touchscreen (the Pi is
# headless). The Pi ships a small snapshot to the ESP32, which displays it and
# sends back the tapped corners. Image is 160x160 RGB565 = 51200 bytes, ~4.5s
# over 115200 baud - fine for a one-off calibration frame.
# 16:9 calibration image (the C920's native aspect). Sent at CAL_W x CAL_H,
# shown on the screen at x3. 216x122 -> 648x366 on screen: big and correct.
CAL_W = 176
CAL_H = 99

K1_DEFAULT = 0.0
_cam_cal = {"k1": K1_DEFAULT, "corners": [], "frame": None}

def _send_snapshot():
    """
    Grab a frame, undistort, shrink, send as RGB565 with a handshake:
      Pi   -> IMG_START:w:h
      ESP  -> IMG_READY     (it has parsed the header and is now reading bytes)
      Pi   -> <w*h*2 raw bytes>
    The ack removes the race where bytes arrive before the ESP switches to
    binary-read mode.
    """
    # wait up to 6s for the camera to have produced a frame
    for _ in range(120):
        if _cam_cal["frame"] is not None:
            break
        time.sleep(0.05)
    f = _cam_cal["frame"]
    if f is None:
        if cam_status["error"]:
            print(f"[Cam] camera failed to start: {cam_status['error']}")
            send("CAM_ERR:camera error")
        else:
            print("[Cam] camera still warming up - wait a few seconds and retry")
            send("CAM_ERR:warming up, retry")
        return

    img = undistort(f, _cam_cal["k1"])

    # Send the image at the camera's real aspect ratio (the C920 is 16:9), so
    # nothing is squished and no resolution is wasted on letterbox bars. The
    # ESP shows it at the same aspect. CAL_W x CAL_H are the sent dimensions.
    fh, fw = img.shape[:2]
    small = cv2.resize(img, (CAL_W, CAL_H), interpolation=cv2.INTER_AREA)
    _cam_cal["src"] = (fw, fh)          # real camera size, for tap-mapping

    # a mild unsharp mask so the board squares read crisply after scaling up
    blur = cv2.GaussianBlur(small, (0, 0), 1.0)
    small = cv2.addWeighted(small, 1.5, blur, -0.5, 0)
    b = small[:, :, 0].astype(np.uint16)
    g = small[:, :, 1].astype(np.uint16)
    r = small[:, :, 2].astype(np.uint16)
    rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    payload = rgb565.astype(">u2").tobytes()

    esp.reset_input_buffer()
    send(f"IMG_START:{CAL_W}:{CAL_H}")

    # wait for the ESP to say it's ready for the raw bytes
    t0 = time.time()
    ack = b""
    while time.time() - t0 < 2.0:
        ack += esp.read(64)
        if b"IMG_READY" in ack:
            break
    if b"IMG_READY" not in ack:
        print("[Cam] no IMG_READY ack - sending anyway")

    esp.write(payload)
    esp.flush()
    time.sleep(0.02)
    send(f"CAM_K1:{_cam_cal['k1']:+.2f}")      # so the screen can display it
    print(f"[Cam] sent {len(payload)} bytes (k1={_cam_cal['k1']:+.2f})")


cam_status = {"ready": False, "error": None}

def _avg_signatures(snaps):
    """Average several per-square signature dicts into one stable signature."""
    if not snaps:
        return None
    out = {}
    for name in snaps[0]:
        stack = [s[name] for s in snaps if s.get(name) is not None]
        out[name] = np.mean(stack, axis=0) if stack else None
    return out

def camera_thread():
    try:
        cap = PiCam()
    except Exception as e:
        import traceback
        cam_status["error"] = str(e)
        print(f"[Cam] could not start: {e}")
        traceback.print_exc()
        return

    print("=" * 50)
    print("[Cam] CAMERA READY")
    print("=" * 50)
    cam_status["ready"] = True

    # --- automatic detection state machine ---
    # STABLE   : board quiet, this is a trusted position
    # BUSY     : lots of motion (a hand) - do NOT compare
    # SETTLING : motion stopped, counting calm frames before we trust it
    phase = "stable"
    prev_gray = None
    quiet = 0
    recent = []                      # rolling stable frames, averaged for noise
    ACT_ON  = 0.020                  # >2% pixels moving = hand present
    ACT_OFF = 0.005                  # <0.5% = calm
    QUIET_NEED = 6                   # calm frames before trusting the board
    AVG_FRAMES = 4                   # frames averaged into a signature

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue
        frame = undistort(frame, vision["k1"])
        _cam_cal["frame"] = frame
        act = cam_cmd["action"]

        if vision["corners"] is None:
            time.sleep(0.03)
            continue

        warped = warp_board(frame, vision["corners"])

        # ---- manual requests still work (baseline / detect via END_TURN) ----
        if act == "baseline":
            recent = []
            for _ in range(AVG_FRAMES):
                ok2, f2 = cap.read()
                if ok2:
                    w2 = warp_board(undistort(f2, vision["k1"]), vision["corners"])
                    recent.append({n: cell_signature(c)
                                   for n, c in square_cells(w2).items()})
                time.sleep(0.03)
            _baseline["snap"] = _avg_signatures(recent) or {
                n: cell_signature(c) for n, c in square_cells(warped).items()}
            phase = "stable"; prev_gray = None
            cam_cmd["result"] = True
            cam_cmd["action"] = None
            cam_cmd["done"].set()
            print("[Cam] baseline captured (averaged)")

        elif act == "detect":
            now = {n: cell_signature(c) for n, c in square_cells(warped).items()}
            uci = None
            if _baseline["snap"] is not None:
                uci = detect_move_by_colour(
                    orient_snap(_baseline["snap"]), orient_snap(now), board)
                if uci == "partial":
                    print("[Cam] castle half-done - move the rook too, then END TURN again")
                    uci = None
                elif uci:
                    print(f"[Cam] colour-detect: {uci}")
                else:
                    kept, _ = diff_squares(_baseline["snap"], now)
                    info = classify_change(kept, _baseline["snap"], now)
                    uci = info["uci"]
                    colour = moved_colour(kept, _baseline["snap"], now)
                    if uci and colour == "blue":
                        print(f"[Cam] ignoring BLUE move {uci} (arm's piece)")
                        uci = None
                    elif uci:
                        print(f"[Cam] diff-detect {info['kind']}: {uci} ({colour})")
            _baseline["snap"] = now
            cam_cmd["result"] = uci
            cam_cmd["action"] = None
            cam_cmd["done"].set()

        # ---- AUTOMATIC detection (runs when armed and no manual request) ----
        elif (auto_detect["on"] and not auto_detect["arm_busy"]
              and _baseline["snap"] is not None):
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev_gray is not None:
                activity = float((cv2.absdiff(prev_gray, gray) > 25).mean())

                if phase == "stable":
                    if activity > ACT_ON:
                        phase = "busy"
                        print("[Auto] motion - waiting for board to clear")
                elif phase == "busy":
                    if activity < ACT_OFF:
                        phase = "settling"; quiet = 0; recent = []
                elif phase == "settling":
                    if activity < ACT_OFF:
                        quiet += 1
                        # collect stable frames to average
                        recent.append({n: cell_signature(c)
                                       for n, c in square_cells(warped).items()})
                        if len(recent) > AVG_FRAMES:
                            recent.pop(0)
                        if quiet >= QUIET_NEED:
                            now = _avg_signatures(recent)
                            # 1st choice: colour-state change validated against
                            # the LEGAL MOVES of the position
                            uci = detect_move_by_colour(
                                orient_snap(_baseline["snap"]), orient_snap(now), board)
                            if uci == "partial":
                                # king moved, rook not yet - wait for the rook
                                # with the SAME baseline so the full castle is
                                # seen as one change
                                print("[Auto] castle in progress - waiting for the rook")
                                phase = "stable"
                                continue
                            if uci:
                                # sanity gate: play the move on a COPY and check
                                # the camera really shows that position. A
                                # phantom move fails this and is discarded.
                                trial = board.copy()
                                try:
                                    trial.push_uci(uci)
                                except Exception:
                                    trial = None
                                if (not move_looks_real(uci, orient_snap(now), board)
                                        or (trial is not None
                                            and not state_is_sane(trial, orient_snap(now), limit=2))):
                                    print(f"[Auto] REJECTED {uci} - the board does "
                                          f"not look like that. Nothing played.")
                                    uci = None
                                else:
                                    print(f"[Auto] colour-detect: {uci}")
                                    auto_detect["queue"].append(uci)
                            else:
                                # fallback: old image-diff classifier
                                kept, _ = diff_squares(_baseline["snap"], now)
                                info = classify_change(kept, _baseline["snap"], now)
                                uci = info["uci"]
                                colour = moved_colour(kept, _baseline["snap"], now)
                                if uci and colour == "blue":
                                    print(f"[Auto] ignoring BLUE move {uci}")
                                elif uci:
                                    print(f"[Auto] diff-detect {info['kind']}: {uci} ({colour})")
                                    auto_detect["queue"].append(uci)
                                else:
                                    print("[Auto] settled but no clear move")
                            _baseline["snap"] = now
                            phase = "stable"
                    else:
                        phase = "busy"        # moved again, keep waiting
            prev_gray = gray

        time.sleep(0.03)


# ================================================================ ARM CALIBRATION
# The screen sends CAL_CAPTURE:<square>; we read the arm's live pose. Corners a1
# h1 a8 h8 build the SQUARE_MAP by best-fitting a square grid to them
# (fit_square_grid). Extra named spots: "gw"/"gb" = capture piles, "park" =
# the high park spot, "floor" = the captured-piece table surface.
_cal = {}
_cal_changed = set()          # squares captured THIS session (from CAL_CHANGED)
CAL_SQUARES = ["a1", "h1", "a8", "h8"]

def read_pose():
    for i in range(6):
        try:
            p = arm.pose()
            if p is not None:
                return p[0], p[1], p[2]
        except (AttributeError, TypeError):
            pass
        print(f"  [Cal] arm didn't answer, retry {i+1}/6")
        time.sleep(0.4)
    raise RuntimeError("could not read arm pose")

def _seed_cal_from_config():
    """
    Pre-load the four corners (and off/gw/gb) from the current board_config so
    the user can re-capture just ONE spot and still SAVE - the untouched ones
    come from what's already saved.
    """
    try:
        smap = board_config.SQUARE_MAP
        for c in CAL_SQUARES:
            e = smap[c]["p"]
            _cal[c] = (e["x"], e["y"], e["z"])
        if getattr(board_config, "GRAVEYARD_WHITE", None):
            _cal["gw"] = tuple(board_config.GRAVEYARD_WHITE)
        if getattr(board_config, "GRAVEYARD_BLACK", None):
            _cal["gb"] = tuple(board_config.GRAVEYARD_BLACK)
        if getattr(board_config, "PARK", None):
            _cal["park"] = tuple(board_config.PARK)
        if getattr(board_config, "FLOOR_Z", None) is not None:
            _cal["floor"] = (0.0, 0.0, board_config.FLOOR_Z)
        print(f"[Cal] pre-loaded {list(_cal)} from board_config")
    except Exception as e:
        print(f"[Cal] nothing to pre-load ({e}) - fresh calibration")


def cal_capture(square):
    square = square.lower()
    x, y, z = read_pose()
    if square != "floor":
        for other, (ox, oy, _) in _cal.items():
            if other not in (square, "floor") and math.dist((x, y), (ox, oy)) < 10.0:
                send(f"CAL_ERR:{square} same spot as {other}")
                return
    _cal[square] = (x, y, z)
    print(f"[Cal] {square} = ({x:.1f}, {y:.1f}, {z:.1f})")
    send(f"CAL_OK:{square}:{x:.1f},{y:.1f},{z:.1f}")

def cal_save():
    have_corners = all(c in _cal for c in CAL_SQUARES)

    # If all four corners are present, (re)build the full board map from them.
    # If not, we only update the extras (park / graveyards) and KEEP whatever
    # SQUARE_MAP is already saved - so you can set HIGH PARK on its own.
    L = ["# generated by chess_pi.py on-screen calibration"]

    if have_corners:
        if _corners_swapped(_cal["a1"], _cal["h1"], _cal["a8"], _cal["h8"]):
            print("[Cal] a8/h8 tapped SWAPPED - correcting automatically")
            _cal["a8"], _cal["h8"] = _cal["h8"], _cal["a8"]
        d = lambda a, b: math.dist(_cal[a][:2], _cal[b][:2])
        # Edge consistency is a WARNING, never a blocker: hand-jogged captures
        # are rarely repeatable to a few mm, and the grid interpolation still
        # works from slightly inconsistent corners. The numbers tell you how
        # trustworthy the capture is.
        e1 = abs(d("a1","h1")-d("a8","h8")); e2 = abs(d("a1","a8")-d("h1","h8"))
        # SQUARENESS: a real board's four corner-square centres form a SQUARE,
        # so the rank edges and the file edges must ALSO match each other.
        # Comparing only opposite edges lets a rectangle through unnoticed.
        rank_edge = (d("a1","h1") + d("a8","h8")) / 2.0
        file_edge = (d("a1","a8") + d("h1","h8")) / 2.0
        if abs(rank_edge - file_edge) > 8:
            pitch = rank_edge / 7.0
            print("=" * 60)
            print(f"[Cal] NOT SQUARE: rank edges {rank_edge:.0f}mm but file edges "
                  f"{file_edge:.0f}mm.")
            print(f"[Cal] A real board is square. The difference is "
                  f"{abs(rank_edge-file_edge):.0f}mm ~ {abs(rank_edge-file_edge)/pitch:.1f} squares,")
            print( "[Cal] so one PAIR of corners was tapped on the wrong squares -")
            print( "[Cal] tap the CENTRE of the corner SQUARE (same spot on the")
            print( "[Cal] piece every time), not the board's outer edge.")
            print("=" * 60)
        worst = max(e1, e2)
        if worst > 6:
            print(f"[Cal] WARNING: opposite edges differ by {worst:.1f}mm "
                  f"(rank edges {e1:.1f}, file edges {e2:.1f}).")
            print("[Cal] Saved anyway - but if grabs land off-centre, re-tap the")
            print("[Cal] corners keeping the SAME claw reference on every one.")
        model, pitch, rot, resid = fit_square_grid(
            _cal["a1"], _cal["h1"], _cal["a8"], _cal["h8"])
        print(f"[Cal] best-fit grid: pitch {pitch:.2f}mm/square "
              f"(board {pitch*7:.0f}mm), rotation {rot:.1f}deg")
        print("[Cal] corner residuals: "
              + ", ".join(f"{r:.1f}" for r in resid)
              + f" mm (worst {max(resid):.1f})")
        if max(resid) > 6:
            print("[Cal] WARNING: a corner is >6mm from the best-fit square -")
            print("[Cal] re-tap using the SAME reference point on every corner.")
        L.append("SQUARE_MAP = {")
        for r, rank in enumerate(RANKS):
            for f, file in enumerate(FILES):
                x, y, z = model(f, r)
                L.append(f'    "{file}{rank}": {{')
                for pc in "prnbqk":
                    L.append(f'        "{pc}": {{"x": {x:.1f}, "y": {y:.1f}, "z": {z:.1f}}},')
                L.append("    },")
        L.append("}")
    else:
        # Park-only (or extras-only) save: keep the ENTIRE existing file but
        # strip the old PARK / GRAVEYARD lines, then re-append fresh ones below.
        try:
            existing = open("board_config.py").read()
        except FileNotFoundError:
            send("CAL_ERR:no saved board - capture the 4 corners first"); return
        kept = [ln for ln in existing.splitlines()
                if not ln.startswith(("PARK", "GRAVEYARD_WHITE",
                                      "GRAVEYARD_BLACK", "FLOOR_Z"))]
        L = kept

    # extras: only write ones we have (freshly captured or pre-seeded)
    if "gw" in _cal:
        L.append(f"GRAVEYARD_WHITE = {tuple(round(v,1) for v in _cal['gw'])}")
    if "gb" in _cal:
        L.append(f"GRAVEYARD_BLACK = {tuple(round(v,1) for v in _cal['gb'])}")
    if "park" in _cal:
        L.append(f"PARK = {tuple(round(v,1) for v in _cal['park'])}")
    if "floor" in _cal:
        L.append(f"FLOOR_Z = {round(_cal['floor'][2], 1)}")
    L.append("")
    open("board_config.py", "w").write("\n".join(L))
    print(f"[Cal] saved. park={_cal.get('park')}  corners_rebuilt={have_corners}")

    import importlib
    importlib.reload(board_config)
    global GRAVE_W, GRAVE_B, PARK, FLOOR_Z
    FLOOR_Z = getattr(board_config, "FLOOR_Z", None)
    GRAVE_W = getattr(board_config, "GRAVEYARD_WHITE", None)
    GRAVE_B = getattr(board_config, "GRAVEYARD_BLACK", None)
    PARK = getattr(board_config, "PARK", PARK)
    _recompute_geometry()      # travel lane + pitch from the NEW board map
    if FLOOR_Z is not None:
        print(f"[Cal] floor at Z={FLOOR_Z:.1f}, travel lane >= {FLOOR_Z + FLOOR_MARGIN:.1f}")
    print(f"[Cal] board_config.py written, travel lane {TRAVEL_Z:.1f}")
    send("CAL_SAVED")


# ================================================================ MAIN LOOP
def gripper_test():
    """Open and close the gripper so you can confirm it physically works."""
    print("\n[GripTest] testing the gripper - watch/listen for the pump...")
    try:
        print("[GripTest] CLOSING (pump on)...")
        control_claw(True)
        time.sleep(0.5)
        print("[GripTest] OPENING (pump off)...")
        control_claw(False)
        print("[GripTest] done. Did the claw close then open, and the pump run?")
        print("           If nothing happened: check the air tube is connected")
        print("           to the pump box and the pump box has power.\n")
    except Exception as e:
        import traceback
        print(f"[GripTest] FAILED: {e}")
        traceback.print_exc()

def arm_selftest():
    """
    Prove the arm moves + the gripper works, using the game's own motion
    primitive. (It used to call move_z/move_xy, which this file never defined,
    so the self-test always died on a NameError before touching the arm.)
    """
    print("\n[SelfTest] moving the arm through a small square...")
    try:
        px, py, _pz = PARK
        pos = _read_pose()
        if pos:
            _last[0], _last[1], _last[2] = pos
            if pos[2] < TRAVEL_Z:      # straight up first, never across the board
                _goto(pos[0], pos[1], TRAVEL_Z, "selftest: into the travel lane")
        _goto(px, py, TRAVEL_Z, "selftest: over the park spot")
        _goto(px, py - 30.0, TRAVEL_Z, "selftest: sideways")
        _goto(px, py - 30.0, TRAVEL_Z - 20.0, "selftest: down a little")
        _goto(px, py, TRAVEL_Z, "selftest: back")
        print("[SelfTest] motion DONE.")
    except Exception as e:
        import traceback
        print(f"[SelfTest] arm move FAILED: {e}")
        traceback.print_exc()
    gripper_test()

def main():
    global human_is_white

    threading.Thread(target=camera_thread, daemon=True).start()

    if SELFTEST_AT_START:          # off by default - see the CONFIG section
        arm_selftest()
    else:
        print("[Host] skipping the startup self-test (SELFTEST_AT_START=False)."
              " Press ARM TEST on the screen to run it.")

    print("\n[Host] ready. Drive everything from the touchscreen.\n")
    go_park()

    esp.timeout = 0.2                 # short read so we can also poll auto-detect

    while True:
        # ---- play any move the camera auto-detected ----
        if auto_detect["on"] and auto_detect["queue"] and not auto_detect["arm_busy"]:
            uci = auto_detect["queue"].pop(0)
            print(f"[Auto] playing detected move {uci}")
            if not board.is_game_over():
                apply_human_move(uci)      # records human move + triggers engine
                                           # reply, which re-baselines on park

        line = esp.readline().decode(errors="ignore").strip()
        if not line:
            continue
        print(f"<- ESP: {line}")

        try:
            # ---- arm calibration ----
            if line == "CAL_ENTER":
                _cal.clear()
                _cal_changed.clear()
                _seed_cal_from_config()
            elif line.startswith("CAL_CAPTURE:"):
                cal_capture(line.split(":")[1].strip())
            elif line.startswith("CAL_CHANGED:"):
                _cal_changed.clear()
                for sq in line.split(":", 1)[1].split(","):
                    if sq.strip():
                        _cal_changed.add(sq.strip())
            elif line == "CAL_SAVE":
                cal_save()
            elif line == "CAL_EXIT":
                pass

            # ---- camera calibration, entirely on the touchscreen ----
            elif line == "CAM_CAL":
                # start fresh from the default so it can't get stuck on a saved
                # value, and clear any old corner taps
                _cam_cal["k1"] = K1_DEFAULT
                _cam_cal["corners"] = []
                time.sleep(0.3)
                _send_snapshot()

            elif line == "CAM_K1UP":
                _cam_cal["k1"] += 0.02
                _send_snapshot()

            elif line == "CAM_K1DN":
                _cam_cal["k1"] -= 0.02
                _send_snapshot()

            elif line == "CAM_REFRESH":
                _send_snapshot()

            elif line.startswith("CAM_CORNER:"):
                # "CAM_CORNER:ix,iy" in 0..CAL_IMG image pixels
                xy = line.split(":")[1].split(",")
                ix, iy = int(xy[0]), int(xy[1])
                _cam_cal["corners"].append((ix, iy))
                print(f"[Cam] corner {len(_cam_cal['corners'])}: ({ix},{iy})")

            elif line == "CAM_UNDO":
                if _cam_cal["corners"]:
                    _cam_cal["corners"].pop()

            elif line == "CAM_SAVE":
                if len(_cam_cal["corners"]) != 4:
                    send("CAM_ERR:need 4 corners")
                else:
                    # tapped pixel (ix,iy) is in the CAL_W x CAL_H image; scale
                    # straight back to real camera-frame pixels
                    fw, fh = _cam_cal.get("src", (_cam_cal["frame"].shape[1],
                                                  _cam_cal["frame"].shape[0]))
                    pts_cam = [[ix / CAL_W * fw, iy / CAL_H * fh]
                               for ix, iy in _cam_cal["corners"]]
                    pts = np.float32(pts_cam)
                    vision["corners"] = pts
                    vision["k1"] = _cam_cal["k1"]
                    save_vision()
                    print(f"[Cam] saved corners {pts.tolist()} k1={vision['k1']:+.2f}")
                    send("CAM_DONE")

            elif line == "CAM_CANCEL":
                send("CAM_DONE")

            # ---- game setup ----
            elif line.startswith("SET_ELO:"):
                set_difficulty(int(line.split(":")[1]))
            elif line.startswith("SET_COLOR:"):
                human_is_white = (line.split(":")[1] == "WHITE")
            elif line == "START_MATCH":
                board.reset()
                _graves["w"] = _graves["b"] = 0
                go_park()
                sync_screen()
                snapshot_baseline()          # remember the start position
                auto_detect["queue"].clear()
                auto_detect["on"] = True      # <-- camera now watches automatically
                print("[Auto] automatic board detection ON")
                if not human_is_white:
                    engine_play()
                    snapshot_baseline()

            # ---- the human finished their move ----
            elif line == "END_TURN":
                send("READING")
                uci = detect_move()
                if uci is None:
                    send("NO_MOVE")
                    print("[Turn] camera saw no clear move")
                else:
                    ok = apply_human_move(uci)
                    if not ok:
                        # rejected - resync the screen and let them retry
                        snapshot_baseline()
                    else:
                        snapshot_baseline()  # arm has parked; new 'before'

            elif line == "ARM_TEST":
                arm_selftest()

            elif line == "RESET":
                board.reset()
                auto_detect["on"] = False
                auto_detect["queue"].clear()
                go_park()

            # a typed move still works as a fallback
            elif len(line) >= 4 and line[0] in "abcdefgh":
                if apply_human_move(line):
                    snapshot_baseline()

        except Exception as e:
            print(f"[Error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        control_claw(False)
        try:
            arm.close()
        except Exception:
            pass
        engine.quit()
        esp.close()
