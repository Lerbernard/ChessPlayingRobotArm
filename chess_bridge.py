"""
Host bridge: ESP32 touchscreen  <->  Stockfish  <->  Dobot Magician

THE ARM ONLY PLAYS THE ENGINE'S SIDE:
  - human taps their move on the touchscreen and moves the piece by hand
  - engine replies -> the arm physically makes that move

Capture flow (engine captures one of yours):
  1. Host sends  REMOVE:<square>  to the screen.
  2. Screen shows a modal and blocks input.
  3. Human lifts the captured piece off the board, taps DONE.
  4. Screen sends  REMOVED  -> only then does the arm run the move.

Run this FIRST, wait for "Host ready", then tap through the screen.
"""

VERSION = "chess_bridge v12  (axis-locked motion, no diagonals ever)"

import math
import time
import serial
import chess
import chess.engine
from pydobot import Dobot

import board_config  # your calibrated SQUARE_MAP

# ---------------- CONFIG ----------------
ESP_PORT   = "COM13"   # ESP32 display  (CH340)
DOBOT_PORT = "COM4"    # Dobot Magician (CP210x)
STOCKFISH_PATH = r"stockfish\stockfish-windows-x86-64-avx2.exe"

CLEARANCE  = 30.0               # mm the claw MUST rise above a piece before any
                                # sideways motion happens. 25-30 is the range.
TRAVEL_Z_FLOOR = 45.0           # the travel lane is never lower than this
SETTLE     = 0.35               # pause after an XY move so the arm stops swaying
MAX_REACH  = 320.0              # Dobot Magician's usable radius, mm from the base
MOVE_SPEED = 120.0              # mm/s - how fast we ASSUME the arm travels
MOVE_BASE  = 0.6                # s - fixed overhead added to every move
THINK_TIME = 0.5                # seconds per engine move
PARK       = (200.0, 0.0, 60.0) # rest position, clear of the board

# Captured pieces get stacked out in a grid starting at the calibrated DROP spot.
GRAVE_STEP    = 28.0            # mm between dropped pieces
GRAVE_PER_ROW = 8               # how many before it starts a new row

# ---------------- GRIPPER ----------------
# pydobot forks disagree on the gripper API. Probe once at startup, then reuse.
_grip_impl = None

def _detect_gripper(a):
    global _grip_impl
    candidates = [
        ("grip",    lambda arm, on: arm.grip(on)),
        ("gripper", lambda arm, on: arm.gripper(on)),
        ("_set_end_effector_gripper",
                    lambda arm, on: arm._set_end_effector_gripper(True, on)),
        ("suck",    lambda arm, on: arm.suck(on)),
    ]
    for name, fn in candidates:
        if hasattr(a, name):
            try:
                fn(a, False)                     # test call: open / off
                _grip_impl = fn
                print(f"[Gripper] using arm.{name}()")
                return
            except Exception as e:
                print(f"[Gripper] arm.{name}() exists but failed: {e}")
    print("[Gripper] WARNING: no working method found.")

def control_claw(grab):
    if _grip_impl is None:
        return
    try:
        dobot(_grip_impl, arm, bool(grab))
    except Exception as e:
        print(f"[Gripper] command failed: {e}")
    time.sleep(1.0)                              # let the jaws physically finish

# ---------------- HARDWARE INIT ----------------
print("=" * 60)
print(VERSION)
print("=" * 60)

# Open without toggling DTR/RTS, so we don't reset the ESP32 into the bootloader.
esp = serial.Serial()
esp.port     = ESP_PORT
esp.baudrate = 115200
esp.timeout  = 0.1
esp.dtr = False
esp.rts = False
esp.open()
time.sleep(0.3)
esp.reset_input_buffer()

arm = Dobot(port=DOBOT_PORT, verbose=False)

# pydobot reads the arm's reply with a short serial timeout. If the reply is
# late it returns None, and pydobot then does None.params -> AttributeError.
# Give it more time to answer.
for attr in ("ser", "_serial", "serial"):
    s = getattr(arm, attr, None)
    if s is not None and hasattr(s, "timeout"):
        s.timeout = 2.0
        print(f"[Serial] Dobot read timeout raised to 2.0 s (arm.{attr})")
        break

arm.speed(100, 100)
_detect_gripper(arm)


def dobot(fn, *a, tries=4, **kw):
    """Call a pydobot method, retrying through the None-response hiccup."""
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except (AttributeError, TypeError) as e:
            print(f"  [Serial] Dobot didn't answer ({e}) - retry {i + 1}/{tries}")
            time.sleep(0.5)
    raise RuntimeError("Dobot stopped responding. Power-cycle the arm.")


# MOVL_XYZ = straight line in tool space.
# MOVJ (pydobot's default) interpolates the JOINTS, so the claw traces a CURVE
# through the air: Z drifts while XY moves. That breaks the axis lock below, so
# we try hard to get MOVL and complain loudly if we cannot.
_MOVL = None
if hasattr(arm, "_set_ptp_cmd"):
    for getter in (
        lambda: __import__("pydobot.enums", fromlist=["PTPMode"]).PTPMode.MOVL_XYZ,
        lambda: __import__("pydobot.enums.ptp_mode", fromlist=["PTPMode"]).PTPMode.MOVL_XYZ,
        lambda: 2,          # MOVL_XYZ is mode 2 in the Dobot protocol
    ):
        try:
            _MOVL = getter()
            print(f"[Motion] linear MOVL_XYZ enabled (mode={_MOVL})")
            break
        except Exception:
            continue

if _MOVL is None:
    print("!" * 66)
    print("[Motion] WARNING: could not enable MOVL. Falling back to MOVJ, which")
    print("         ARCS through the air - the claw will drift in Z while moving")
    print("         sideways and WILL clip pieces. Tell me if you see this.")
    print("!" * 66)


# ---------------- ENGINE + GAME STATE ----------------
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
board = chess.Board()
human_is_white = True
arm_moves_all  = False          # False = arm plays the engine's side only
_graves = {"w": 0, "b": 0}      # how many pieces are stacked in each pile

# Set on the calibration screen; written into board_config.py.
GRAVE_W = getattr(board_config, "GRAVEYARD_WHITE", None)   # captured WHITE pieces
GRAVE_B = getattr(board_config, "GRAVEYARD_BLACK", None)   # captured BLACK pieces
for _n, _g in (("white", GRAVE_W), ("black", GRAVE_B)):
    print(f"[Grave] {_n} drop spot: {_g if _g else 'NOT calibrated (will prompt you)'}")


def send(msg):
    esp.write((msg + "\n").encode())
    print(f"  -> ESP: {msg}")

# ---------------- DIFFICULTY ----------------
def set_difficulty(elo):
    """Stockfish's UCI_Elo floor is ~1320. Below that, fall back to Skill Level."""
    opt = engine.options.get("UCI_Elo")
    lo = opt.min if opt else 1320
    hi = opt.max if opt else 3190

    if elo < lo:
        skill = max(0, min(20, round((elo - 400) / 75)))   # 800 -> ~5
        engine.configure({"UCI_LimitStrength": False, "Skill Level": skill})
        print(f"[Engine] {elo} is below the UCI_Elo floor ({lo}) -> Skill Level {skill}")
    else:
        target = max(lo, min(hi, elo))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": target})
        print(f"[Engine] UCI_Elo {target}")

# ---------------- ARM ----------------
# ---------------- HARD-CODED SQUARE OVERRIDES ----------------
# Anything listed here wins over board_config.py. Use it for squares the
# interpolated map keeps getting wrong. Format: "square": (x, y, z)
OVERRIDES = {
    "e5": (209.6, 14.2, -13.9),
}


def coords(square, piece_symbol):
    if square in OVERRIDES:
        return OVERRIDES[square]
    e = board_config.SQUARE_MAP[square][piece_symbol.lower()]
    return e["x"], e["y"], e["z"]


# pydobot's pose() returns None whenever the serial read times out, which
# blows up mid-move. So we never ask the arm where it is: we remember.
_last = list(PARK)

# The plane we cross the board at: high enough to clear the tallest piece
# anywhere on the board, plus CLEARANCE.
_BOARD_TOP_Z = max(sq["p"]["z"] for sq in board_config.SQUARE_MAP.values())
TRAVEL_Z = max(_BOARD_TOP_Z + CLEARANCE, TRAVEL_Z_FLOOR)
print(f"[Motion] board top Z = {_BOARD_TOP_Z:.1f} mm  +{CLEARANCE:.0f} clearance"
      f"  ->  travel lane = {TRAVEL_Z:.1f} mm")


def _send(x, y, z):
    """
    One raw straight-line command. Nothing else talks to the arm.

    HARD RULE: a single command may change XY *or* Z, never both. If this
    assertion ever fires it means some caller tried to move diagonally, and
    I want to know about it rather than have the claw plough through a piece.
    """
    dxy = math.dist((_last[0], _last[1]), (x, y))
    dz  = abs(_last[2] - z)
    if dxy > 0.1 and dz > 0.1:
        raise AssertionError(
            f"AXIS LOCK VIOLATED: tried to move XY ({dxy:.1f} mm) and "
            f"Z ({dz:.1f} mm) in one command. This is a bug - report it.")

    dist = math.dist((_last[0], _last[1], _last[2]), (x, y, z))
    if _MOVL is not None:
        dobot(arm._set_ptp_cmd, x, y, z, 0, mode=_MOVL, wait=False)
    else:
        dobot(arm.move_to, x, y, z, 0, wait=False)   # MOVJ: arcs. Not ideal.
    time.sleep(MOVE_BASE + dist / MOVE_SPEED)
    _last[0], _last[1], _last[2] = x, y, z


# ---- the ONLY two motions allowed. Neither can produce a diagonal. ----

def move_z(z):
    """Pure vertical. XY is untouched."""
    if abs(_last[2] - z) < 0.1:
        return
    print(f"     | Z {_last[2]:.1f} -> {z:.1f}")
    _send(_last[0], _last[1], z)


def move_xy(x, y):
    """Pure horizontal, and ONLY at the travel plane. Refuses to run low."""
    if _last[2] < TRAVEL_Z - 1.0:
        move_z(TRAVEL_Z)                      # never slide sideways down low
    if math.dist((_last[0], _last[1]), (x, y)) < 0.1:
        return
    print(f"     - XY ({_last[0]:.0f},{_last[1]:.0f}) -> ({x:.0f},{y:.0f}) at Z {TRAVEL_Z:.1f}")
    _send(x, y, TRAVEL_Z)
    time.sleep(SETTLE)                        # let the arm stop swaying


def reach_ok(x, y):
    r = math.hypot(x, y)
    if r > MAX_REACH:
        print(f"  [!] ({x:.1f},{y:.1f}) is {r:.0f} mm out - past the arm's "
              f"{MAX_REACH:.0f} mm reach. It will not go down.")
        return False
    return True


def pick(x, y, z):
    move_z(TRAVEL_Z)      # 1. UP first, always
    move_xy(x, y)         # 2. ACROSS, high above everything
    move_z(z)             # 3. DOWN, straight
    time.sleep(SETTLE)
    control_claw(True)    # 4. grab
    move_z(TRAVEL_Z)      # 5. UP, straight


def place(x, y, z):
    move_z(TRAVEL_Z)      # 1. UP first, always
    move_xy(x, y)         # 2. ACROSS, high above everything
    move_z(z)             # 3. DOWN, straight
    time.sleep(SETTLE)
    control_claw(False)   # 4. release
    move_z(TRAVEL_Z)      # 5. UP, straight


def arm_move_piece(from_sq, to_sq, piece_symbol):
    fx, fy, fz = coords(from_sq, piece_symbol)
    tx, ty, tz = coords(to_sq, piece_symbol)
    print(f"  [ARM] {from_sq} -> {to_sq}")

    reach_ok(fx, fy)
    reach_ok(tx, ty)

    control_claw(False)
    pick(fx, fy, fz)
    place(tx, ty, tz)

    move_xy(PARK[0], PARK[1])     # park, still up at the travel plane


def grave_spot(colour):
    """Drop spot for a captured piece of this colour, or None if not calibrated."""
    return GRAVE_W if colour == "w" else GRAVE_B


def grave_slot(colour):
    """Next free spot in that colour's pile, so pieces don't land on each other."""
    gx, gy, gz = grave_spot(colour)
    row, col = divmod(_graves[colour], GRAVE_PER_ROW)
    return gx + row * GRAVE_STEP, gy + col * GRAVE_STEP, gz


def arm_remove_piece(square, piece_symbol):
    """Arm lifts the captured piece off and stacks it on its own colour's pile."""
    colour = "w" if piece_symbol.isupper() else "b"
    px, py, pz = coords(square, piece_symbol)
    gx, gy, gz = grave_slot(colour)
    pile = "WHITE" if colour == "w" else "BLACK"
    print(f"  [ARM] capture: {square} -> {pile} pile, slot {_graves[colour]}")

    reach_ok(px, py)
    reach_ok(gx, gy)

    control_claw(False)
    pick(px, py, pz)
    place(gx, gy, gz)
    _graves[colour] += 1


def wait_for_removal(square):
    """Ask the human to clear a captured piece; block until they tap DONE."""
    send(f"REMOVE:{square}")
    print(f"  [WAIT] clear {square.upper()} and tap DONE on the screen...")
    while True:
        line = esp.readline().decode(errors="ignore").strip()
        if line == "REMOVED":
            print(f"  [OK] {square.upper()} clear.")
            return
        if line == "RESET":
            raise KeyboardInterrupt
        time.sleep(0.02)


def victim_square(move):
    """Square the captured piece physically sits on (en passant != destination)."""
    if board.is_en_passant(move):
        f = chess.square_file(move.to_square)
        r = chess.square_rank(move.from_square)
        return chess.square_name(chess.square(f, r))
    return chess.square_name(move.to_square)


def execute_on_board(move, use_arm):
    """Run one legal move. use_arm=False means the human moves it by hand."""
    piece   = board.piece_at(move.from_square)
    symbol  = piece.symbol()
    from_sq = chess.square_name(move.from_square)
    to_sq   = chess.square_name(move.to_square)

    # ---- captures ----
    if board.is_capture(move):
        vsq = victim_square(move)
        victim = board.piece_at(chess.parse_square(vsq))
        vsym = victim.symbol() if victim else "p"

        vcolour = "w" if vsym.isupper() else "b"

        if use_arm and grave_spot(vcolour):
            arm_remove_piece(vsq, vsym)          # arm stacks it on that colour's pile
        elif use_arm:
            wait_for_removal(vsq)                # that pile isn't calibrated: ask
        # if the human is moving the pieces, they take it off themselves

    if not use_arm:
        board.push(move)
        return

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
    report_status()


def sync_screen():
    """Push the authoritative position. The screen renders THIS, never a guess."""
    send("FEN:" + board.board_fen())


def report_status():
    if board.is_checkmate():
        send("STATUS:CHECKMATE")
    elif board.is_stalemate():
        send("STATUS:STALEMATE")
    elif board.is_insufficient_material():
        send("STATUS:DRAW")


def play_engine_move():
    result = engine.play(board, chess.engine.Limit(time=THINK_TIME))
    move = result.move
    uci = move.uci()
    print(f"[Engine] plays {uci}")
    execute_on_board(move, use_arm=True)
    send(uci)          # for the "AI Played:" caption
    sync_screen()      # then overwrite the whole board with the truth


def handle_human_move(uci):
    """The human moves their own pieces by hand. We only track the position."""
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        send("ILLEGAL")
        return
    if move not in board.legal_moves:
        print(f"[Human] {uci} rejected")
        send("ILLEGAL")
        return

    if arm_moves_all:
        print(f"[Human] {uci}  (arm will move it)")
        execute_on_board(move, use_arm=True)
    else:
        print(f"[Human] {uci}  (you move it by hand)")
        execute_on_board(move, use_arm=False)
    sync_screen()
    report_status()

    if not board.is_game_over():
        play_engine_move()


# ---------------- ON-SCREEN CALIBRATION ----------------
CAL_CORNERS = ["a1", "h1", "a8", "h8"]   # "gw"/"gb" are the optional drop spots
_cal = {}                                    # square -> (x, y, z)

FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"]
PIECES = ["p", "r", "n", "b", "q", "k"]


def read_pose():
    """pydobot's pose() returns None on a serial timeout, so retry it."""
    for i in range(6):
        try:
            p = arm.pose()
            if p is not None:
                return p[0], p[1], p[2]
        except (AttributeError, TypeError):
            pass
        print(f"  [Cal] arm didn't answer, retry {i + 1}/6")
        time.sleep(0.4)
    raise RuntimeError("could not read the arm's position")


def cal_capture(square):
    square = square.lower()
    x, y, z = read_pose()

    # The arm hasn't moved since another corner was captured -> almost certainly
    # a double-tap on CAPTURE. Refuse it instead of silently ruining the map.
    for other, (ox, oy, oz) in _cal.items():
        if other != square and math.dist((x, y), (ox, oy)) < 10.0:
            print(f"[Cal] REJECTED {square}: same spot as {other} "
                  f"- did you move the arm?")
            send(f"CAL_ERR:{square} is the same spot as {other} - move the arm first")
            return

    _cal[square] = (x, y, z)
    print(f"[Cal] {square} = ({x:.1f}, {y:.1f}, {z:.1f})")
    send(f"CAL_OK:{square}:{x:.1f},{y:.1f},{z:.1f}")


def _bilinear(u, v):
    out = []
    for i in range(3):
        out.append((1 - u) * (1 - v) * _cal["a1"][i] +
                   (    u) * (1 - v) * _cal["h1"][i] +
                   (1 - u) * (    v) * _cal["a8"][i] +
                   (    u) * (    v) * _cal["h8"][i])
    return out


def cal_save():
    missing = [c for c in CAL_CORNERS if c not in _cal]
    if missing:
        send("CAL_ERR:missing " + ",".join(missing))
        return

    # sanity: opposite edges of the board should match
    d = lambda a, b: math.dist(_cal[a][:2], _cal[b][:2])
    w1, w2 = d("a1", "h1"), d("a8", "h8")
    h1_, h2 = d("a1", "a8"), d("h1", "h8")
    print(f"[Cal] edges  rank1 {w1:.1f}  rank8 {w2:.1f}  "
          f"afile {h1_:.1f}  hfile {h2:.1f}")
    if abs(w1 - w2) > 6 or abs(h1_ - h2) > 6:
        send(f"CAL_ERR:edges off by {max(abs(w1-w2), abs(h1_-h2)):.0f}mm - recheck")
        return

    L = ["# " + "=" * 66,
         "#   CHESSBOARD MAP - captured on the touchscreen (CAL_SAVE)",
         "# " + "=" * 66]
    for c in CAL_CORNERS:
        x, y, z = _cal[c]
        L.append(f"#   {c} = ({x:.1f}, {y:.1f}, {z:.1f})")
    L += ["", "SQUARE_MAP = {"]

    for r, rank in enumerate(RANKS):
        L.append(f"\n    # ---------------- RANK {rank} ----------------")
        for f, file in enumerate(FILES):
            x, y, z = _bilinear(f / 7.0, r / 7.0)
            L.append(f'    "{file}{rank}": {{')
            for pc in PIECES:
                L.append(f'        "{pc}": {{"x": {x:.1f}, "y": {y:.1f}, "z": {z:.1f}}},')
            L.append("    },")
    L += ["}", ""]

    if "gw" in _cal or "gb" in _cal:
        L += ["# Off-board spots where the arm stacks captured pieces."]
    if "gw" in _cal:
        x, y, z = _cal["gw"]
        L += [f"GRAVEYARD_WHITE = ({x:.1f}, {y:.1f}, {z:.1f})"]
    if "gb" in _cal:
        x, y, z = _cal["gb"]
        L += [f"GRAVEYARD_BLACK = ({x:.1f}, {y:.1f}, {z:.1f})"]
    L += [""]

    with open("board_config.py", "w") as fh:
        fh.write("\n".join(L))

    import importlib
    importlib.reload(board_config)

    global TRAVEL_Z, _BOARD_TOP_Z, GRAVE_W, GRAVE_B
    _BOARD_TOP_Z = max(sq["p"]["z"] for sq in board_config.SQUARE_MAP.values())
    TRAVEL_Z = max(_BOARD_TOP_Z + CLEARANCE, TRAVEL_Z_FLOOR)
    GRAVE_W = getattr(board_config, "GRAVEYARD_WHITE", None)
    GRAVE_B = getattr(board_config, "GRAVEYARD_BLACK", None)
    print(f"[Grave] white -> {GRAVE_W}   black -> {GRAVE_B}")

    print(f"[Cal] board_config.py written. New travel lane = {TRAVEL_Z:.1f} mm")
    send("CAL_SAVED")


# ---------------- MAIN LOOP ----------------
print("Host ready. Waiting for the terminal...")
try:
    move_z(TRAVEL_Z)
    move_xy(PARK[0], PARK[1])

    while True:
        line = esp.readline().decode(errors="ignore").strip()
        if not line:
            continue
        print(f"<- ESP: {line}")

        try:
            if line == "CAL_ENTER":
                _cal.clear()
                print("[Cal] calibration mode - jog the arm and tap CAPTURE")

            elif line.startswith("CAL_CAPTURE:"):
                cal_capture(line.split(":")[1].strip())

            elif line == "CAL_SAVE":
                cal_save()

            elif line == "CAL_EXIT":
                print("[Cal] leaving calibration mode")

            elif line.startswith("SET_ELO:"):
                set_difficulty(int(line.split(":")[1]))

            elif line.startswith("SET_COLOR:"):
                human_is_white = (line.split(":")[1] == "WHITE")

            elif line.startswith("SET_ARM:"):
                arm_moves_all = (line.split(":")[1].strip() == "ALL")
                print(f"[Mode] arm moves "
                      f"{'BOTH sides' if arm_moves_all else 'the engine only'}")

            elif line == "START_MATCH":
                board.reset()
                _graves["w"] = _graves["b"] = 0
                sync_screen()
                if not human_is_white:
                    play_engine_move()           # engine opens as White

            elif line == "RESET":
                board.reset()

            elif len(line) >= 4 and line[0] in "abcdefgh":
                handle_human_move(line)

        except Exception as e:
            # One bad line shouldn't kill the game and strand the arm mid-air.
            print(f"[Error] {type(e).__name__}: {e}")

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