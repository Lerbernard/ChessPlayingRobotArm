"""
Dry run of the GAME logic - no arm, no camera, no Stockfish, but the real
python-chess. It checks the two things that used to strand a game:

  1. a human move the camera diff missed (or rejected and then absorbed into
     its baseline) can still be read straight off the board, by asking which
     legal move makes the position look like what the camera sees;
  2. an engine reply that refuses to play on a board the camera disputes can
     be unblocked - by fixing the board, or by insisting.

    python3 test_game_logic.py
"""

import sys
import types
from unittest import mock


def install_fakes():
    """Hardware stubs; chess stays real, only the Stockfish process is faked."""
    import test_arm_path as T
    T.install_fakes()
    for name in ("chess", "chess.engine"):
        sys.modules.pop(name, None)      # test_arm_path mocks chess - undo that
    import chess
    import chess.engine
    fake_engine = mock.MagicMock()
    fake_engine.options = {}
    chess.engine.SimpleEngine.popen_uci = mock.MagicMock(return_value=fake_engine)
    return fake_engine


def looks_like(chess_pi, chessboard):
    """A camera snapshot that shows exactly this position."""
    return dict(chess_pi.expected_colours(chessboard))


def main():
    fake_engine = install_fakes()
    import chess
    import chess_pi

    chess_pi.time.sleep = lambda *a: None
    chess_pi.piece_colour = lambda sig: sig       # snapshots are colour words
    chess_pi._request = lambda action, timeout=0: True   # no camera thread
    sent = []
    chess_pi.send = lambda m: sent.append(m)

    failures = []

    def check(cond, what):
        print(("  ok   " if cond else "  FAIL ") + what)
        if not cond:
            failures.append(what)

    # ---------------- 1. reading a move off the board ----------------
    print("\n=== infer_human_move ===")
    b = chess.Board()
    seen = looks_like(chess_pi, b)
    check(chess_pi.infer_human_move(seen, b) == (None, "unchanged"),
          "start position, nothing moved -> unchanged")

    after = b.copy(); after.push_uci("e2e4")
    check(chess_pi.infer_human_move(looks_like(chess_pi, after), b) == ("e2e4", "move"),
          "e2e4 played -> e2e4")

    noisy = looks_like(chess_pi, after); noisy["h8"] = "none"   # one bad square
    check(chess_pi.infer_human_move(noisy, b) == ("e2e4", "move"),
          "e2e4 with one misread square -> still e2e4")

    # a capture, mid-game
    b2 = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    after2 = b2.copy(); after2.push_uci("e4d5")
    check(chess_pi.infer_human_move(looks_like(chess_pi, after2), b2) == ("e4d5", "move"),
          "capture exd5 -> e4d5")

    # castling: king AND rook moved
    b3 = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    after3 = b3.copy(); after3.push_uci("e1g1")
    check(chess_pi.infer_human_move(looks_like(chess_pi, after3), b3) == ("e1g1", "move"),
          "kingside castle -> e1g1")

    # garbage: half the board wrong
    junk = {sq: "green" for sq in seen}
    check(chess_pi.infer_human_move(junk, b)[1] == "unclear",
          "nonsense picture -> unclear, nothing guessed")

    # ---------------- 2. the engine lockout ----------------
    print("\n=== engine lockout ===")
    chess_pi.board.reset()
    chess_pi.human_is_white = True
    chess_pi.board.push_uci("e2e4")                # human moved; engine to reply
    fake_engine.play.return_value = types.SimpleNamespace(
        move=chess.Move.from_uci("e7e5"))
    chess_pi._pending["count"] = 0
    chess_pi._last[:] = [205.1, -5.4, 120.0]

    # camera shows a board with two extra wrong squares -> refuse, say which
    wrong = looks_like(chess_pi, chess_pi.board)
    wrong["a3"] = "green"; wrong["h6"] = "blue"
    chess_pi._baseline["snap"] = wrong
    sent.clear()
    chess_pi.engine_play()
    check(chess_pi._pending["count"] == 1, "disputed board -> engine waits (count 1)")
    check(any(m.startswith("MSG:Fix ") and "a3" in m for m in sent),
          f"screen told which squares: {[m for m in sent if m.startswith('MSG:')]}")
    check(all(len(m) - 4 <= chess_pi.MSG_MAX for m in sent if m.startswith("MSG:")),
          "message fits the screen")
    check(chess_pi.board.turn == chess.BLACK, "engine has NOT moved")

    # the human fixes the board -> a matching snapshot unblocks it
    chess_pi._baseline["snap"] = looks_like(chess_pi, chess_pi.board)
    chess_pi.engine_play()
    check(chess_pi._pending["count"] == 0 and chess_pi.board.turn == chess.WHITE,
          "board fixed -> engine replied e7e5")

    # or the human insists: second END TURN forces it
    chess_pi.board.push_uci("g1f3")
    fake_engine.play.return_value = types.SimpleNamespace(
        move=chess.Move.from_uci("b8c6"))
    chess_pi._baseline["snap"] = wrong
    chess_pi.engine_play()                         # refused, count 1
    chess_pi.engine_play()                         # refused, count 2
    check(chess_pi._pending["count"] == 2 and any("force" in m.lower() for m in sent),
          "second refusal offers the override")
    chess_pi.engine_play(force=chess_pi._pending["count"] >= 2)
    check(chess_pi._pending["count"] == 0 and chess_pi.board.turn == chess.WHITE,
          "forced -> engine played b8c6 anyway")

    # ---------------- 3. a capture never stacks pieces ----------------
    print("\n=== capture: the square must be empty before the mover goes down ===")
    import test_arm_path as T

    def setup_capture():
        # after 1.e4 d5 - the engine (black) can take: d5xe4
        chess_pi.board.set_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2")
        chess_pi.human_is_white = True
        fake_engine.play.return_value = types.SimpleNamespace(
            move=chess.Move.from_uci("d5e4"))
        chess_pi._pending["count"] = 0
        chess_pi._last[:] = [205.1, -5.4, 120.0]
        chess_pi._claw["closed"] = None
        T.MOVES.clear()
        sent.clear()

    def closes():
        return sum(1 for m in T.MOVES if m[0] == "grip" and m[1] is True)

    before = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2")

    # A: the arm takes the pawn off e4 and the camera agrees -> normal capture
    setup_capture()
    cleared = looks_like(chess_pi, before); cleared["e4"] = "none"   # pawn gone
    chess_pi._baseline["snap"] = cleared
    chess_pi.engine_play()
    check(not any(m.startswith("REMOVE:") for m in sent) and closes() == 2,
          "arm cleared e4, camera agrees -> no prompt, mover placed")

    # B: the arm "removed" it but the camera still sees the pawn -> ask the human
    setup_capture()
    chess_pi._baseline["snap"] = looks_like(chess_pi, before)  # e4 still green
    chess_pi.esp.readline = lambda: b"REMOVED\n"
    chess_pi.engine_play()
    check("REMOVE:e4" in sent, "grab missed -> screen asked to take e4 off")
    check(closes() == 2 and chess_pi.board.turn == chess.WHITE,
          "after DONE the mover was placed and the move recorded")

    # C: nobody clears it -> record the move but do NOT put the mover on top
    setup_capture()
    chess_pi._baseline["snap"] = looks_like(chess_pi, before)
    chess_pi.esp.readline = lambda: b""
    real_ask = chess_pi.ask_human_to_clear
    chess_pi.ask_human_to_clear = lambda sq, timeout=0.0: real_ask(sq, timeout=0.0)
    chess_pi.engine_play()
    chess_pi.ask_human_to_clear = real_ask
    check(closes() == 1, "no answer -> the mover was NOT placed on the occupied square")
    check(any("by hand" in m for m in sent), f"asked for the move by hand: {sent[-3:]}")

    # D: RESET tapped on the prompt -> the game is abandoned cleanly
    setup_capture()
    chess_pi._baseline["snap"] = looks_like(chess_pi, before)
    chess_pi.esp.readline = lambda: b"RESET\n"
    try:
        chess_pi.engine_play()
        check(False, "RESET during the prompt should raise GameReset")
    except chess_pi.GameReset:
        check(closes() == 1, "RESET -> stopped before placing anything")
    chess_pi.esp.readline = lambda: b""

    print()
    if failures:
        print(f"{len(failures)} FAILED")
        return 1
    print("OK: moves can be read off the board, the lockout has two ways out, "
          "and a capture never stacks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
