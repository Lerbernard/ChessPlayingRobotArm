import sys
import time
from pydobot import Dobot

# Import your calibrated map file
try:
    import board_config
except ImportError:
    print("[Error] board_config.py not found in this directory!")
    sys.exit()

DOBOT_PORT = 'COM4'   # Verify this matches your setup
SAFE_Z_OFFSET = 40.0  # Height (mm) to lift above the piece during transit

print("[Test] Connecting to Dobot Magician...")
try:
    device = Dobot(port=DOBOT_PORT, verbose=False)
    print("[Test] Connected! Homing to standby...")
    device.move_to(200, 0, 50, 0, wait=True)
    
    # Ensure claw starts open
    device.grip(False)
except Exception as e:
    print(f"[Hardware Error] Connection failed: {e}")
    sys.exit()

def control_claw(grab=True):
    """
    Controls the physical robotic claw attachment using native pydobot wrappers.
    grab=True  -> Closes the claw around the piece
    grab=False -> Opens the claw to release the piece
    """
    if grab:
        print("  [Claw] Closing gripper...")
        device.grip(True)
    else:
        print("  [Claw] Opening gripper...")
        device.grip(False)
    time.sleep(1.0)  # Complete physical jaw transit delay

def test_pawn_move(from_sq, to_sq):
    """Simulates a pick-and-place operation specifically using claw actuation."""
    print(f"\n--- Testing Pawn Transit: {from_sq.upper()} -> {to_sq.upper()} ---")
    
    from_pawn = board_config.SQUARE_MAP[from_sq]['p']
    to_pawn = board_config.SQUARE_MAP[to_sq]['p']
    
    fx, fy, fz = from_pawn["x"], from_pawn["y"], from_pawn["z"]
    tx, ty, tz = to_pawn["x"], to_pawn["y"], to_pawn["z"]
    
    if (fx == 0.0 and fy == 0.0) or (tx == 0.0 and ty == 0.0):
        print(f"[Skip] Either {from_sq.upper()} or {to_sq.upper()} hasn't been calibrated yet!")
        return

    # --- EXECUTION PIPELINE ---
    
    # 1. Approach: Open claw and hover above source pawn
    control_claw(grab=False)
    device.move_to(fx, fy, fz + SAFE_Z_OFFSET, 0, wait=True)
    
    # 2. Descend: Move down over the piece body
    device.move_to(fx, fy, fz, 0, wait=True)
    
    # 3. Grip: Close the claw tightly onto the piece cap
    control_claw(grab=True)
    
    # 4. Lift: Ascend vertically with the piece secure
    device.move_to(fx, fy, fz + SAFE_Z_OFFSET, 0, wait=True)
    
    # 5. Travel: Move over destination square safely
    device.move_to(tx, ty, tz + SAFE_Z_OFFSET, 0, wait=True)
    
    # 6. Drop: Lower the piece until it meets the surface line
    device.move_to(tx, ty, tz, 0, wait=True)
    
    # 7. Release: Open the claw to let go of the piece
    control_claw(grab=False)
    
    # 8. Clear: Lift clear up to finish move cleanly without knocking it over
    device.move_to(tx, ty, tz + SAFE_Z_OFFSET, 0, wait=True)
    print("Move Complete!")

try:
    print("\nStarting Claw Validation Test Run...")
    
    # Run test sequence (e2 -> e4 and back)
    test_pawn_move("e2", "e4")
    time.sleep(2.0)
    test_pawn_move("e4", "e2") 

    print("\n[Success] Test routine finished. Returning to standby.")
    device.move_to(200, 0, 50, 0, wait=True)

except KeyboardInterrupt:
    print("\n[Test] Interrupted by user.")
finally:
    # Open gripper and release control lines cleanly upon program exit
    try:
        device.grip(False)
    except Exception:
        pass
    device.close()
    print("Dobot connection closed cleanly.")