import sys
import time
import msvcrt
from pydobot import Dobot

PORT = "COM4"  # Make sure this matches your Dobot COM port
CONFIG_FILE = "board_config.py"

PIECE_LABELS = {
    'p': "Pawn (p)",
    'r': "Rook (r)",
    'n': "Knight (n)",
    'b': "Bishop (b)",
    'q': "Queen (q)",
    'k': "King (k)"
}

print(f"Connecting to Dobot Magician on {PORT}...")
try:
    device = Dobot(port=PORT, verbose=False)
    print("Connected successfully!")
except Exception as e:
    print(f"[ERROR] Could not connect to the arm: {e}")
    sys.exit()

print("\n[!] Unlocking arm motors. Move the arm freely.")
device.speed(100, 100)
device.suck(False)

def update_config_file(square, piece, x, y, z):
    """Safely finds and overwrites the exact line for a specific square and piece."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            lines = f.readlines()
        
        inside_target_square = False
        updated = False
        
        for i, line in enumerate(lines):
            # Check if we are entering our target square block
            if f'"{square}":' in line:
                inside_target_square = True
                continue
            
            # If inside the block, find the specific piece line
            if inside_target_square:
                if f'"{piece}":' in line:
                    # Construct the perfectly formatted replacement line
                    lines[i] = f'        "{piece}": {{"x": {x:.1f}, "y": {y:.1f}, "z": {z:.1f}}},\n'
                    updated = True
                    break
                # If we hit another square identifier before finding the piece, break safety check
                if '": {' in line and not line.strip().startswith(f'"{piece}":'):
                    inside_target_square = False
                    
        if not updated:
            print(f"\n[File Error] Could not find the structure for square '{square}' piece '{piece}' in {CONFIG_FILE}.")
            return False

        with open(CONFIG_FILE, 'w') as f:
            f.writelines(lines)
        return True
    except Exception as e:
        print(f"\n[File Error] Failed to write to config file: {e}")
        return False

# --- RE-ORDERED: PIECE-FIRST ITERATION ---
pieces = ['p', 'r', 'n', 'b', 'q', 'k']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# Generate queue: All pawns across every square, then all rooks, etc.
calibration_queue = []
for piece in pieces:
    for rank in ranks:
        for file in files:
            square = f"{file}{rank}"
            calibration_queue.append((square, piece))

# Dynamic resume check
try:
    import board_config
    # Reloading module to pick up manual modifications if restarted
    import importlib
    importlib.reload(board_config)
    
    start_index = 0
    for idx, (sq, pc) in enumerate(calibration_queue):
        vals = board_config.SQUARE_MAP[sq][pc]
        if vals["x"] == 0.0 and vals["y"] == 0.0 and vals["z"] == 0.0:
            start_index = idx
            break
    print(f"\n[System] Found progress! Resuming at item {start_index + 1}/{len(calibration_queue)}")
except Exception as e:
    start_index = 0
    print("\n[System] Starting fresh from the beginning.")

print("\n=================== AUTO-MAPPING INTERFACE ===================")
print("  Instructions:")
print("    1. Position arm onto the targeted square/piece setup.")
print("    2. Press [ SPACEBAR ] to save instantly and advance.")
print("    3. Press [ ESC ] to stop and save your session.")
print("==============================================================")

current_idx = start_index

try:
    while current_idx < len(calibration_queue):
        target_square, target_piece = calibration_queue[current_idx]
        piece_fullname = PIECE_LABELS[target_piece]
        
        print(f"\r==> Align arm for piece: {piece_fullname.upper()} on square: [{target_square.upper()}] and press [SPACEBAR]...", end="", flush=True)
        
        if msvcrt.kbhit():
            key = msvcrt.getch()
            
            if key == b'\x1b':  # ESC
                print("\n\nCalibration paused safely. You can pick up right here next time!")
                break
                
            if key == b' ':  # Spacebar
                x, y, z = device.pose()[:3]
                
                success = update_config_file(target_square, target_piece, x, y, z)
                if success:
                    print(f"\n Saved! [{target_square.upper()}] {target_piece} -> X:{x:.1f}, Y:{y:.1f}, Z:{z:.1f}")
                    current_idx += 1
                    time.sleep(0.4)  # Anti-bounce delay
                    
        time.sleep(0.02)

    if current_idx >= len(calibration_queue):
        print("\n\n[Success] Complete mapping profile generated!")

except KeyboardInterrupt:
    print("\nScript terminated.")
finally:
    device.close()