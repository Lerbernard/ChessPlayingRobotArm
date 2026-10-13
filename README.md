force merge, I dare you

## Updating the Pi

Send the Pi script from the laptop (PowerShell), then run it:

    scp C:\Users\leema\Downloads\chess_pi.py viper-lab-chess@10.100.98.242:~/
    ssh viper-lab-chess@10.100.98.242
    python3 chess_pi.py

chess_pi.py lives in the Pi's home folder (~), next to board_config.py.
The touchscreen sketch is flashed from the Arduino IDE on the laptop, not sent to the Pi.
