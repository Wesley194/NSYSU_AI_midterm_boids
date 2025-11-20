import json
import threading
import control_window

def MAIN():
    # load setting
    with open("data/setting.json", "r") as f:
        Setting = json.load(f)
    control_window.set_tkinter(Setting)

if __name__ == "__main__":
    MAIN()