import json
import initial
def read_Setting():
    try:
        with open("data/setting.json", "r") as f:
            Setting = json.load(f)
    except FileNotFoundError:
        initial.setGlobal()
        with open("data/setting.json", "r") as f:
            Setting = json.load(f)
    return Setting