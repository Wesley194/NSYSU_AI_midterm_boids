import json
import initial
def read_Setting():
    try:
        with open("data/setting.json", "r") as f:
            Setting = json.load(f)
    except FileNotFoundError:
        print("""Can not find "data/setting.json". Create new one.""")
        initial.create_default()
        with open("data/setting.json", "r") as f:
            Setting = json.load(f)
    return Setting

if __name__ == "__main__":
    read_Setting()