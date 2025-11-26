import json
import initial
def read_Setting(file_name = "default_setting"):
    try:
        with open("data/%s.json" % file_name, "r") as f:
            Setting = json.load(f)
    except FileNotFoundError:
        print("""Can not find "data/%s.json" . Create new one.""" % file_name)
        initial.create_default()
        with open("data/default_setting.json", "r") as f:
            Setting = json.load(f)
    return Setting

if __name__ == "__main__":
    read_Setting()