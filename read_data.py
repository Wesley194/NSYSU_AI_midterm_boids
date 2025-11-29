import json
import initial
def read_Setting(file_name = "data/default_setting.json"):
    try:
        with open(file_name, "r") as f:
            Setting = json.load(f)
    except FileNotFoundError:
        print(f"Can not find {file_name} . Create new one.")
        initial.create_default()
        file_name = initial.create_default()
        with open(file_name, "r") as f:
            Setting = json.load(f)
    return Setting,file_name

def read_Setting_no_default(file_name):
    try:
        with open(file_name, "r") as f:
            Setting = json.load(f)
    except FileNotFoundError:
        return None
    return Setting

def read_OLD(file_name = None):
    if file_name is None:
        return None,"No load any file"
    else:
        try:
            with open(file_name, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Can not find {file_name}")
            return None,"No load any file"
        return data,file_name
if __name__ == "__main__":
    read_Setting()