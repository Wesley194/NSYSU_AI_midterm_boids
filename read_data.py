import json
import initial
from datetime import datetime

def read_Setting(file_name = "data/default_setting.json"):
    if ".json" in file_name:
        try:
            with open(file_name, "r") as f:
                Setting = json.load(f)
                if Setting.get("verify","")=="Setting" :return Setting,file_name
        except FileNotFoundError:
            pass
    print(f'Can not find "{file_name}" or the format error. Create new one.')
    initial.create_default()
    file_name = initial.create_default()
    with open(file_name, "r") as f:
        Setting = json.load(f)
    return Setting,file_name

def read_Setting_no_default(file_name):
    Setting={}
    if ".json" in file_name:
        try:
            with open(file_name, "r") as f:
                Setting = json.load(f)
        except FileNotFoundError:
            pass
    if Setting.get("verify","")=="Setting": return Setting
    else : return None

def read_OLD(file_name = None):
    if file_name is None or ".json" not in file_name:
        return None,"No load any file"
    else:
        try:
            with open(file_name, "r") as f:
                data = json.load(f)
                if data.get("verify","")=="OLD": return data,file_name
        except FileNotFoundError:
            pass
        print(f'Can not find "{file_name}" or the format error.')
        return None,"No load any file"

def save_OLD(data,file_name = ""):
    if ".json" not in file_name:
        file_name = f"data/record_at_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(file_name, "w") as f:
            json.dump(data, f)