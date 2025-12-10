import json

file_Name = [
    """data/record/record_2025-12-10_15-38-50.json""",
    """data/record/record_2025-12-10_15-50-48.json""",
]

output_path = """data/record/combine.json"""

Combine = []
time_offset = 0
for name in file_Name:
    with open(name, "r") as f:
        data = json.load(f)
    if len(data)>0:
        for i in data:
            i["time"]+=time_offset
        time_offset = data[-1]["time"]
        Combine += data
with open(output_path, "w") as f:
        json.dump(Combine, f)