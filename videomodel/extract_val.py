import json
from extract import instance_videos


with open("./WLASL_v0.3.json") as file: 
    data = json.load(file)

for datum in data:
    instance_videos(datum["gloss"], datum["instances"], "val")