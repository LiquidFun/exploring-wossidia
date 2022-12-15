import requests

data = requests.get("http://api.wossidia.de/nodes/am_place")
with open("am_place.json", "wb") as file:
    file.write(data.content)

