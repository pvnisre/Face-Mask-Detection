import json
import plotly.express as px

with open("static/logs/mask_violations.json") as f:
    data = json.load(f)

lats = [entry['latitude'] for entry in data]
lons = [entry['longitude'] for entry in data]

fig = px.density_mapbox(
    lat=lats,
    lon=lons,
    center={"lat": 13.630, "lon": 79.420},
    zoom=13,
    radius=30,
    mapbox_style="stamen-terrain"
)

fig.show()
