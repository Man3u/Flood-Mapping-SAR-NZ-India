import ee

ee.Initialize(project="floodmapping-506505")

AOI = ee.Geometry.Rectangle([78.30, 17.20, 78.65, 17.50])

FLOOD_DATE = ee.Date("2024-09-01")
PRE_WINDOW_START = FLOOD_DATE.advance(-40, "day")
POST_WINDOW_END = FLOOD_DATE.advance(12, "day")

s1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(AOI)
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
)

post_collection = s1.filterDate(FLOOD_DATE, POST_WINDOW_END).sort("system:time_start", True)
post_count = post_collection.size().getInfo()
print(f"Found {post_count} post-flood images")
if post_count == 0:
    raise SystemExit("No post-flood images found - widen POST_WINDOW_END and retry")

post_image = ee.Image(post_collection.first())
post_date = post_image.date().format("YYYY-MM-dd").getInfo()
post_orbit = post_image.get("orbitProperties_pass").getInfo()
print(f"Post-flood image date: {post_date}, orbit pass: {post_orbit}")

pre_collection = (
    s1.filterDate(PRE_WINDOW_START, FLOOD_DATE)
    .filter(ee.Filter.eq("orbitProperties_pass", post_orbit))
    .sort("system:time_start", False)
)
pre_count = pre_collection.size().getInfo()
print(f"Found {pre_count} pre-flood images matching orbit pass {post_orbit}")
if pre_count == 0:
    raise SystemExit(f"No pre-flood images on matching orbit ({post_orbit}) - widen PRE_WINDOW_START and retry")

pre_image = ee.Image(pre_collection.first())
pre_date = pre_image.date().format("YYYY-MM-dd").getInfo()
print(f"Pre-flood image date: {pre_date}, orbit pass: {post_orbit} (matched)")

pre_vv_vh = pre_image.select(["VV", "VH"]).clip(AOI)
post_vv_vh = post_image.select(["VV", "VH"]).clip(AOI)

ee.batch.Export.image.toDrive(
    image=pre_vv_vh,
    description="Hyderabad_S1_PreFlood",
    folder="FloodMappingExports",
    fileNamePrefix=f"hyderabad_preflood_{pre_date}",
    region=AOI,
    scale=10,
    crs="EPSG:4326",
    maxPixels=1e10,
).start()

ee.batch.Export.image.toDrive(
    image=post_vv_vh,
    description="Hyderabad_S1_PostFlood",
    folder="FloodMappingExports",
    fileNamePrefix=f"hyderabad_postflood_{post_date}",
    region=AOI,
    scale=10,
    crs="EPSG:4326",
    maxPixels=1e10,
).start()

print("Both exports started (orbit-matched). Check https://code.earthengine.google.com/tasks")
