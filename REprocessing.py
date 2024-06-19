import json, os
def distancebetween(point1, point2): return (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2
data = []
for file in os.listdir("C:/Users/tahir/Desktop/HANDS"):
    if file.endswith(".json") and not "normalised" in file:
        shapes_data = json.load(open(file))["shapes"]
        thumb_points = shapes_data[0 if shapes_data[0]["label"] == "thumb" else 1]["points"]
        thumb_endpoints = [thumb_points[0], thumb_points[-1]]
        index_points = shapes_data[0 if shapes_data[0]["label"] == "index" else 1]["points"]
        index_endpoints = [index_points[0], index_points[-1]]
        thumb_startpoint_i = None
        index_startpoint_i = None
        min_distance = float("inf")
        for i in range(len(thumb_endpoints)):
            for j in range(len(index_endpoints)):
                distance = distancebetween(thumb_endpoints[i], index_endpoints[j])
                if distance < min_distance:
                    min_distance = distance
                    thumb_startpoint_i = i
                    index_startpoint_i = j
        if thumb_startpoint_i != 0: # invert order
            thumb_points = thumb_points[::-1]
        if index_startpoint_i != 0:
            index_points = index_points[::-1]
        thumb_points = [(point[0]/1920, point[1]/1080) for point in thumb_points]
        index_points = [(point[0]/1920, point[1]/1080) for point in index_points]
        data.append({"imagePath": file[:file.rindex(".")] + ".jpg",
                     "targets": thumb_points + index_points})
json.dump(data, open("normalised3.json", "w"))