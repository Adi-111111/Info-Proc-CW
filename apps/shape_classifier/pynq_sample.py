import json
from preprocess import preprocess_to_vector

# load raw stroke JSON
with open("dataset/rectangle/circle_1772898438729.json", "r") as f:
    data = json.load(f)

stroke = data["stroke"]

# run YOUR training preprocessing
features = preprocess_to_vector(stroke)

print("Feature length:", len(features))

# save processed features
out = {
    "features": features
}

with open("rectangle_features.json", "w") as f:
    json.dump(out, f)

print(len(features))
print(features[:10])

print("Saved to rectangle_features.json")