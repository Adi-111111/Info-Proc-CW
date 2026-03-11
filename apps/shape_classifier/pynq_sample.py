import json
from preprocess import preprocess_to_vector

# load raw stroke JSON
with open("dataset/circle/circle_1772893918629.json", "r") as f:
    data = json.load(f)

stroke = data["stroke"]

# run YOUR training preprocessing
features = preprocess_to_vector(stroke)

print("Feature length:", len(features))

# save processed features
out = {
    "features": features
}

with open("triangle_features.json", "w") as f:
    json.dump(out, f)

print(len(features))
print(features[:10])

print("Saved to triangle_features.json")