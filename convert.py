import pandas as pd
import os
base_path = "/Users/kavi/Downloads/human_detection_in_classroom_zipped/test"
csv_file = os.path.join(base_path, "_annotations.csv")

labels_path = os.path.join(base_path, "labels")
os.makedirs(labels_path, exist_ok=True)

df = pd.read_csv(csv_file)

# 🔥 IMPORTANT: "Person" (capital P)
class_map = {"Person": 0}

for _, row in df.iterrows():
    filename = row['filename']
    width = row['width']
    height = row['height']
    
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']
    
    # YOLO format
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    
    class_id = class_map[row['class']]
    
    label_file = os.path.join(labels_path, filename.replace(".jpg", ".txt"))
    
    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")
