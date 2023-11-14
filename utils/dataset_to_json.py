import os
import json

# Initialize COCO json format
coco_json = {
    "images": [],
    "annotations": [],
    "categories": [],
    }

# Paths
img_dir = '../ExDark/ExDark'
box_dir = '../ExDark_Annno/ExDark_Annno'
coc_dir = '../ExDark_COCO'

# Categories (Subfolders)
categories = os.listdir(img_dir)
for i, c in enumerate(categories):
    coco_json["categories"].append({"id": i, 
                                    "name": c})

# Add Images & Annotations
xx = 0
anno_id = 0
for i, c in enumerate(categories):

    print(c)

    category_id = i
    cat_dir = os.path.join(img_dir, c)

    for j, img in enumerate(os.listdir(cat_dir)):

        # Add Image
        coco_json["images"].append({"id": j, 
                                    "file_name": img})
        
        # Add Annotation
        with open(os.path.join(box_dir, c, img + '.txt'), 'r') as f:
            lines = f.readlines()[1:]
            for l in lines:
                coco_json['annotations'].append({"id": anno_id,
                                                 "image_id": j,
                                                 "category_id": category_id,
                                                 "bbox": l.split(' ')[1:5]})
                anno_id += 1

# Save json file
json.dump(coco_json, open(os.path.join(coc_dir, 'coco_anno_format.json'), 'w'))